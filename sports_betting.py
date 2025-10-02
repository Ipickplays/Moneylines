import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
import time
import os
from collections import Counter

# --- API Key ---
ODDS_API_KEY = '6dcf1fafc93b0e7f96353ed3e29bd718'

# --- League Config ---
leagues = {
    'NBA': {'sport_key': 'basketball_nba', 'csv': 'nba_data.csv'},
    'NFL': {'sport_key': 'americanfootball_nfl', 'csv': 'nfl_data.csv'},
    'NHL': {'sport_key': 'icehockey_nhl', 'csv': 'nhl_data.csv'},
    'MLB': {'sport_key': 'baseball_mlb', 'csv': 'mlb_data.csv'}
}

# --- Record CSVs ---
record_csvs = {'all_time': 'all_time_record.csv'}
for league in leagues:
    record_csvs[league] = f"{league.lower()}_record.csv"

MAX_API_RETRIES = 3
today = datetime.now().strftime('%Y-%m-%d')
yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
predictions = []

# --- Helper: mode ---
def most_common(lst):
    if not lst:
        return None
    counts = Counter(lst)
    return counts.most_common(1)[0][0]

# --- Fetch historical games ---
def fetch_historical_games(path):
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, encoding='latin1', on_bad_lines='skip')
            df = df[pd.to_datetime(df['date'], errors='coerce').notna()]
            return df
        except Exception as e:
            print(f"Error reading {path}: {e}")
    return pd.DataFrame(columns=['date','home_team','away_team','outcome','home_score','away_score'])

# --- Build team stats ---
def build_team_stats(df):
    team_stats = {}
    h2h_stats = {}
    h2h_history = {}
    for row in df.to_dict('records'):
        home = row['home_team']
        away = row['away_team']
        outcome = row['outcome']
        home_score = row.get('home_score',0)
        away_score = row.get('away_score',0)

        for team in [home, away]:
            if team not in team_stats:
                team_stats[team] = {'form':[], 'last_game':None, 'points_scored':[], 'points_allowed':[]}

        team_stats[home]['form'].append(1 if outcome==1 else 0)
        team_stats[away]['form'].append(0 if outcome==1 else 1)
        team_stats[home]['points_scored'].append(home_score)
        team_stats[home]['points_allowed'].append(away_score)
        team_stats[away]['points_scored'].append(away_score)
        team_stats[away]['points_allowed'].append(home_score)
        team_stats[home]['last_game'] = row['date']
        team_stats[away]['last_game'] = row['date']

        pair = tuple(sorted([home, away]))
        if pair not in h2h_stats:
            h2h_stats[pair] = {'home_wins':0,'away_wins':0}
            h2h_history[pair] = []
        if outcome==1:
            h2h_stats[pair]['home_wins'] += 1
            h2h_history[pair].append(home)
        else:
            h2h_stats[pair]['away_wins'] += 1
            h2h_history[pair].append(away)

    return team_stats, h2h_stats, h2h_history

# --- Train model ---
def train_model(df, team_stats, h2h_stats):
    X, y = [], []
    for row in df.to_dict('records'):
        home = row['home_team']
        away = row['away_team']

        try:
            last_game_home = team_stats.get(home, {'last_game': today})['last_game']
            last_game_away = team_stats.get(away, {'last_game': today})['last_game']
            rest_days_home = (datetime.now() - datetime.strptime(last_game_home, "%Y-%m-%d")).days
            rest_days_away = (datetime.now() - datetime.strptime(last_game_away, "%Y-%m-%d")).days
        except:
            rest_days_home, rest_days_away = 0, 0

        home_form = sum(team_stats.get(home,{'form':[3]})['form'][-5:])
        away_form = sum(team_stats.get(away,{'form':[3]})['form'][-5:])
        home_avg_points = np.mean(team_stats.get(home,{'points_scored':[0]})['points_scored'][-15:])
        away_avg_points = np.mean(team_stats.get(away,{'points_scored':[0]})['points_scored'][-15:])
        home_allowed = np.mean(team_stats.get(home,{'points_allowed':[0]})['points_allowed'][-15:])
        away_allowed = np.mean(team_stats.get(away,{'points_allowed':[0]})['points_allowed'][-15:])
        pair = tuple(sorted([home, away]))
        h2h_home = h2h_stats.get(pair,{'home_wins':0})['home_wins']
        h2h_away = h2h_stats.get(pair,{'away_wins':0})['away_wins']
        home_advantage = 1

        X.append([home_form, away_form, home_avg_points, away_avg_points,
                  home_allowed, away_allowed, h2h_home, h2h_away,
                  home_advantage, rest_days_home, rest_days_away])
        y.append(row['outcome'])

    clf = RandomForestClassifier(n_estimators=150, random_state=42)
    if X and y:
        clf.fit(X, y)
    return clf

# --- Fetch odds/outcomes from Odds API ---
def fetch_odds(sport_key, target_day):
    for attempt in range(MAX_API_RETRIES):
        try:
            url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/?apiKey={ODDS_API_KEY}&regions=us&markets=h2h&oddsFormat=american&dateFormat=iso&commenceTimeFrom={target_day}T00:00:00Z&commenceTimeTo={target_day}T23:59:59Z"
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"Attempt {attempt+1} failed for {sport_key}: {e}")
            time.sleep(1)
    return []

# --- Ensure record CSVs exist ---
def init_record_csv(path):
    if not os.path.exists(path):
        df = pd.DataFrame(columns=['date','matchup','predicted_winner','actual_winner','result','home_score','away_score'])
        df.to_csv(path, index=False)

for path in record_csvs.values():
    init_record_csv(path)

# --- Reset CSV for offseason leagues (keep headers only + backup) ---
for league, cfg in leagues.items():
    if os.path.exists(cfg['csv']):
        try:
            hist_df = pd.read_csv(cfg['csv'], on_bad_lines='skip')
            hist_df = hist_df[pd.to_datetime(hist_df['date'], errors='coerce').notna()]
            if not hist_df.empty:
                last_game_date = max(pd.to_datetime(hist_df['date'], errors='coerce'))
                days_since_last_game = (datetime.now() - last_game_date).days
                if days_since_last_game > 30:
                    backup_path = cfg['csv'].replace('.csv','_backup.csv')
                    hist_df.to_csv(backup_path, index=False)
                    hist_df.iloc[0:0].to_csv(cfg['csv'], index=False)
                    print(f"{cfg['csv']} wiped (offseason), headers kept. Backup saved.")
        except Exception as e:
            print(f"Error checking {cfg['csv']}: {e}")

# --- Function to update yesterday's predictions ---
def update_yesterday_records():
    for league, cfg in leagues.items():
        record_path = record_csvs[league]
        df_record = pd.read_csv(record_path)
        updated = False
        for idx, row in df_record.iterrows():
            if row['date'] == yesterday and pd.isna(row['actual_winner']):
                # Fetch actual scores from Odds API or other reliable API
                games = fetch_odds(cfg['sport_key'], yesterday)
                for g in games:
                    if f"{g['away_team']} @ {g['home_team']}" == row['matchup']:
                        # Get actual winner based on score
                        home_score = g.get('home_score')
                        away_score = g.get('away_score')
                        if home_score is not None and away_score is not None:
                            actual_winner = g['home_team'] if home_score>away_score else g['away_team']
                            result = 'Win' if row['predicted_winner']==actual_winner else 'Loss'
                            df_record.at[idx,'actual_winner'] = actual_winner
                            df_record.at[idx,'result'] = result
                            df_record.at[idx,'home_score'] = home_score
                            df_record.at[idx,'away_score'] = away_score
                            updated = True
        if updated:
            df_record.to_csv(record_path, index=False)
            # Update all-time record
            all_time_df = pd.read_csv(record_csvs['all_time'])
            for idx, row in df_record.iterrows():
                if row['date'] == yesterday:
                    mask = (all_time_df['matchup'] == row['matchup']) & (all_time_df['date'] == yesterday)
                    all_time_df.loc[mask, ['actual_winner','result','home_score','away_score']] = row[['actual_winner','result','home_score','away_score']]
            all_time_df.to_csv(record_csvs['all_time'], index=False)

# --- Update yesterday's records first ---
update_yesterday_records()

# --- Main loop for today ---
for league, cfg in leagues.items():
    print(f"Processing {league}...")
    hist_df = fetch_historical_games(cfg['csv'])
    team_stats, h2h_stats, h2h_history = build_team_stats(hist_df)
    clf = train_model(hist_df, team_stats, h2h_stats)
    games = fetch_odds(cfg['sport_key'], today)
    if not games:
        print(f"No Odds API games found for {league} today.")
        continue

    new_rows = []
    for g in games:
        try:
            home = g['home_team']
            away = g['away_team']

            # --- H2H optional ---
            pair = tuple(sorted([home, away]))
            h2h_home = h2h_stats.get(pair, {'home_wins':0})['home_wins'] if pair in h2h_stats else 0
            h2h_away = h2h_stats.get(pair, {'away_wins':0})['away_wins'] if pair in h2h_stats else 0

            home_form = sum(team_stats.get(home,{'form':[3]})['form'][-5:])
            away_form = sum(team_stats.get(away,{'form':[3]})['form'][-5:])
            home_avg_points = np.mean(team_stats.get(home,{'points_scored':[0]})['points_scored'][-15:])
            away_avg_points = np.mean(team_stats.get(away,{'points_scored':[0]})['points_scored'][-15:])
            home_allowed = np.mean(team_stats.get(home,{'points_allowed':[0]})['points_allowed'][-15:])
            away_allowed = np.mean(team_stats.get(away,{'points_allowed':[0]})['points_allowed'][-15:])
            home_advantage = 1
            try:
                last_game_home = team_stats.get(home, {'last_game': today})['last_game']
                last_game_away = team_stats.get(away, {'last_game': today})['last_game']
                rest_days_home = (datetime.now() - datetime.strptime(last_game_home, "%Y-%m-%d")).days
                rest_days_away = (datetime.now() - datetime.strptime(last_game_away, "%Y-%m-%d")).days
            except:
                rest_days_home, rest_days_away = 0, 0

            # Skip if odds missing
            home_prices = []
            away_prices = []
            for bookmaker in g.get('bookmakers', []):
                try:
                    outcomes = bookmaker.get('markets',[{}])[0].get('outcomes', [])
                    for o in outcomes:
                        if o['name']==home and -110 <= o.get('price',0) <= 130:
                            home_prices.append(o.get('price'))
                        elif o['name']==away and -110 <= o.get('price',0) <= 130:
                            away_prices.append(o.get('price'))
                except:
                    continue

            home_price_mode = most_common([p for p in home_prices if p is not None])
            away_price_mode = most_common([p for p in away_prices if p is not None])
            if home_price_mode is None or away_price_mode is None:
                continue

            pred_input = np.array([[home_form, away_form, home_avg_points, away_avg_points,
                                    home_allowed, away_allowed, h2h_home, h2h_away,
                                    home_advantage, rest_days_home, rest_days_away]])
            pred = clf.predict(pred_input)[0]
            conf_raw = clf.predict_proba(pred_input)[0][pred] if hasattr(clf,"predict_proba") else 0.6
            recommended = "Yes" if conf_raw >= 0.7 else "No"

            predictions.append({
                'matchup': f"{away} @ {home}",
                'predicted_winner': home if pred==1 else away,
                'confidence': recommended,
                'recommended_bet': recommended,
                'home_odds': home_price_mode,
                'away_odds': away_price_mode
            })

            # Append new row to historical CSV (allow duplicates)
            new_row = {'date': today, 'home_team': home, 'away_team': away,
                       'outcome': pred, 'home_score': None, 'away_score': None}
            new_rows.append(new_row)

            # Update per-league record
            record_path = record_csvs[league]
            df_record = pd.read_csv(record_path)
            df_record = pd.concat([df_record, pd.DataFrame([{
                'date': today,
                'matchup': f"{away} @ {home}",
                'predicted_winner': home if pred==1 else away,
                'actual_winner': None,
                'result': None,
                'home_score': None,
                'away_score': None
            }])], ignore_index=True)
            df_record.to_csv(record_path, index=False)

            # Update all-time record
            all_time_df = pd.read_csv(record_csvs['all_time'])
            all_time_df = pd.concat([all_time_df, pd.DataFrame([{
                'date': today,
                'matchup': f"{away} @ {home}",
                'predicted_winner': home if pred==1 else away,
                'actual_winner': None,
                'result': None,
                'home_score': None,
                'away_score': None
            }])], ignore_index=True)
            all_time_df.to_csv(record_csvs['all_time'], index=False)

        except Exception as e:
            print(f"Error processing {g.get('home_team')} @ {g.get('away_team')}: {e}")

    if new_rows:
        hist_df = pd.concat([hist_df, pd.DataFrame(new_rows)], ignore_index=True)
        hist_df.to_csv(cfg['csv'], index=False, encoding="utf-8")

# --- Output HTML ---
df_pred = pd.DataFrame(predictions)
if not df_pred.empty:
    df_pred = df_pred[['matchup','predicted_winner','confidence','recommended_bet','home_odds','away_odds']]
    table_html = df_pred.to_html(index=False, escape=False)
else:
    table_html = "<p>No games to display</p>"

html_content = f"""
<html>
<head>
    <title>Sports Predictions</title>
    <style>
        body {{ font-family: Arial, sans-serif; background-color: #f7f7f7; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        tr:hover {{ background-color: #ddd; }}
    </style>
</head>
<body>
    <h2>Sports Predictions for {today}</h2>
    {table_html}
</body>
</html>
"""

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("Predictions saved to index.html")
os.system("git add .")
os.system('git commit -m "Update predictions"')
os.system("git push origin main")
print("Pushed updated index.html to GitHub")
os.system("shutdown /s /t 60 /f")



