import requests
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
import time
import os
from collections import Counter

# --- API Key ---
ODDS_API_KEY = '6dcf1fafc93b0e7f96353ed3e29bd718'

# --- League Config ---
leagues = {
    'NBA': {'sport_key': 'basketball_nba', 'csv': 'nba_data.csv', 'record': 'nba_record.csv'},
    'NFL': {'sport_key': 'americanfootball_nfl', 'csv': 'nfl_data.csv', 'record': 'nfl_record.csv'},
    'NHL': {'sport_key': 'icehockey_nhl', 'csv': 'nhl_data.csv', 'record': 'nhl_record.csv'},
    'MLB': {'sport_key': 'baseball_mlb', 'csv': 'mlb_data.csv', 'record': 'mlb_record.csv'}
}
all_time_record_file = "all_time_record.csv"

MAX_API_RETRIES = 3
today = datetime.now().strftime('%Y-%m-%d')
predictions = []

# --- Helper: mode ---
def most_common(lst):
    if not lst:
        return None
    counts = Counter(lst)
    return counts.most_common(1)[0][0]

# --- CSV Helpers ---
def ensure_csv(path, headers):
    if not os.path.exists(path):
        pd.DataFrame(columns=headers).to_csv(path, index=False)

def reset_csv_if_needed(csv_path, backup_path):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, on_bad_lines="skip")
        if not df.empty:
            last_date = pd.to_datetime(df["date"], errors="coerce").max()
            if (datetime.now() - last_date.to_pydatetime()).days > 30:
                print(f"⚠️ Resetting {csv_path} (offseason). Backup saved.")
                os.replace(csv_path, backup_path)
                df.iloc[0:0].to_csv(csv_path, index=False)
    else:
        print(f"Creating {csv_path} fresh.")
        pd.DataFrame(columns=['date','home_team','away_team','outcome','home_score','away_score']).to_csv(csv_path, index=False)

def restore_backup_if_needed(csv_path, backup_path):
    if not os.path.exists(csv_path) and os.path.exists(backup_path):
        backup_df = pd.read_csv(backup_path, on_bad_lines="skip")
        if not backup_df.empty:
            last_date = pd.to_datetime(backup_df["date"], errors="coerce").max()
            if (datetime.now() - last_date.to_pydatetime()).days <= 10:
                print(f"🔄 Restoring {csv_path} from backup.")
                backup_df.to_csv(csv_path, index=False)

def update_record(record_file, league, outcome):
    ensure_csv(record_file, ["league", "wins", "losses"])
    df = pd.read_csv(record_file)
    if league not in df["league"].values:
        df.loc[len(df)] = [league, 0, 0]
    if outcome == 1:
        df.loc[df["league"] == league, "wins"] += 1
    else:
        df.loc[df["league"] == league, "losses"] += 1
    df.to_csv(record_file, index=False)

def update_all_time(outcome):
    ensure_csv(all_time_record_file, ["league","wins","losses"])
    df = pd.read_csv(all_time_record_file)
    if "ALL" not in df["league"].values:
        df.loc[len(df)] = ["ALL", 0, 0]
    if outcome == 1:
        df.loc[df["league"] == "ALL","wins"] += 1
    else:
        df.loc[df["league"] == "ALL","losses"] += 1
    df.to_csv(all_time_record_file, index=False)

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

# --- Fetch odds ---
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

# --- Main loop ---
for league, cfg in leagues.items():
    print(f"\n📊 Processing {league}...")
    csv_path, record_path = cfg["csv"], cfg["record"]
    backup_path = f"{csv_path}.backup"

    reset_csv_if_needed(csv_path, backup_path)
    restore_backup_if_needed(csv_path, backup_path)

    hist_df = fetch_historical_games(csv_path)
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

            # quick placeholder features
            X_test = np.array([[0,0,0,0,0,0,0,0,1,0,0]])
            pred = clf.predict(X_test)[0]

            # Update records
            update_record(record_path, league, pred)
            update_all_time(pred)

            new_row = {'date': today, 'home_team': home, 'away_team': away,
                       'outcome': pred, 'home_score': None, 'away_score': None}
            if not ((hist_df['home_team']==home) & (hist_df['away_team']==away) & (hist_df['date']==today)).any():
                new_rows.append(new_row)

            predictions.append({
                'matchup': f"{away} @ {home}",
                'predicted_winner': home if pred==1 else away,
                'confidence': "N/A",
                'recommended_bet': "N/A",
                'home_odds': None,
                'away_odds': None
            })
        except Exception as e:
            print(f"Error processing game: {e}")

    if new_rows:
        hist_df = pd.concat([hist_df, pd.DataFrame(new_rows)], ignore_index=True)
    hist_df.to_csv(csv_path, index=False, encoding='utf-8')

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


