import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
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

# --- Record CSV ---
record_csv = 'league_wl_record.csv'  # single CSV for all leagues + all-time

MAX_API_RETRIES = 3
today = datetime.now().strftime('%Y-%m-%d')
predictions = []

# --- Helper: mode ---
def most_common(lst):
    if not lst:
        return None
    counts = Counter(lst)
    return counts.most_common(1)[0][0]

# --- New: American odds to probability ---
def american_to_prob(odds):
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return -odds / (-odds + 100)

# --- Fetch historical games ---
def fetch_historical_games(path):
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, encoding='utf-8', on_bad_lines='skip', header=0)
            # Drop rows missing critical info
            df = df[df['home_team'].notna() & df['away_team'].notna()]
            df = df[pd.to_datetime(df['date'], errors='coerce').notna()]
            # Handle outliers: cap scores at reasonable max (e.g., 200 for NBA)
            df['home_score'] = df['home_score'].clip(0, 200)
            df['away_score'] = df['away_score'].clip(0, 200)
            return df
        except Exception as e:
            print(f"Error reading {path}: {e}")
    return pd.DataFrame(columns=['date','home_team','away_team','outcome','home_score','away_score'])

# --- New: Update historical outcomes with actual scores ---
def update_historical_outcomes(league, cfg, api_key, hist_df):
    # Find rows with missing scores (recent games)
    pending = hist_df[hist_df['home_score'].isna()]
    if pending.empty:
        return hist_df
    
    updated_count = 0
    for idx, row in pending.iterrows():
        date = row['date']
        home = row['home_team']
        away = row['away_team']
        # Fetch scores via API (using Odds API scores endpoint)
        for attempt in range(MAX_API_RETRIES):
            try:
                url = f"https://api.the-odds-api.com/v4/sports/{cfg['sport_key']}/scores/?apiKey={api_key}&daysFrom=3&dateFormat=iso"
                r = requests.get(url, timeout=10)
                r.raise_for_status()
                scores = r.json()
                for game in scores:
                    if game.get('home_team') == home and game.get('away_team') == away and game.get('commence_time', '').startswith(date):
                        home_score = game['scores'][0]['score'] if game.get('scores') else None
                        away_score = game['scores'][1]['score'] if game.get('scores') else None
                        if home_score is not None and away_score is not None:
                            outcome = 1 if home_score > away_score else 0
                            hist_df.at[idx, 'home_score'] = home_score
                            hist_df.at[idx, 'away_score'] = away_score
                            hist_df.at[idx, 'outcome'] = outcome
                            updated_count += 1
                            break
                break
            except Exception as e:
                print(f"Attempt {attempt+1} failed updating {home} vs {away}: {e}")
                time.sleep(1)
    
    if updated_count > 0:
        hist_df.to_csv(cfg['csv'], index=False)
        print(f"Updated {updated_count} games for {league}")
    return hist_df

# --- Build team stats (with time-weighting) ---
def build_team_stats(df):
    team_stats = {}
    h2h_stats = {}
    h2h_history = {}
    for row in df.to_dict('records'):
        home = row['home_team']
        away = row['away_team']
        outcome = row.get('outcome', 1)  # default to 1 if missing
        home_score = row.get('home_score', 0)
        away_score = row.get('away_score', 0)
        date = pd.to_datetime(row['date'])

        for team in [home, away]:
            if team not in team_stats:
                team_stats[team] = {'form':[], 'last_game':None, 'points_scored':[], 'points_allowed':[], 'dates':[]}

        team_stats[home]['form'].append(1 if outcome==1 else 0)
        team_stats[away]['form'].append(0 if outcome==1 else 1)
        team_stats[home]['points_scored'].append(home_score)
        team_stats[home]['points_allowed'].append(away_score)
        team_stats[away]['points_scored'].append(away_score)
        team_stats[away]['points_allowed'].append(home_score)
        team_stats[home]['last_game'] = row['date']
        team_stats[away]['last_game'] = row['date']
        team_stats[home]['dates'].append(date)
        team_stats[away]['dates'].append(date)

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

    # Time-weighted averages (exponential decay, recent games weighted more)
    for team in team_stats:
        if team_stats[team]['dates']:
            sorted_idx = np.argsort(team_stats[team]['dates'])
            weights = np.exp(-0.1 * np.arange(len(sorted_idx))[::-1])  # Decay factor 0.1
            team_stats[team]['weighted_scored'] = np.average(np.array(team_stats[team]['points_scored'])[sorted_idx][-15:], weights=weights[-15:])
            team_stats[team]['weighted_allowed'] = np.average(np.array(team_stats[team]['points_allowed'])[sorted_idx][-15:], weights=weights[-15:])
        else:
            team_stats[team]['weighted_scored'] = 0
            team_stats[team]['weighted_allowed'] = 0

    return team_stats, h2h_stats, h2h_history

# --- Train model (with tuning) ---
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
        home_avg_points = team_stats.get(home,{'weighted_scored':0})['weighted_scored']
        away_avg_points = team_stats.get(away,{'weighted_scored':0})['weighted_scored']
        home_allowed = team_stats.get(home,{'weighted_allowed':0})['weighted_allowed']
        away_allowed = team_stats.get(away,{'weighted_allowed':0})['weighted_allowed']
        pair = tuple(sorted([home, away]))
        h2h_home = h2h_stats.get(pair,{'home_wins':0})['home_wins']
        h2h_away = h2h_stats.get(pair,{'away_wins':0})['away_wins']
        home_advantage = 1
        # Placeholder for implied probs; will add in prediction phase

        X.append([home_form, away_form, home_avg_points, away_avg_points,
                  home_allowed, away_allowed, h2h_home, h2h_away,
                  home_advantage, rest_days_home, rest_days_away])
        y.append(row['outcome'])

    if not X or not y:
        return RandomForestClassifier(n_estimators=150, random_state=42)

    param_grid = {'n_estimators': [100, 150, 200], 'max_depth': [10, 15, 20]}
    tscv = TimeSeriesSplit(n_splits=5)
    clf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=tscv, scoring='accuracy')
    clf.fit(X, y)
    print(f"Best params for model: {clf.best_params_}")
    return clf.best_estimator_

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

# --- Initialize league record CSV ---
if not os.path.exists(record_csv):
    with open(record_csv, 'w') as f:
        f.write("MLB: 0-0\nNHL: 0-0\nNBA: 0-0\nNFL: 0-0\nALL TIME RECORD: 0-0\n")

# --- Reset CSVs for offseason leagues (backup and wipe) ---
for league, cfg in leagues.items():
    if os.path.exists(cfg['csv']):
        try:
            hist_df = pd.read_csv(cfg['csv'], on_bad_lines='skip', encoding='utf-8')
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

# --- Main loop ---
for league, cfg in leagues.items():
    print(f"Processing {league}...")
    hist_df = fetch_historical_games(cfg['csv'])
    hist_df = update_historical_outcomes(league, cfg, ODDS_API_KEY, hist_df)
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
            home_avg_points = team_stats.get(home,{'weighted_scored':0})['weighted_scored']
            away_avg_points = team_stats.get(away,{'weighted_scored':0})['weighted_scored']
            home_allowed = team_stats.get(home,{'weighted_allowed':0})['weighted_allowed']
            away_allowed = team_stats.get(away,{'weighted_allowed':0})['weighted_allowed']
            home_advantage = 1
            try:
                last_game_home = team_stats.get(home, {'last_game': today})['last_game']
                last_game_away = team_stats.get(away, {'last_game': today})['last_game']
                rest_days_home = (datetime.now() - datetime.strptime(last_game_home, "%Y-%m-%d")).days
                rest_days_away = (datetime.now() - datetime.strptime(last_game_away, "%Y-%m-%d")).days
            except:
                rest_days_home, rest_days_away = 0, 0

            # Skip if odds missing
            home_prices, away_prices = [], []
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

            home_imp_prob = american_to_prob(home_price_mode)
            away_imp_prob = american_to_prob(away_price_mode)

            pred_input = np.array([[home_form, away_form, home_avg_points, away_avg_points,
                                    home_allowed, away_allowed, h2h_home, h2h_away,
                                    home_advantage, rest_days_home, rest_days_away,
                                    home_imp_prob, away_imp_prob]])
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

            # --- Append to historical CSV --- (with placeholder for actuals)
            new_row = {'date': today, 'home_team': home, 'away_team': away,
                       'outcome': None, 'home_score': None, 'away_score': None}  # Outcome now updated later
            new_rows.append(new_row)

        except Exception as e:
            print(f"Error processing {g.get('home_team')} @ {g.get('away_team')}: {e}")

    # Save new rows for historical CSV
    if new_rows:
        hist_df = pd.concat([hist_df, pd.DataFrame(new_rows)], ignore_index=True)
        hist_df.to_csv(cfg['csv'], index=False, encoding='utf-8')

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