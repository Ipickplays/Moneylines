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
    'NBA': {'sport_key': 'basketball_nba', 'csv': 'nba_data.csv'},
    'NFL': {'sport_key': 'americanfootball_nfl', 'csv': 'nfl_data.csv'},
    'NHL': {'sport_key': 'icehockey_nhl', 'csv': 'nhl_data.csv'},
    'MLB': {'sport_key': 'baseball_mlb', 'csv': 'mlb_data.csv'}
}

MAX_API_RETRIES = 3
MAX_GAMES_PER_LEAGUE = 300
today = datetime.now().strftime('%Y-%m-%d')
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
            return pd.read_csv(path, encoding='latin1', on_bad_lines='skip')
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
        rest_days_home = (datetime.now() - datetime.strptime(team_stats.get(home,{'last_game':today})['last_game'], "%Y-%m-%d")).days
        rest_days_away = (datetime.now() - datetime.strptime(team_stats.get(away,{'last_game':today})['last_game'], "%Y-%m-%d")).days

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

# --- Check and reset CSV if offseason (no games in 30+ days) ---
for league, cfg in leagues.items():
    if os.path.exists(cfg['csv']):
        try:
            hist_df = pd.read_csv(cfg['csv'], on_bad_lines='skip')
            if not hist_df.empty:
                last_game_date = max(pd.to_datetime(hist_df['date'], errors='coerce'))
                days_since_last_game = (datetime.now() - last_game_date).days
                if days_since_last_game > 30:
                    os.remove(cfg['csv'])
                    print(f"{cfg['csv']} reset because {league} has not played in over 30 days")
        except Exception as e:
            print(f"Error checking {cfg['csv']}: {e}")

# --- Main loop ---
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

            home_form = sum(team_stats.get(home,{'form':[3]})['form'][-5:])
            away_form = sum(team_stats.get(away,{'form':[3]})['form'][-5:])
            home_avg_points = np.mean(team_stats.get(home,{'points_scored':[0]})['points_scored'][-15:])
            away_avg_points = np.mean(team_stats.get(away,{'points_scored':[0]})['points_scored'][-15:])
            home_allowed = np.mean(team_stats.get(home,{'points_allowed':[0]})['points_allowed'][-15:])
            away_allowed = np.mean(team_stats.get(away,{'points_allowed':[0]})['points_allowed'][-15:])
            pair = tuple(sorted([home,away]))
            h2h_home = h2h_stats.get(pair,{'home_wins':0})['home_wins']
            h2h_away = h2h_stats.get(pair,{'away_wins':0})['away_wins']
            home_advantage = 1
            rest_days_home = (datetime.now() - datetime.strptime(team_stats.get(home,{'last_game':today})['last_game'], "%Y-%m-%d")).days
            rest_days_away = (datetime.now() - datetime.strptime(team_stats.get(away,{'last_game':today})['last_game'], "%Y-%m-%d")).days

            # Skip adding matchup to CSV until outcome is known
            if not h2h_history.get(pair):
                continue

            pred_input = np.array([[home_form, away_form, home_avg_points, away_avg_points,
                                    home_allowed, away_allowed, h2h_home, h2h_away,
                                    home_advantage, rest_days_home, rest_days_away]])
            pred = clf.predict(pred_input)[0]
            conf_raw = clf.predict_proba(pred_input)[0][pred] if hasattr(clf,"predict_proba") else 0.6

            recommended = "Yes" if conf_raw >= 0.7 else "No"

            # Odds mode mapping to prevent mix-up
            home_prices = []
            away_prices = []
            for bookmaker in g.get('bookmakers', []):
                try:
                    outcomes = bookmaker.get('markets',[{}])[0].get('outcomes', [])
                    if len(outcomes) >= 2:
                        for o in outcomes:
                            if o['name'] == home:
                                home_prices.append(o.get('price'))
                            elif o['name'] == away:
                                away_prices.append(o.get('price'))
                except:
                    continue

            home_price_mode = most_common([p for p in home_prices if p is not None])
            away_price_mode = most_common([p for p in away_prices if p is not None])

            # --- Odds filter between -110 and +130 ---
            if home_price_mode is not None and away_price_mode is not None:
                if -110 <= home_price_mode <= 130 and -110 <= away_price_mode <= 130:
                    predictions.append({
                        'matchup': f"{away} @ {home}",
                        'predicted_winner': home if pred==1 else away,
                        'confidence': recommended,
                        'recommended_bet': recommended,
                        'home_odds': home_price_mode,
                        'away_odds': away_price_mode
                    })

            # --- Append to CSV if outcome exists ---
            outcome_known = any([home in h2h_history[pair][-1:] or away in h2h_history[pair][-1:]])
            if outcome_known:
                new_row = {'date': today, 'home_team': home, 'away_team': away,
                           'outcome': pred, 'home_score': None, 'away_score': None}
                if not ((hist_df['home_team'] == home) & (hist_df['away_team'] == away) & (hist_df['date'] == today)).any():
                    new_rows.append(new_row)

        except Exception as e:
            print(f"Error processing game {g.get('home_team')} @ {g.get('away_team')}: {e}")

    # --- Append new games and prune old ones ---
    if new_rows:
        hist_df = pd.concat([hist_df, pd.DataFrame(new_rows)], ignore_index=True)
    if len(hist_df) > MAX_GAMES_PER_LEAGUE:
        hist_df = hist_df.tail(MAX_GAMES_PER_LEAGUE)
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

