import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
import joblib
from collections import Counter

# === CONFIG ===
ODDS_API_KEY = "6dcf1fafc93b0e7f96353ed3e29bd718"  # Your Odds API key
WEATHER_API_KEY = "a7076980eeb88a2bb07b34b8bb6f7137"  # OpenWeather key
MODEL_DIR = 'models'
DATA_FILES = {
    'NFL': 'nfl_data.csv',
    'NBA': 'nba_data.csv',
    'NHL': 'nhl_data.csv',
    'MLB': 'mlb_data.csv'
}
MAX_API_RETRIES = 3
today = datetime.now().strftime('%Y-%m-%d')

# === HELPER FUNCTIONS ===
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def most_common(lst):
    if not lst:
        return None
    counts = Counter(lst)
    return counts.most_common(1)[0][0]

def get_weather(city):
    """Fetch weather internally (not displayed)."""
    try:
        url = f"http://api.weatherstack.com/current?access_key={WEATHER_API_KEY}&query={city}"
        res = requests.get(url)
        data = res.json()
        if 'current' in data:
            temp = data['current']['temperature']
            wind = data['current']['wind_speed']
            humidity = data['current']['humidity']
            return temp, wind, humidity
    except:
        pass
    return None, None, None

def fetch_espn_injuries(league):
    urls = {
        'NFL': 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries',
        'NBA': 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries',
        'NHL': 'https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/injuries',
        'MLB': 'https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/injuries'
    }
    injured_teams = {}
    try:
        r = requests.get(urls[league])
        if r.status_code == 200:
            data = r.json()
            for team in data.get("teams", []):
                name = team["team"]["displayName"]
                injured_players = [p['player']['displayName'] for p in team.get("injuries", [])]
                injured_teams[name] = injured_players
    except:
        pass
    return injured_teams

def fetch_odds(sport_key):
    for attempt in range(MAX_API_RETRIES):
        try:
            url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/?apiKey={ODDS_API_KEY}&regions=us&markets=h2h&oddsFormat=american&dateFormat=iso"
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            return r.json()
        except:
            time.sleep(1)
    return []

def fetch_historical_games(path):
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, encoding='utf-8', on_bad_lines='skip')
            df = df[df['home_team'].notna() & df['away_team'].notna()]
            df = df[pd.to_datetime(df['date'], errors='coerce').notna()]
            return df
        except:
            pass
    return pd.DataFrame(columns=['date','home_team','away_team','outcome','home_score','away_score'])

def build_team_stats(df):
    team_stats, h2h_stats = {}, {}
    for row in df.to_dict('records'):
        home, away, outcome = row['home_team'], row['away_team'], row.get('outcome',1)
        home_score, away_score = row.get('home_score',0), row.get('away_score',0)
        for team in [home, away]:
            if team not in team_stats:
                team_stats[team] = {'form': [], 'last_game': None, 'points_scored': [], 'points_allowed': []}
        team_stats[home]['form'].append(1 if outcome==1 else 0)
        team_stats[away]['form'].append(0 if outcome==1 else 1)
        team_stats[home]['points_scored'].append(home_score)
        team_stats[away]['points_scored'].append(away_score)
        team_stats[home]['points_allowed'].append(away_score)
        team_stats[away]['points_allowed'].append(home_score)
        team_stats[home]['last_game'] = row['date']
        team_stats[away]['last_game'] = row['date']
        pair = tuple(sorted([home, away]))
        if pair not in h2h_stats:
            h2h_stats[pair] = {'home_wins':0,'away_wins':0}
        if outcome == 1:
            h2h_stats[pair]['home_wins'] +=1
        else:
            h2h_stats[pair]['away_wins'] +=1
    return team_stats, h2h_stats

def train_model(df, team_stats, h2h_stats):
    X, y = [], []
    for row in df.to_dict('records'):
        home, away = row['home_team'], row['away_team']
        try:
            last_home = team_stats.get(home, {'last_game': today})['last_game']
            last_away = team_stats.get(away, {'last_game': today})['last_game']
            rest_home = (datetime.now() - datetime.strptime(last_home,"%Y-%m-%d")).days
            rest_away = (datetime.now() - datetime.strptime(last_away,"%Y-%m-%d")).days
        except:
            rest_home, rest_away = 0,0
        home_form = sum(team_stats.get(home, {'form':[3]})['form'][-5:])
        away_form = sum(team_stats.get(away, {'form':[3]})['form'][-5:])
        home_avg = np.mean(team_stats.get(home, {'points_scored':[0]})['points_scored'][-15:])
        away_avg = np.mean(team_stats.get(away, {'points_scored':[0]})['points_scored'][-15:])
        home_allowed = np.mean(team_stats.get(home, {'points_allowed':[0]})['points_allowed'][-15:])
        away_allowed = np.mean(team_stats.get(away, {'points_allowed':[0]})['points_allowed'][-15:])
        pair = tuple(sorted([home, away]))
        h2h_home = h2h_stats.get(pair,{'home_wins':0})['home_wins']
        h2h_away = h2h_stats.get(pair,{'away_wins':0})['away_wins']
        home_adv = 1
        X.append([home_form, away_form, home_avg, away_avg, home_allowed, away_allowed, h2h_home, h2h_away, home_adv, rest_home, rest_away])
        y.append(row['outcome'])
    clf = RandomForestClassifier(n_estimators=150, random_state=42)
    if X and y:
        clf.fit(X,y)
    return clf

def save_model(clf, league):
    ensure_dir(MODEL_DIR)
    path = os.path.join(MODEL_DIR,f"{league.lower()}_model.pkl")
    joblib.dump(clf,path)

def load_model(league):
    path = os.path.join(MODEL_DIR,f"{league.lower()}_model.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None

# === MAIN SCRIPT ===
ensure_dir(MODEL_DIR)
predictions = []

for league, csv_file in DATA_FILES.items():
    print(f"\nProcessing {league}...")
    hist_df = fetch_historical_games(csv_file)
    team_stats, h2h_stats = build_team_stats(hist_df)
    
    clf = load_model(league)
    if clf is None:
        clf = train_model(hist_df, team_stats, h2h_stats)
        save_model(clf, league)
    
    # Odds and sport keys
    sport_keys = {
        'NFL':'americanfootball_nfl',
        'NBA':'basketball_nba',
        'NHL':'icehockey_nhl',
        'MLB':'baseball_mlb'
    }
    games = fetch_odds(sport_keys[league])
    if not games:
        print(f"No Odds API games found for {league} today.")
        continue
    
    # ESPN injuries
    injuries = fetch_espn_injuries(league)
    
    for g in games:
        try:
            home, away = g['home_team'], g['away_team']
            pair = tuple(sorted([home, away]))
            h2h_home = h2h_stats.get(pair,{'home_wins':0})['home_wins']
            h2h_away = h2h_stats.get(pair,{'away_wins':0})['away_wins']

            home_form = sum(team_stats.get(home,{'form':[3]})['form'][-5:])
            away_form = sum(team_stats.get(away,{'form':[3]})['form'][-5:])
            home_avg = np.mean(team_stats.get(home,{'points_scored':[0]})['points_scored'][-15:])
            away_avg = np.mean(team_stats.get(away,{'points_scored':[0]})['points_scored'][-15:])
            home_allowed = np.mean(team_stats.get(home,{'points_allowed':[0]})['points_allowed'][-15:])
            away_allowed = np.mean(team_stats.get(away,{'points_allowed':[0]})['points_allowed'][-15:])
            home_adv = 1

            try:
                last_home = team_stats.get(home, {'last_game': today})['last_game']
                last_away = team_stats.get(away, {'last_game': today})['last_game']
                rest_home = (datetime.now() - datetime.strptime(last_home,"%Y-%m-%d")).days
                rest_away = (datetime.now() - datetime.strptime(last_away,"%Y-%m-%d")).days
            except:
                rest_home, rest_away = 0,0

            # Odds
            home_prices, away_prices = [], []
            for bookmaker in g.get('bookmakers',[]):
                try:
                    outcomes = bookmaker.get('markets',[{}])[0].get('outcomes',[])
                    for o in outcomes:
                        if o['name']==home:
                            home_prices.append(o.get('price'))
                        elif o['name']==away:
                            away_prices.append(o.get('price'))
                except:
                    continue
            home_odds = most_common([p for p in home_prices if p is not None])
            away_odds = most_common([p for p in away_prices if p is not None])
            if home_odds is None or away_odds is None:
                continue

            # Prediction & 60% threshold
            X_input = np.array([[home_form, away_form, home_avg, away_avg, home_allowed, away_allowed,
                                 h2h_home, h2h_away, home_adv, rest_home, rest_away]])
            pred = clf.predict(X_input)[0]
            conf_raw = clf.predict_proba(X_input)[0][pred] if hasattr(clf,"predict_proba") else 0.6
            recommended = "Yes" if conf_raw >= 0.6 else "No"

            injured_players = injuries.get(home,[]) + injuries.get(away,[])

            predictions.append({
                'matchup': f"{away} @ {home}",
                'predicted_winner': home if pred==1 else away,
                'confidence': recommended,
                'recommended_bet': recommended,
                'home_odds': home_odds,
                'away_odds': away_odds,
                'injuries': injured_players
            })

            # Update historical CSV automatically
            new_row = {'date': today, 'home_team': home, 'away_team': away,
                       'outcome': pred, 'home_score': None, 'away_score': None}
            hist_df = pd.concat([hist_df, pd.DataFrame([new_row])], ignore_index=True)
            hist_df.to_csv(csv_file, index=False, encoding='utf-8')

        except Exception as e:
            print(f"Error processing {g.get('home_team')} @ {g.get('away_team')}: {e}")

# --- Update league_wl_record.csv automatically ---
league_records = {}
all_wins, all_losses = 0, 0
for league, csv_file in DATA_FILES.items():
    df = fetch_historical_games(csv_file)
    if not df.empty:
        home_wins = df['outcome'].sum()
        total_games = len(df)
        away_wins = total_games - home_wins
        league_records[league] = f"{home_wins}-{away_wins}"
        all_wins += home_wins
        all_losses += away_wins
    else:
        league_records[league] = "0-0"
league_records['ALL TIME RECORD'] = f"{all_wins}-{all_losses}"
with open('league_wl_record.csv','w') as f:
    for league, record in league_records.items():
        f.write(f"{league}: {record}\n")
print("Updated league_wl_record.csv")

# --- Generate HTML ---
df_pred = pd.DataFrame(predictions)
if not df_pred.empty:
    df_pred = df_pred[['matchup','predicted_winner','confidence','recommended_bet','home_odds','away_odds','injuries']]
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

with open("index.html","w",encoding="utf-8") as f:
    f.write(html_content)

print("Predictions saved to index.html")
os.system("git add .")
os.system('git commit -m "Update predictions"')
os.system("git push origin main")
print("Pushed updated index.html to GitHub")
os.system("shutdown /s /t 60 /f")

