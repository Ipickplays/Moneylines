import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import joblib
from collections import Counter
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from dateutil import parser
import pytz
from dotenv import load_dotenv  # NEW

# === LOAD CONFIG (.env) ===
load_dotenv()
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", 0.65))

# === CONFIG ===
MODEL_DIR = 'models'
DATA_FILES = {
    'NFL': 'nfl_data.csv',
    'NBA': 'nba_data.csv',
    'NHL': 'nhl_data.csv',
    'MLB': 'mlb_data.csv'
}
MAX_API_RETRIES = 3

# Set Central Time for IL
central = pytz.timezone("America/Chicago")
today = datetime.now(central).strftime('%Y-%m-%d')
today_date = datetime.now(central).date()
yesterday_date = (datetime.now(central) - timedelta(days=1)).date()

# === HELPER FUNCTIONS ===
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def most_common(lst):
    if not lst:
        return None
    counts = Counter(lst)
    return counts.most_common(1)[0][0]

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

# === UPDATED HISTORICAL GAME FETCH ===
def fetch_historical_games(path):
    cols = ['date','home_team','away_team','home_score','away_score','outcome',
            'home_odds','away_odds','injuries_home','injuries_away']
    if os.path.exists(path):
        df = pd.read_csv(path, on_bad_lines='skip')
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        df.dropna(subset=['home_team','away_team','date'], inplace=True)
        df.drop_duplicates(subset=['date','home_team','away_team'], inplace=True)
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.strftime('%Y-%m-%d')
        df.fillna(0, inplace=True)
        return df[cols]
    return pd.DataFrame(columns=cols)

def fetch_scores_from_espn(league, game_date):
    urls = {
        'NFL': f'https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?dates={game_date.strftime("%Y%m%d")}',
        'NBA': f'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={game_date.strftime("%Y%m%d")}',
        'NHL': f'https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard?dates={game_date.strftime("%Y%m%d")}',
        'MLB': f'https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates={game_date.strftime("%Y%m%d")}'
    }
    scores = {}
    try:
        r = requests.get(urls[league])
        if r.status_code == 200:
            data = r.json()
            for evt in data.get('events', []):
                home = evt['competitions'][0]['competitors'][0]['team']['displayName']
                away = evt['competitions'][0]['competitors'][1]['team']['displayName']
                home_score = int(evt['competitions'][0]['competitors'][0].get('score',0))
                away_score = int(evt['competitions'][0]['competitors'][1].get('score',0))
                outcome = 1 if home_score > away_score else 0
                scores[(home, away)] = {'home_score': home_score, 'away_score': away_score, 'outcome': outcome}
    except:
        pass
    return scores

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

# === UPDATED TRAIN MODEL ===
def train_model(df, team_stats, h2h_stats):
    X, y = [], []
    for row in df.to_dict('records'):
        home, away = row['home_team'], row['away_team']
        pair = tuple(sorted([home, away]))

        home_form = sum(team_stats.get(home, {'form':[3]})['form'][-5:])
        away_form = sum(team_stats.get(away, {'form':[3]})['form'][-5:])
        home_avg = np.mean(team_stats.get(home, {'points_scored':[0]})['points_scored'][-15:])
        away_avg = np.mean(team_stats.get(away, {'points_scored':[0]})['points_scored'][-15:])
        home_diff = np.mean(np.array(team_stats.get(home, {'points_scored':[0]})['points_scored'][-10:]) -
                            np.array(team_stats.get(home, {'points_allowed':[0]})['points_allowed'][-10:]))
        away_diff = np.mean(np.array(team_stats.get(away, {'points_scored':[0]})['points_scored'][-10:]) -
                            np.array(team_stats.get(away, {'points_allowed':[0]})['points_allowed'][-10:]))
        h2h_home = h2h_stats.get(pair, {'home_wins':0})['home_wins']
        h2h_away = h2h_stats.get(pair, {'away_wins':0})['away_wins']
        rest_home, rest_away = 0, 0
        home_odds = row.get('home_odds', 0)
        away_odds = row.get('away_odds', 0)
        injuries_home = row.get('injuries_home', 0)
        injuries_away = row.get('injuries_away', 0)

        def american_to_prob(o):
            if o > 0:
                return 100 / (o + 100)
            elif o < 0:
                return abs(o) / (abs(o) + 100)
            return 0.5

        home_prob = american_to_prob(home_odds)
        away_prob = american_to_prob(away_odds)

        X.append([
            home_form, away_form, home_avg, away_avg,
            home_diff, away_diff, h2h_home, h2h_away,
            rest_home, rest_away, injuries_home, injuries_away,
            home_prob, away_prob
        ])
        y.append(row['outcome'])

    if X and y:
        clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        clf.fit(X, y)
        acc = clf.score(X, y)
        print(f"Training accuracy: {acc:.3f}")
        calibrated = CalibratedClassifierCV(clf, method='isotonic', cv='prefit')
        calibrated.fit(X, y)
        return calibrated, acc
    return None, 0

# === UPDATED SAVE MODEL ===
def save_model(clf, league, accuracy):
    ensure_dir(MODEL_DIR)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_path = os.path.join(MODEL_DIR, f"{league.lower()}_model_{timestamp}.pkl")
    meta_path = os.path.join(MODEL_DIR, f"{league.lower()}_meta_{timestamp}.txt")
    joblib.dump(clf, model_path)
    with open(meta_path, "w") as f:
        f.write(f"Accuracy: {accuracy:.3f}\nTrained: {timestamp}\n")

def load_model(league):
    path = os.path.join(MODEL_DIR,f"{league.lower()}_model.pkl")
    if os.path.exists(path):
        clf = joblib.load(path)
        try:
            check_is_fitted(clf)
            return clf
        except NotFittedError:
            return None
    return None

# === MAIN LOOP ===
ensure_dir(MODEL_DIR)
predictions = []

for league, csv_file in DATA_FILES.items():
    print(f"\nProcessing {league}...")
    hist_df = fetch_historical_games(csv_file)
    team_stats, h2h_stats = build_team_stats(hist_df)

    clf = load_model(league)
    if clf is None:
        clf, acc = train_model(hist_df, team_stats, h2h_stats)
        if clf is not None:
            save_model(clf, league, acc)

    if clf is None:
        print(f"Insufficient data to train model for {league}. Skipping predictions.")
        continue

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

    scores_yesterday = fetch_scores_from_espn(league, yesterday_date)
    for (home, away), score_data in scores_yesterday.items():
        hist_df = pd.concat([hist_df, pd.DataFrame([{
            'date': yesterday_date.strftime('%Y-%m-%d'),
            'home_team': home,
            'away_team': away,
            'outcome': score_data['outcome'],
            'home_score': score_data['home_score'],
            'away_score': score_data['away_score'],
            'home_odds': 0,
            'away_odds': 0,
            'injuries_home': 0,
            'injuries_away': 0
        }])], ignore_index=True)
    hist_df.to_csv(csv_file, index=False, encoding='utf-8')

    games_today = []
    for g in games:
        if 'commence_time' in g:
            try:
                game_dt_utc = parser.isoparse(g['commence_time'])
                game_dt_local = game_dt_utc.astimezone(central)
                if game_dt_local.date() == today_date:
                    games_today.append(g)
            except:
                continue

    injuries = fetch_espn_injuries(league)

    for g in games_today:
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

            rest_home, rest_away = 0,0

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
            home_odds = most_common([p for p in home_prices if p is not None]) or "N/A"
            away_odds = most_common([p for p in away_prices if p is not None]) or "N/A"

            X_input = np.array([[home_form, away_form, home_avg, away_avg,
                                 home_allowed, away_allowed, h2h_home, h2h_away,
                                 home_adv, rest_home, rest_away, 0, 0, 0, 0]])
            pred = clf.predict(X_input)[0]
            conf_raw = clf.predict_proba(X_input)[0][pred] if hasattr(clf,"predict_proba") else 0.6
            recommended = "Yes" if conf_raw >= CONF_THRESHOLD else "No"
            confidence = round(conf_raw, 2)
            injured_players = injuries.get(home,[]) + injuries.get(away,[])

            predictions.append({
                'matchup': f"{away} @ {home}",
                'predicted_winner': home if pred==1 else away,
                'confidence': confidence,
                'recommended_bet': recommended,
                'home_odds': home_odds,
                'away_odds': away_odds,
                'injuries': injured_players
            })

        except Exception as e:
            print(f"Error processing {g.get('home_team')} @ {g.get('away_team')}: {e}")

# === HTML OUTPUT ===
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
