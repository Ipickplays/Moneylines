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

# === CONFIG ===
ODDS_API_KEY = "6dcf1fafc93b0e7f96353ed3e29bd718"
MODEL_DIR = 'models'
DATA_FILES = {
    'NFL': 'nfl_data.csv',
    'NBA': 'nba_data.csv',
    'NHL': 'nhl_data.csv',
    'MLB': 'mlb_data.csv'
}
MAX_API_RETRIES = 3
CONF_THRESHOLD = 0.65  # threshold for recommended bet

# Set Central Time for IL
central = pytz.timezone("America/Chicago")
today = datetime.now(central).strftime('%Y-%m-%d')
today_date = datetime.now(central).date()
yesterday_date = (datetime.now(central) - timedelta(days=1)).date()

# === HELPERS ===
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def most_common(lst):
    if not lst:
        return None
    counts = Counter(lst)
    return counts.most_common(1)[0][0]

def normalize_team_name(name):
    """Normalize team names so ESPN and Odds API match properly."""
    name = name.lower().replace(" ", "").replace(".", "")
    replacements = {
        "newyorkjets": "jets", "newyorkgiants": "giants", "sanfrancisco49ers": "49ers",
        "greenbaypackers": "packers", "chicagobears": "bears", "cincinnatibengals": "bengals",
        "buffalobills": "bills", "philadelphiaeagles": "eagles", "dallascowboys": "cowboys",
        "miamidolphins": "dolphins", "kansascitychiefs": "chiefs", "minnesotavikings": "vikings",
        "tampabaybuccaneers": "buccaneers", "losangeleschargers": "chargers",
        "losangelesrams": "rams", "baltimoreravens": "ravens", "pittsburghsteelers": "steelers",
        "clevelandbrowns": "browns", "denverbroncos": "broncos", "detroitlions": "lions",
        "houstontexans": "texans", "atlantafalcons": "falcons", "neworleanssaints": "saints",
        "seattleseahawks": "seahawks", "carolinapanthers": "panthers", "arizonacardinals": "cardinals",
        "tennesseetitans": "titans", "indianapoliscolts": "colts", "washingtoncommanders": "commanders",
        "lasvegasraiders": "raiders",

        # NBA
        "chicagobulls": "bulls", "milwaukeebucks": "bucks", "losangeleslakers": "lakers",
        "losangelesclippers": "clippers", "miamiheat": "heat", "bostonceltics": "celtics",
        "brooklynnets": "nets", "newyorkknicks": "knicks", "philadelphia76ers": "76ers",
        "clevelandcavaliers": "cavaliers", "dallasmavericks": "mavericks",
        "phoenixsuns": "suns", "goldenstatewarriors": "warriors",
        "sacramentokings": "kings", "minnesotatimberwolves": "timberwolves",
        "denvernuggets": "nuggets", "memphisgrizzlies": "grizzlies",
        "portlandtrailblazers": "trailblazers", "oklahomacitythunder": "thunder",
        "utahjazz": "jazz", "neworleanspelicans": "pelicans", "orlandomagic": "magic",
        "torontoraptors": "raptors", "atlantahawks": "hawks", "charlottehornets": "hornets",
        "detroitpistons": "pistons", "washingtonwizards": "wizards", "sanantoniospurs": "spurs",
        "indianapacers": "pacers", "houstonrockets": "rockets",

        # NHL
        "chicagoblackhawks": "blackhawks", "detroitredwings": "redwings", "bostonbruins": "bruins",
        "newyorkrangers": "rangers", "torontomapleleafs": "mapleleafs", "montrealcanadiens": "canadiens",
        "floridapanthers": "panthersnhl", "edmontonoilers": "oilers", "vancouvercanucks": "canucks",
        "coloradoavalanche": "avalanche", "pittsburghpenguins": "penguins", "nashvillepredators": "predators",
        "dallasstars": "stars", "stlouisblues": "blues", "newjerseydevils": "devils",
        "washingtoncapitals": "capitals", "ottawasenators": "senators", "tampabaylightning": "lightning",
        "minnesotawild": "wild", "calgaryflames": "flames"
    }
    for k, v in replacements.items():
        if k in name:
            return v
    return name

# === ESPN INJURY FETCHER ===
def fetch_espn_injuries(league):
    urls = {
        'NFL': 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries',
        'NBA': 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries',
        'NHL': 'https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/injuries'
    }

    if league not in urls:
        return {}

    injured_teams = {}
    try:
        r = requests.get(urls[league], timeout=10)
        r.raise_for_status()
        data = r.json()
        for team_data in data.get("teams", []):
            team_name = team_data.get("team", {}).get("displayName", "Unknown Team")
            injured_players = []
            for injury in team_data.get("injuries", []):
                player_info = injury.get("player") or injury.get("athlete")
                if not player_info:
                    continue
                player_name = player_info.get("displayName", "Unknown Player")
                position = player_info.get("position", {}).get("abbreviation", "")
                status = injury.get("status", "")
                description = injury.get("description", "")
                formatted = f"{player_name} ({position}) - {status}: {description}".strip(" -:")
                injured_players.append(formatted)
            injured_teams[team_name] = injured_players
    except Exception as e:
        print(f"Error fetching {league} injuries: {e}")
    return injured_teams

# === ODDS + SCORE FETCHERS ===
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
            numeric_cols = ['outcome', 'home_score', 'away_score']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna(subset=numeric_cols)
            for col in ['date','home_team','away_team','outcome','home_score','away_score']:
                if col not in df.columns:
                    df[col] = None
            return df
        except:
            pass
    return pd.DataFrame(columns=['date','home_team','away_team','outcome','home_score','away_score'])

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

# === STATS + MODEL TRAINING ===
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
            rest_home = (datetime.now(central) - datetime.strptime(last_home,"%Y-%m-%d")).days
            rest_away = (datetime.now(central) - datetime.strptime(last_away,"%Y-%m-%d")).days
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
    if X and y:
        clf = RandomForestClassifier(n_estimators=150, random_state=42)
        clf.fit(X, y)
        calibrated_clf = CalibratedClassifierCV(clf, method='isotonic', cv='prefit')
        calibrated_clf.fit(X, y)
        return calibrated_clf
    return None

def save_model(clf, league):
    ensure_dir(MODEL_DIR)
    path = os.path.join(MODEL_DIR,f"{league.lower()}_model.pkl")
    joblib.dump(clf,path)

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
        clf = train_model(hist_df, team_stats, h2h_stats)
        if clf is not None:
            save_model(clf, league)

    if clf is None:
        print(f"Insufficient data for {league}. Skipping predictions.")
        continue

    sport_keys = {
        'NFL':'americanfootball_nfl',
        'NBA':'basketball_nba',
        'NHL':'icehockey_nhl'
    }
    games = fetch_odds(sport_keys.get(league, ""))
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
            'away_score': score_data['away_score']
        }])], ignore_index=True)
    hist_df.to_csv(csv_file, index=False, encoding='utf-8')

    injuries = fetch_espn_injuries(league)
    normalized_injuries = {normalize_team_name(k): v for k, v in injuries.items()}

    for g in games:
        try:
            home, away = g['home_team'], g['away_team']
            home_norm = normalize_team_name(home)
            away_norm = normalize_team_name(away)
            home_injury_list = normalized_injuries.get(home_norm, [])
            away_injury_list = normalized_injuries.get(away_norm, [])

            home_injuries = len(home_injury_list)
            away_injuries = len(away_injury_list)

            pair = tuple(sorted([home, away]))
            h2h_home = h2h_stats.get(pair,{'home_wins':0})['home_wins']
            h2h_away = h2h_stats.get(pair,{'away_wins':0})['away_wins']

            home_form = sum(team_stats.get(home,{'form':[3]})['form'][-5:])
            away_form = sum(team_stats.get(away,{'form':[3]})['form'][-5:])
            home_avg = np.mean(team_stats.get(home,{'points_scored':[0]})['points_scored'][-15:])
            away_avg = np.mean(team_stats.get(away,{'points_scored':[0]})['points_scored'][-15:])
            home_allowed = np.mean(team_stats.get(home,{'points_allowed':[0]})['points_allowed'][-15:])
            away_allowed = np.mean(team_stats.get(away,{'points_allowed':[0]})['points_allowed'][-15:])
            last_home = team_stats.get(home,{'last_game':today})['last_game']
            last_away = team_stats.get(away,{'last_game':today})['last_game']
            rest_home = (datetime.now(central) - datetime.strptime(last_home,"%Y-%m-%d")).days if last_home else 0
            rest_away = (datetime.now(central) - datetime.strptime(last_away,"%Y-%m-%d")).days if last_away else 0

            X_pred = [[home_form, away_form, home_avg, away_avg, home_allowed, away_allowed,
                      h2h_home, h2h_away, 1, rest_home, rest_away]]
            prob = clf.predict_proba(X_pred)[0]
            conf = max(prob)
            pred = "Home" if prob[1] > prob[0] else "Away"
            rec = "Bet" if conf >= CONF_THRESHOLD else "Skip"

            predictions.append({
                "Date": today,
                "League": league,
                "Matchup": f"{away} @ {home}",
                "Predicted Winner": home if pred == "Home" else away,
                "Confidence": round(conf, 3),
                "Recommended Bet": rec,
                "Home Odds": g['bookmakers'][0]['markets'][0]['outcomes'][0]['price'],
                "Away Odds": g['bookmakers'][0]['markets'][0]['outcomes'][1]['price'],
                "Home Injuries": "; ".join(home_injury_list),
                "Away Injuries": "; ".join(away_injury_list)
            })
        except Exception as e:
            print(f"Error processing game: {e}")

df = pd.DataFrame(predictions)
df.to_csv("predictions_with_injuries.csv", index=False, encoding='utf-8')
print("\n✅ Predictions saved to predictions_with_injuries.csv with Home & Away injuries included.")
os.system("git add .")
os.system('git commit -m "Update predictions"')
os.system("git push origin main")
print("Pushed updated index.html to GitHub")
os.system("shutdown /s /t 60 /f")

