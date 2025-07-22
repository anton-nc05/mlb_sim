#main model + workflow

import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

#UPDATED
CONFIG = {
    'team_inputs': "Team_Inputs(1).csv",
    'historical': "Historical_data.csv",
    'odds': "mlb_odds_2025-07-22.csv",
    'elos': "inseason_elos_fixed.csv", #fixed file
    'poisson_weight': 0.75, #rebalance 
    'elo_weight': 0.25, 
    'min_confidence': 0.51, #threshold for more picks
    'hfa_adjustment': 1.025 #homefield adv 
}

TEAM_MAPPING = {
    #full names to codes
    'Arizona Diamondbacks': 'ARI', 'Atlanta Braves': 'ATL', 'Baltimore Orioles': 'BAL',
    'Boston Red Sox': 'BOS', 'Chicago Cubs': 'CHC', 'Chicago White Sox': 'CWS',
    'Cincinnati Reds': 'CIN', 'Cleveland Guardians': 'CLE', 'Colorado Rockies': 'COL',
    'Detroit Tigers': 'DET', 'Houston Astros': 'HOU', 'Kansas City Royals': 'KCR',
    'Los Angeles Angels': 'LAA', 'Los Angeles Dodgers': 'LAD', 'Miami Marlins': 'MIA',
    'Milwaukee Brewers': 'MIL', 'Minnesota Twins': 'MIN', 'New York Mets': 'NYM',
    'New York Yankees': 'NYY', 'Oakland Athletics': 'OAK', 'Philadelphia Phillies': 'PHI',
    'Pittsburgh Pirates': 'PIT', 'San Diego Padres': 'SDP', 'San Francisco Giants': 'SFG',
    'Seattle Mariners': 'SEA', 'St. Louis Cardinals': 'STL', 'Tampa Bay Rays': 'TBR',
    'Texas Rangers': 'TEX', 'Toronto Blue Jays': 'TOR', 'Washington Nationals': 'WSN',
    
    #alternative codes DO NOT CHANGE
    'CHN': 'CHC', 'LAN': 'LAD', 'SLN': 'STL', 'WAS': 'WSN', 'KC': 'KCR', 
    'TB': 'TBR', 'SD': 'SDP', 'SF': 'SFG'
}

REVERSE_TEAM_MAPPING = {v: k for k, v in TEAM_MAPPING.items() if len(k) == 3}

def poisson_pmf(k, lam):
    try:
        return lam**k * math.exp(-lam) / math.factorial(k)
    except (OverflowError, ValueError):
        return 0

def win_prob_poisson(lam_home, lam_away, max_runs=15):
    prob_home_wins = 0
    for home_runs in range(max_runs + 1):
        prob_home_score = poisson_pmf(home_runs, lam_home)
        if prob_home_score == 0:
            continue
            
        for away_runs in range(home_runs):
            prob_away_score = poisson_pmf(away_runs, lam_away)
            prob_home_wins += prob_home_score * prob_away_score
            
    return min(max(prob_home_wins, 0.01), 0.99)  

def win_prob_elo(elo_home, elo_away):
    try:
        diff = min(max(elo_away - elo_home, -800), 800)  
        return 1.0 / (1.0 + 10 ** (diff / 400.0))
    except (OverflowError, ZeroDivisionError):
        return 0.5

def calculate_kelly_bet_size(prob_win, odds):
    try:
        if odds > 0:
            decimal_odds = (odds / 100) + 1
        else:
            decimal_odds = (100 / abs(odds)) + 1
        
        edge = prob_win - (1 / decimal_odds)
        if edge <= 0:
            return 0
        
        kelly_fraction = edge / (decimal_odds - 1)
        return max(0, min(kelly_fraction, 0.25))  
    except:
        return 0

def load_and_validate_elos():
    try:
        df = pd.read_csv(CONFIG['elos'], parse_dates=['date'])
        print(f"Raw Elo Data: {len(df)} entries")
        df['team_standardized'] = df['team'].replace(TEAM_MAPPING)
        
        df_clean = df.drop_duplicates(['date', 'team_standardized']).copy()
        latest_date = df_clean['date'].max()
        df_latest = df_clean[df_clean['date'] == latest_date].copy()
        
        elo_dict = df_latest.groupby('team_standardized')['elo'].mean().to_dict()
        league_avg = sum(elo_dict.values()) / len(elo_dict) if elo_dict else 1500
        
        all_mlb_teams = ['ARI', 'ATL', 'BAL', 'BOS', 'CHC', 'CWS', 'CIN', 'CLE', 'COL',
                        'DET', 'HOU', 'KCR', 'LAA', 'LAD', 'MIA', 'MIL', 'MIN', 'NYM',
                        'NYY', 'OAK', 'PHI', 'PIT', 'SDP', 'SFG', 'SEA', 'STL', 'TBR',
                        'TEX', 'TOR', 'WSN']
        
        missing_teams = set(all_mlb_teams) - set(elo_dict.keys())
        
        if missing_teams:
            print(f"Missing Elo for {len(missing_teams)} teams, estimating from performance...")
            try:
                team_df = pd.read_csv(CONFIG['team_inputs'])
                team_df['Team'] = team_df['Team'].replace(TEAM_MAPPING)
                team_df['perf_score'] = (team_df['wRC+'] - 100) + (100 - (team_df['ERA'] - 4.00) * 25)
                perf_mean = team_df['perf_score'].mean()
                perf_std = team_df['perf_score'].std()
                
                for team in missing_teams:
                    if team in team_df['Team'].values:
                        team_perf = team_df[team_df['Team'] == team]['perf_score'].iloc[0]
                        estimated_elo = league_avg + (team_perf - perf_mean) / perf_std * 100
                        elo_dict[team] = max(1200, min(1800, estimated_elo))
                    else:
                        elo_dict[team] = league_avg  
                        
            except Exception as e:
                print(f"   Couldn't load team metrics for estimation: {e}")
                for team in missing_teams:
                    elo_dict[team] = league_avg
        
        print(f"Final Elo Ratings: {len(elo_dict)} teams (avg: {league_avg:.0f})")
        return elo_dict, league_avg
        
    except Exception as e:
        print(f"Error loading Elo data: {e}")
        print("   Using default Elo ratings...")
        
        #fallbackc basic Elo ratings
        default_elos = {
            'NYY': 1600, 'LAD': 1580, 'BAL': 1550, 'HOU': 1540, 'PHI': 1530,
            'ATL': 1520, 'MIL': 1510, 'SDP': 1500, 'NYM': 1490, 'ARI': 1480,
            'BOS': 1470, 'MIN': 1460, 'SEA': 1450, 'TOR': 1440, 'KCR': 1430,
            'SFG': 1420, 'STL': 1410, 'CHC': 1400, 'DET': 1490, 'TBR': 1380,
            'CLE': 1370, 'TEX': 1360, 'MIA': 1350, 'CIN': 1340, 'PIT': 1330,
            'LAA': 1320, 'WSN': 1310, 'OAK': 1300, 'COL': 1290, 'CWS': 1280
        }
        return default_elos, 1500

def load_and_validate_team_metrics():
    """Load team metrics with enhanced validation"""
    try:
        df = pd.read_csv(CONFIG['team_inputs'])
        df['Team'] = df['Team'].replace(TEAM_MAPPING)
        df['rpg'] = df['R'] / df['G'].replace(0, 1)
        df['off_rating'] = (df['wRC+'] / 100.0) * 0.7 + (df['rpg'] / 4.5) * 0.3
        
        league_era = df['ERA'].median()
        df['def_rating'] = (league_era / df['ERA'].replace(0, league_era)) * 0.8 + \
                          (df['FIP'] / league_era) * 0.2
        
        df['off_rating'] = np.clip(df['off_rating'], 0.75, 1.35)
        df['def_rating'] = np.clip(df['def_rating'], 0.75, 1.35)
        
        print(f"Team Metrics: {len(df)} teams loaded")
        print(f"   Offensive Range: {df['off_rating'].min():.3f} - {df['off_rating'].max():.3f}")
        print(f"   Defensive Range: {df['def_rating'].min():.3f} - {df['def_rating'].max():.3f}")
        
        return df.set_index('Team')[['off_rating', 'def_rating']].to_dict('index')
        
    except Exception as e:
        print(f"Error loading team metrics: {e}")
        return {}

def load_todays_games():
    """Load today's games with enhanced validation"""
    try:
        df = pd.read_csv(CONFIG['odds'])
        df['home_code'] = df['home_team'].replace(TEAM_MAPPING)
        df['away_code'] = df['away_team'].replace(TEAM_MAPPING)
        
        unmapped_home = df[df['home_code'] == df['home_team']]['home_team'].unique()
        unmapped_away = df[df['away_code'] == df['away_team']]['away_team'].unique()
        
        if len(unmapped_home) > 0 or len(unmapped_away) > 0:
            print(f"Some teams couldn't be mapped to codes:")
            for team in set(list(unmapped_home) + list(unmapped_away)):
                print(f"{team}")
        print(f"Today's Games: {len(df)} matchups loaded")

        return df
        
    except Exception as e:
        print(f"Error loading games: {e}")
        return pd.DataFrame()

def calculate_game_prediction(home_team, away_team, team_metrics, elos, league_rpg=4.5, hfa_factor=1.025):
    """Calculate comprehensive game prediction"""
    
    home_metrics = team_metrics.get(home_team, {'off_rating': 1.0, 'def_rating': 1.0})
    away_metrics = team_metrics.get(away_team, {'off_rating': 1.0, 'def_rating': 1.0})
    home_elo = elos.get(home_team, 1500)
    away_elo = elos.get(away_team, 1500)
    
    #run expectations
    home_runs_exp = (home_metrics['off_rating'] * 
                     away_metrics['def_rating'] * 
                     league_rpg * 
                     hfa_factor * 
                     CONFIG['hfa_adjustment'])
    
    away_runs_exp = (away_metrics['off_rating'] * 
                     home_metrics['def_rating'] * 
                     league_rpg)
    
    #calculate probabilities
    prob_poisson = win_prob_poisson(home_runs_exp, away_runs_exp)
    prob_elo = win_prob_elo(home_elo, away_elo)
    
    #weighted combo
    prob_combined = (CONFIG['poisson_weight'] * prob_poisson + 
                    CONFIG['elo_weight'] * prob_elo)
    
    return {
        'home_elo': home_elo,
        'away_elo': away_elo,
        'home_runs_exp': home_runs_exp,
        'away_runs_exp': away_runs_exp,
        'prob_poisson': prob_poisson,
        'prob_elo': prob_elo,
        'prob_combined': prob_combined,
        'total_runs_exp': home_runs_exp + away_runs_exp
    }

def generate_enhanced_predictions():
    """Enhanced prediction engine with robust error handling"""
    
    print("Enhanced MLB Prediction Engine v2.1")
    print("=" * 60)
    
    #load all data
    elos, league_elo = load_and_validate_elos()
    team_metrics = load_and_validate_team_metrics()
    games_df = load_todays_games()
    
    if games_df.empty:
        print("No games found to predict")
        return []
    
    #historical averages (can fix later)
    league_rpg = 4.5
    hfa_factor = 1.025
    
    print(f"\n League Settings:")
    print(f"   Runs/Game: {league_rpg}")
    print(f"   HFA Factor: {hfa_factor}")
    print(f"   Model Weights: {CONFIG['poisson_weight']:.0%} Poisson, {CONFIG['elo_weight']:.0%} Elo")
    
    print("\n PREDICTIONS")
    print("=" * 80)
    
    predictions = []
    high_confidence_picks = 0
    
    for _, game in games_df.iterrows():
        home_team = game['home_code']
        away_team = game['away_code']
        
        #skip unmapped teams
        if pd.isna(home_team) or pd.isna(away_team):
            print(f"  Skipping unmapped teams: {game['away_team']} @ {game['home_team']}")
            continue
        
        #calculate prediction
        pred = calculate_game_prediction(home_team, away_team, team_metrics, elos, league_rpg, hfa_factor)
        
        #determine pick
        confidence = abs(pred['prob_combined'] - 0.5)
        
        if pred['prob_combined'] >= CONFIG['min_confidence']:
            pick = game['home_team']
            pick_prob = pred['prob_combined']
        elif pred['prob_combined'] <= (1 - CONFIG['min_confidence']):
            pick = game['away_team']
            pick_prob = 1 - pred['prob_combined']
        else:
            pick = "PASS"
            pick_prob = max(pred['prob_combined'], 1 - pred['prob_combined'])
        
        kelly_size = 0
        if pick != "PASS" and 'home_ml' in game and pd.notna(game['home_ml']):
            if pick == game['home_team'] and pd.notna(game['home_ml']):
                kelly_size = calculate_kelly_bet_size(pick_prob, game['home_ml'])
            elif pick == game['away_team'] and 'away_ml' in game and pd.notna(game['away_ml']):
                kelly_size = calculate_kelly_bet_size(pick_prob, game['away_ml'])
        
        prediction = {
            'away_team': game['away_team'],
            'home_team': game['home_team'],
            'pick': pick,
            'confidence': confidence,
            'prob_combined': pick_prob,
            'kelly_size': kelly_size,
            **pred
        }
        
        predictions.append(prediction)
        status = " PICK" if pick != "PASS" else "PASS"
        confidence_pct = confidence * 100
        
        if pick != "PASS":
            high_confidence_picks += 1
            
        print(f"{status} {away_team} @ {home_team} → {pick}")
        print(f"  Combined: {pred['prob_combined']:.3f} | Confidence: {confidence_pct:.1f}%")
        print(f"  Poisson: {pred['prob_poisson']:.3f} | Elo: {pred['prob_elo']:.3f}")
        print(f"  Expected: {home_team} {pred['home_runs_exp']:.1f} - {away_team} {pred['away_runs_exp']:.1f}")
        
        if kelly_size > 0.02:  #only show if > 2
            print(f" Kelly Size: {kelly_size:.1%}")
        print()
    
    #summary
    total_games = len(predictions)
    picks_made = sum(1 for p in predictions if p['pick'] != "PASS")
    avg_confidence = np.mean([p['confidence'] for p in predictions if p['pick'] != "PASS"]) if picks_made > 0 else 0
    
    print(" SUMMARY")
    print(f"   Games Analyzed: {total_games}")
    print(f"   Picks Made: {picks_made} ({picks_made/total_games:.0%})")
    print(f"   Average Confidence: {avg_confidence:.1%}")
    
    if picks_made == 0:
        print(f"\nTry lowering min_confidence below {CONFIG['min_confidence']} to get more picks")
    
    return predictions

if __name__ == "__main__":
    predictions = generate_enhanced_predictions()