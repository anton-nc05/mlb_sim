#!/usr/bin/env python3
"""
Improved MLB Simulator with comprehensive validation and better modeling
"""

import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

CONFIG = {
    'team_inputs': "Team_Inputs(1).csv",
    'historical': "Historical_data.csv", 
    'odds': "mlb_odds_2025-07-20.csv",
    'elos': "inseason_elos.csv",
    'poisson_weight': 0.8,    # Increase since Elo isn't working well
    'elo_weight': 0.2,        # Decrease until Elo is fixed
    'min_confidence': 0.52,   # Lower threshold to 52%
    'hfa_adjustment': 1.025   # Slightly lower HFA
}

# Comprehensive team name standardization
TEAM_MAPPING = {
    # Full names to codes
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
    
    # Alternative codes to standard
    'CHN': 'CHC', 'LAN': 'LAD', 'SLN': 'STL', 'WAS': 'WSN', 'KC': 'KCR', 
    'TB': 'TBR', 'SD': 'SDP', 'SF': 'SFG'
}

# ─── STATISTICAL FUNCTIONS ───────────────────────────────────────────────────

def poisson_pmf(k, lam):
    """Poisson probability mass function"""
    return lam**k * math.exp(-lam) / math.factorial(k)

def win_prob_poisson(lam_home, lam_away, max_runs=20):
    """Calculate win probability using Poisson distribution"""
    prob_home_wins = 0
    for home_runs in range(max_runs + 1):
        prob_home_score = poisson_pmf(home_runs, lam_home)
        for away_runs in range(home_runs):  # Home wins if they score more
            prob_away_score = poisson_pmf(away_runs, lam_away)
            prob_home_wins += prob_home_score * prob_away_score
    return prob_home_wins

def win_prob_elo(elo_home, elo_away):
    """Standard Elo win probability with error handling"""
    try:
        return 1.0 / (1.0 + 10 ** ((elo_away - elo_home) / 400.0))
    except (OverflowError, ZeroDivisionError):
        return 0.5

def calculate_kelly_bet_size(prob_win, odds):
    """Calculate Kelly criterion bet size"""
    if odds > 0:
        decimal_odds = (odds / 100) + 1
    else:
        decimal_odds = (100 / abs(odds)) + 1
    
    edge = prob_win - (1 / decimal_odds)
    if edge <= 0:
        return 0
    
    kelly_fraction = edge / (decimal_odds - 1)
    return max(0, min(kelly_fraction, 0.25))  # Cap at 25%

# ─── DATA LOADING & VALIDATION ───────────────────────────────────────────────

def load_and_validate_historical():
    """Load historical data with validation"""
    try:
        df = pd.read_csv(CONFIG['historical'], parse_dates=['date'])
        df = df[df['gametype'] == 'regular'].copy()
        
        if len(df) < 1000:
            print(f"⚠️  Warning: Only {len(df)} historical games found")
        
        # Calculate league averages
        home_rpg = df['hruns'].mean()
        away_rpg = df['vruns'].mean()
        total_rpg = (home_rpg + away_rpg) / 2
        hfa_factor = home_rpg / away_rpg
        
        print(f"📊 Historical Data: {len(df)} games")
        print(f"   League R/G: {total_rpg:.2f}")
        print(f"   HFA Factor: {hfa_factor:.3f}")
        
        return total_rpg, hfa_factor
        
    except FileNotFoundError:
        print("⚠️  Historical data not found, using MLB averages")
        return 4.5, 1.025  # Typical MLB values

def load_and_validate_team_metrics():
    """Load team metrics with comprehensive validation"""
    try:
        df = pd.read_csv(CONFIG['team_inputs'])
        
        # Standardize team names
        df['Team'] = df['Team'].replace(TEAM_MAPPING)
        
        # Validate essential columns
        required_cols = ['Team', 'R', 'G', 'ERA', 'wRC+']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # Calculate metrics with error handling
        df['rpg'] = df['R'] / df['G'].replace(0, 1)  # Avoid division by zero
        df['era_adj'] = pd.to_numeric(df['ERA'], errors='coerce').fillna(4.5)
        
        # More sophisticated offensive rating using wRC+
        df['off_rating'] = df['wRC+'] / 100.0  # wRC+ is already park/league adjusted
        
        # Defensive rating from ERA (lower ERA = better defense)
        league_era = df['era_adj'].median()  # Use median to avoid outlier influence
        df['def_rating'] = league_era / df['era_adj']
        
        # Cap extreme values
        df['off_rating'] = np.clip(df['off_rating'], 0.7, 1.4)
        df['def_rating'] = np.clip(df['def_rating'], 0.7, 1.4)
        
        print(f"📈 Team Metrics: {len(df)} teams loaded")
        print(f"   Off Rating Range: {df['off_rating'].min():.2f} - {df['off_rating'].max():.2f}")
        print(f"   Def Rating Range: {df['def_rating'].min():.2f} - {df['def_rating'].max():.2f}")
        
        return df.set_index('Team')[['off_rating', 'def_rating']].to_dict('index')
        
    except Exception as e:
        print(f"⚠️  Error loading team metrics: {e}")
        return {}

def load_and_validate_elos():
    """Load and validate Elo ratings with comprehensive checks"""
    try:
        df = pd.read_csv(CONFIG['elos'], parse_dates=['date'])
        
        # Standardize team codes
        df['team'] = df['team'].replace(TEAM_MAPPING)
        
        # Get most recent rating for each team
        latest_date = df['date'].max()
        df_latest = df[df['date'] == latest_date].copy()
        
        # Remove duplicates by taking mean
        elo_dict = df_latest.groupby('team')['elo'].mean().to_dict()
        
        # Validation checks
        expected_teams = 30
        if len(elo_dict) < expected_teams:
            print(f"⚠️  Warning: Only {len(elo_dict)} teams have Elo ratings")
        
        # Calculate league average
        league_avg = sum(elo_dict.values()) / len(elo_dict) if elo_dict else 1500
        
        # Check for reasonable Elo values (typically 1200-1800)
        extreme_elos = {k: v for k, v in elo_dict.items() if v < 1200 or v > 1800}
        if extreme_elos:
            print(f"⚠️  Extreme Elo values found: {extreme_elos}")
        
        print(f"🎯 Elo Ratings: {len(elo_dict)} teams (avg: {league_avg:.0f})")
        
        return elo_dict, league_avg
        
    except Exception as e:
        print(f"⚠️  Error loading Elo data: {e}")
        return {}, 1500

def load_todays_games():
    """Load today's games with validation"""
    try:
        df = pd.read_csv(CONFIG['odds'])
        
        # Convert date column if needed
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # Standardize team names
        df['home_code'] = df['home_team'].replace(TEAM_MAPPING)
        df['away_code'] = df['away_team'].replace(TEAM_MAPPING)
        
        # Find teams that weren't successfully mapped (still full names)
        unmapped_home = df[df['home_code'] == df['home_team']]['home_team'].unique()
        unmapped_away = df[df['away_code'] == df['away_team']]['away_team'].unique()
        
        unmapped_teams = set(list(unmapped_home) + list(unmapped_away))
        
        if unmapped_teams:
            print(f"⚠️  Unmapped teams: {unmapped_teams}")
            # Still proceed with available mappings
        
        print(f"🏟️  Today's Games: {len(df)} matchups")
        
        return df
        
    except Exception as e:
        print(f"⚠️  Error loading odds: {e}")
        import traceback
        traceback.print_exc()  # This will show the full error
        return pd.DataFrame()


# ─── ENHANCED PREDICTION ENGINE ──────────────────────────────────────────────

def calculate_advanced_metrics(home_team, away_team, team_metrics, elos, league_rpg, hfa_factor):
    """Calculate advanced prediction metrics"""
    
    # Get team ratings with defaults
    home_metrics = team_metrics.get(home_team, {'off_rating': 1.0, 'def_rating': 1.0})
    away_metrics = team_metrics.get(away_team, {'off_rating': 1.0, 'def_rating': 1.0})
    home_elo = elos.get(home_team, 1500)
    away_elo = elos.get(away_team, 1500)
    
    # Enhanced run expectation model
    home_runs_expected = (home_metrics['off_rating'] * 
                         away_metrics['def_rating'] * 
                         league_rpg * 
                         hfa_factor * 
                         CONFIG['hfa_adjustment'])
    
    away_runs_expected = (away_metrics['off_rating'] * 
                         home_metrics['def_rating'] * 
                         league_rpg)
    
    # Calculate probabilities
    prob_poisson = win_prob_poisson(home_runs_expected, away_runs_expected)
    prob_elo = win_prob_elo(home_elo, away_elo)
    
    # Weighted combination
    prob_combined = (CONFIG['poisson_weight'] * prob_poisson + 
                    CONFIG['elo_weight'] * prob_elo)
    
    return {
        'home_elo': home_elo,
        'away_elo': away_elo,
        'home_runs_exp': home_runs_expected,
        'away_runs_exp': away_runs_expected,
        'prob_poisson': prob_poisson,
        'prob_elo': prob_elo,
        'prob_combined': prob_combined,
        'total_runs_exp': home_runs_expected + away_runs_expected
    }

def generate_predictions():
    """Main prediction engine with comprehensive analysis"""
    
    print("🚀 MLB Prediction Engine v2.0")
    print("=" * 50)
    
    # Load all data
    league_rpg, hfa_factor = load_and_validate_historical()
    team_metrics = load_and_validate_team_metrics()
    elos, league_elo = load_and_validate_elos()
    games_df = load_todays_games()
    
    if games_df.empty:
        print("❌ No games to predict")
        return
    
    print("\n📋 PREDICTIONS")
    print("=" * 80)
    
    predictions = []
    
    for _, game in games_df.iterrows():
        home_team = game['home_code']
        away_team = game['away_code']
        
        # Skip if team mapping failed
        if pd.isna(home_team) or pd.isna(away_team):
            continue
        
        # Calculate metrics
        metrics = calculate_advanced_metrics(
            home_team, away_team, team_metrics, elos, league_rpg, hfa_factor
        )
        
        # Determine pick
        confidence = abs(metrics['prob_combined'] - 0.5)
        if metrics['prob_combined'] >= CONFIG['min_confidence']:
            pick = game['home_team']
            pick_prob = metrics['prob_combined']
        elif metrics['prob_combined'] <= (1 - CONFIG['min_confidence']):
            pick = game['away_team'] 
            pick_prob = 1 - metrics['prob_combined']
        else:
            pick = "PASS"
            pick_prob = max(metrics['prob_combined'], 1 - metrics['prob_combined'])
        
        # Calculate value if odds available
        kelly_size = 0
        if pick != "PASS" and 'home_ml' in game and pd.notna(game['home_ml']):
            if pick == game['home_team']:
                kelly_size = calculate_kelly_bet_size(pick_prob, game['home_ml'])
            elif 'away_ml' in game:
                kelly_size = calculate_kelly_bet_size(pick_prob, game['away_ml'])
        
        prediction = {
            'away_team': game['away_team'],
            'home_team': game['home_team'], 
            'pick': pick,
            'confidence': confidence,
            'prob_combined': pick_prob,
            'kelly_size': kelly_size,
            'total_runs': metrics['total_runs_exp'],
            **metrics
        }
        
        predictions.append(prediction)
        
        # Display prediction
        status_emoji = "✅" if pick != "PASS" else "⏸️"
        print(f"{status_emoji} {away_team}@{home_team} → {pick}")
        print(f"   Combined: {metrics['prob_combined']:.3f} | Poisson: {metrics['prob_poisson']:.3f} | Elo: {metrics['prob_elo']:.3f}")
        print(f"   Expected: {home_team} {metrics['home_runs_exp']:.1f} - {away_team} {metrics['away_runs_exp']:.1f}")
        if kelly_size > 0:
            print(f"   Kelly Size: {kelly_size:.1%}")
        print()
    
    # Summary statistics
    total_picks = sum(1 for p in predictions if p['pick'] != "PASS")
    avg_confidence = np.mean([p['confidence'] for p in predictions if p['pick'] != "PASS"])
    
    print("📊 SUMMARY")
    print(f"   Games Analyzed: {len(predictions)}")
    print(f"   Picks Made: {total_picks}")
    print(f"   Average Confidence: {avg_confidence:.1%}")
    
    return predictions

if __name__ == "__main__":
    predictions = generate_predictions()