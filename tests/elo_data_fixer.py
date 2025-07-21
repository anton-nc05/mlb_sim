#!/usr/bin/env python3
"""
Elo Data Fixer - Addresses the specific issues in your Elo dataset
"""

import pandas as pd
import numpy as np
from datetime import datetime

def fix_elo_data():
    """Fix the problematic Elo data"""
    
    print("🔧 FIXING ELO DATA")
    print("=" * 40)
    
    # Load the problematic data
    try:
        df = pd.read_csv('inseason_elos.csv')
        print(f"Loaded {len(df)} Elo entries")
        
        # Team code mapping for the old codes in your data
        code_mapping = {
            'CHN': 'CHC',  # Chicago Cubs
            'LAN': 'LAD',  # Los Angeles Dodgers  
            'SLN': 'STL',  # St. Louis Cardinals
            'WAS': 'WSN'   # Washington Nationals
        }
        
        # Fix team codes
        df['team'] = df['team'].replace(code_mapping)
        
        # Remove duplicates (you have lots of them!)
        df_clean = df.drop_duplicates(['date', 'team']).copy()
        print(f"After removing duplicates: {len(df_clean)} entries")
        
        # Get the unique teams we have data for
        existing_teams = set(df_clean['team'].unique())
        print(f"Teams with Elo data: {sorted(existing_teams)}")
        
        # All 30 MLB teams
        all_mlb_teams = {
            'ARI', 'ATL', 'BAL', 'BOS', 'CHC', 'CWS', 'CIN', 'CLE', 'COL',
            'DET', 'HOU', 'KCR', 'LAA', 'LAD', 'MIA', 'MIL', 'MIN', 'NYM',
            'NYY', 'OAK', 'PHI', 'PIT', 'SDP', 'SFG', 'SEA', 'STL', 'TBR',
            'TEX', 'TOR', 'WSN'
        }
        
        missing_teams = all_mlb_teams - existing_teams
        print(f"Missing teams: {sorted(missing_teams)}")
        
        # Load team performance data to estimate missing Elos
        try:
            team_stats = pd.read_csv('Team_Inputs(1).csv')
            
            # Create performance score (higher = better)
            team_stats['perf_score'] = (team_stats['wRC+'] - 100) - (team_stats['ERA'] - 4.00) * 20
            
            # Calculate league average Elo from existing data
            league_avg_elo = df_clean['elo'].mean()
            perf_avg = team_stats['perf_score'].mean()
            perf_std = team_stats['perf_score'].std()
            
            print(f"League average Elo: {league_avg_elo:.0f}")
            
            # Create synthetic Elo data for missing teams
            synthetic_data = []
            latest_date = df_clean['date'].iloc[0]  # Use the same date
            
            for team in missing_teams:
                team_row = team_stats[team_stats['Team'] == team]
                
                if len(team_row) > 0:
                    perf = team_row['perf_score'].iloc[0]
                    # Convert performance to Elo (roughly)
                    estimated_elo = league_avg_elo + (perf - perf_avg) / perf_std * 80
                    estimated_elo = max(1200, min(1800, estimated_elo))  # Reasonable bounds
                else:
                    estimated_elo = league_avg_elo  # Default if no stats found
                
                synthetic_data.append({
                    'date': latest_date,
                    'team': team,
                    'elo': estimated_elo
                })
                
                print(f"  {team}: {estimated_elo:.0f} (estimated)")
            
            # Combine existing clean data with synthetic data
            synthetic_df = pd.DataFrame(synthetic_data)
            complete_df = pd.concat([df_clean, synthetic_df], ignore_index=True)
            
            # Save the fixed data
            complete_df.to_csv('inseason_elos_fixed.csv', index=False)
            print(f"\n✅ Fixed Elo data saved to 'inseason_elos_fixed.csv'")
            print(f"   Total teams: {complete_df['team'].nunique()}")
            print(f"   Date: {complete_df['date'].iloc[0]}")
            
            # Show sample of fixed data
            print(f"\nSample of fixed data:")
            sample = complete_df.groupby('team')['elo'].first().sort_values(ascending=False).head(10)
            for team, elo in sample.items():
                print(f"  {team}: {elo:.0f}")
                
        except FileNotFoundError:
            print("⚠️  Could not find Team_Inputs(1).csv for performance estimation")
            print("   Creating basic Elo estimates...")
            
            # Fallback: create basic estimates
            synthetic_data = []
            latest_date = df_clean['date'].iloc[0]
            base_elo = df_clean['elo'].mean()
            
            # Rough team tiers (you can adjust these)
            elite_teams = ['NYY', 'LAD', 'HOU', 'ATL']
            good_teams = ['PHI', 'BAL', 'MIL', 'SDP', 'NYM', 'ARI']
            average_teams = ['BOS', 'MIN', 'SEA', 'TOR', 'KCR', 'SFG', 'STL', 'CHC']
            poor_teams = ['DET', 'TBR', 'CLE', 'TEX', 'MIA', 'CIN', 'PIT', 'LAA', 'WSN', 'OAK', 'COL', 'CWS']
            
            for team in missing_teams:
                if team in elite_teams:
                    elo = base_elo + 60
                elif team in good_teams:
                    elo = base_elo + 30
                elif team in average_teams:
                    elo = base_elo
                else:
                    elo = base_elo - 40
                    
                synthetic_data.append({
                    'date': latest_date,
                    'team': team,
                    'elo': elo
                })
            
            synthetic_df = pd.DataFrame(synthetic_data)
            complete_df = pd.concat([df_clean, synthetic_df], ignore_index=True)
            complete_df.to_csv('inseason_elos_fixed.csv', index=False)
            print(f"✅ Basic fixed Elo data saved")
        
    except FileNotFoundError:
        print("❌ Could not find inseason_elos.csv")
        return False
    
    return True

def update_simulator_config():
    """Update the simulator to use the fixed Elo file"""
    
    config_update = """
# Update your CONFIG dictionary in improved_mlb_simulator.py:
CONFIG = {
    'team_inputs': "Team_Inputs(1).csv",
    'historical': "Historical_data.csv", 
    'odds': "mlb_odds_2025-07-20.csv",
    'elos': "inseason_elos_fixed.csv",  # <-- Use the fixed file
    'poisson_weight': 0.75,  # Rebalance the weights
    'elo_weight': 0.25,      # Now that Elo is more complete
    'min_confidence': 0.51,   # Lower threshold for more picks
    'hfa_adjustment': 1.025
}
"""
    print("\n📝 CONFIGURATION UPDATE")
    print("=" * 40)
    print(config_update)

if __name__ == "__main__":
    if fix_elo_data():
        update_simulator_config()
        print("\n🚀 Ready to run improved predictions!")
        print("   Run: python improved_mlb_simulator.py")
    else:
        print("\n❌ Fix failed - check your file paths")