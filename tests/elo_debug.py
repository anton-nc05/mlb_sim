# Add this debug code to see what's wrong with your Elo data
import pandas as pd

def debug_elo_data():
    """Debug the Elo data loading"""
    try:
        df = pd.read_csv('inseason_elos.csv')
        print("📊 ELO FILE ANALYSIS")
        print("=" * 40)
        print(f"Columns: {df.columns.tolist()}")
        print(f"Shape: {df.shape}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Unique teams: {df['team'].nunique()}")
        print(f"Team list: {sorted(df['team'].unique())}")
        print("\nSample data:")
        print(df.head(10))
        
        # Check for team mapping issues
        from improved_mlb_simulator import TEAM_MAPPING
        unmapped = set(df['team'].unique()) - set(TEAM_MAPPING.keys()) - set(TEAM_MAPPING.values())
        if unmapped:
            print(f"\n⚠️ Unmapped teams in Elo file: {unmapped}")
            
    except Exception as e:
        print(f"Error loading Elo file: {e}")

# Run this to debug
debug_elo_data()