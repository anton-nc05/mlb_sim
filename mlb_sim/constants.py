### constants.py
from pathlib import Path
import os

# Project root
ROOT = Path(__file__).resolve().parent

# Input and output paths
data_dir = ROOT / 'data'
HISTORICAL_DATA_PATH = data_dir / 'Historical_data.csv'
TEAM_INPUTS_PATH = data_dir / 'Team_Inputs(1).csv'
PRESEASON_ELO_PATH = ROOT / 'preseason_elos.csv'
HISTORICAL_ELO_PATH = ROOT / 'historical_elos.csv'
PREVIOUS_ELO_PATH = ROOT / 'previous_elos.csv'
INSEASON_ELO_PATH = ROOT / 'inseason_elos.csv'
DAILY_SUMMARY_PATH = ROOT / 'daily_summary.csv'

# Tunable constants
K_METRICS = 5
METRIC_COLUMNS = ['wRC+', 'ERA', 'WAR', 'ISO', 'OBP', 'SLG', 'K/9', 'BB/9', 'BABIP', 'DRS']
ALPHA = 0.5
K_FACTOR = 20
EDGE_THRESHOLD = 0.05

# Betting settings
INITIAL_BANKROLL = 1000.0
BET_AMOUNT = 100.0

# Odds API
ODDS_API_KEY = os.getenv('ODDS_API_KEY', '68256bffa0b9127296003dfddb6c8fca')
BASE_URL = 'https://api.the-odds-api.com/v4'