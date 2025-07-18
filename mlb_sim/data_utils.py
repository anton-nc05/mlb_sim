### data_utils.py

import pandas as pd
import numpy as np
from .constants import HISTORICAL_DATA_PATH, TEAM_INPUTS_PATH, PRESEASON_ELO_PATH, HISTORICAL_ELO_PATH
from .elo import blend_elos


def compute_preseason_elos(k_metrics: int, metric_columns: list) -> pd.DataFrame:
    """Compute preseason Elos from team metrics."""
    df = pd.read_csv(TEAM_INPUTS_PATH)
    cols = metric_columns[:k_metrics]
    # Z-score each metric column
    z = df[cols].apply(lambda x: (x - x.mean()) / x.std(ddof=0), axis=0)
    df['average_z'] = z.mean(axis=1)
    # Map to 1400-1700
    min_z, max_z = df['average_z'].min(), df['average_z'].max()
    df['preseason_elo'] = ((df['average_z'] - min_z) / (max_z - min_z)) * 300 + 1400
    df[['Team', 'preseason_elo']].to_csv(PRESEASON_ELO_PATH, index=False)
    return df[['Team', 'preseason_elo']]


def compute_historical_elos(k_factor: int) -> pd.DataFrame:
    """Compute historical Elos from game results."""
    df = pd.read_csv(HISTORICAL_DATA_PATH, parse_dates=['date'])
    df = df.sort_values('date')
    # Initialize at 1400
    teams = pd.unique(df[['hometeam', 'visteam']].values.ravel())
    elos = {team: 1400.0 for team in teams}
    from .elo import update_elo
    for _, row in df.iterrows():
        a, b = row['hometeam'], row['visteam']
        outcome_a = 1 if row['hruns'] > row['vruns'] else 0
        ra, rb = elos[a], elos[b]
        na, nb = update_elo(ra, rb, outcome_a, k_factor)
        elos[a], elos[b] = na, nb
    hist_df = pd.DataFrame({'team': list(elos.keys()), 'historical_elo': list(elos.values())})
    hist_df.to_csv(HISTORICAL_ELO_PATH, index=False)
    return hist_df
