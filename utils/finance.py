# File: mlb_sim/utils/finance.py
"""
Financial helper functions for odds and bets.
"""

from typing import Tuple, Union
import numpy as np
import pandas as pd

def implied_prob_from_moneyline(ml: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray, pd.Series]:
    """
    Convert American moneyline odds to implied probability.

    Args:
        ml: Moneyline odds (e.g., +150 or -200).

    Returns:
        Implied probability.
    """
    ml_arr = np.array(ml, copy=False)
    prob = np.where(
        ml_arr > 0,
        100 / (ml_arr + 100),
        -ml_arr / (-ml_arr + 100)
    )
    if isinstance(ml, (pd.Series, np.ndarray)):
        return pd.Series(prob, index=getattr(ml, "index", None))
    return float(prob)

def remove_vig_additive(
    p1: Union[float, np.ndarray, pd.Series],
    p2: Union[float, np.ndarray, pd.Series]
) -> Tuple[Union[float, np.ndarray, pd.Series], Union[float, np.ndarray, pd.Series]]:
    """
    Remove bookmaker vig using additive method.

    Args:
        p1: Raw implied probability for side 1.
        p2: Raw implied probability for side 2.

    Returns:
        Tuple of normalized probabilities summing to 1.
    """
    total = p1 + p2
    return p1 / total, p2 / total

def kelly_fraction(
    p: float,
    odds: float
) -> float:
    """
    Calculate optimal Kelly fraction.

    Args:
        p: Probability of winning.
        odds: Decimal odds (payout per unit stake).

    Returns:
        Kelly fraction.
    """
    b = odds - 1
    q = 1 - p
    return (b * p - q) / b if b > 0 else 0.0
