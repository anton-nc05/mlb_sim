# File: mlb_sim/utils/elo.py
"""
Generic Elo rating helpers.
"""

import math
from typing import Tuple

def expected_result(
    rating_a: float,
    rating_b: float,
    hfa: float = 0.0,
    scale: float = 400.0
) -> float:
    """
    Calculate expected score for team A vs team B.

    Args:
        rating_a: Elo rating for team A.
        rating_b: Elo rating for team B.
        hfa: Home field advantage added to team A rating.
        scale: Elo rating scale.

    Returns:
        Expected probability of team A winning.
    """
    return 1 / (1 + 10 ** ((rating_b - (rating_a + hfa)) / scale))

def update_ratings(
    rating_a: float,
    rating_b: float,
    score_a: float,
    k: float,
    multiplier: float = 1.0,
    hfa: float = 0.0,
    scale: float = 400.0
) -> Tuple[float, float]:
    """
    Update Elo ratings for two teams based on outcome.

    Args:
        rating_a: Current rating for team A.
        rating_b: Current rating for team B.
        score_a: Actual score (1 for win, 0 for loss for team A).
        k: K-factor.
        multiplier: Margin-of-victory multiplier.
        hfa: Home field advantage.
        scale: Elo scale denominator.

    Returns:
        Tuple of updated ratings (rating_a_new, rating_b_new).
    """
    exp_a = expected_result(rating_a, rating_b, hfa, scale)
    delta = k * multiplier * (score_a - exp_a)
    return rating_a + delta, rating_b - delta
