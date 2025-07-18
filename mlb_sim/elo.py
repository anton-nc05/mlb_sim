### elo.py
from typing import Tuple, Dict


def win_probability_from_elo_diff(diff: float) -> float:
    """Return win probability for diff = Elo_A - Elo_B."""
    return 1 / (1 + 10 ** (-diff / 400))


def update_elo(rating_a: float, rating_b: float, outcome_a: int, k_factor: float) -> Tuple[float, float]:
    """Update two Elo ratings given outcome for A (1 win, 0 loss)."""
    expected_a = win_probability_from_elo_diff(rating_a - rating_b)
    expected_b = 1 - expected_a
    rating_a_new = rating_a + k_factor * (outcome_a - expected_a)
    rating_b_new = rating_b + k_factor * ((1 - outcome_a) - expected_b)
    return rating_a_new, rating_b_new


def blend_elos(pre: Dict[str, float], hist: Dict[str, float], alpha: float) -> Dict[str, float]:
    """Blend preseason and historical Elos."""
    return {team: alpha * pre[team] + (1 - alpha) * hist[team] for team in pre.keys()}
