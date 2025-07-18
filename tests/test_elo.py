# File: tests/test_elo.py
import os
import pytest
import numpy as np
from importlib.machinery import SourceFileLoader

# Dynamically load modules without relying on package imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

elo_mod = SourceFileLoader(
    "elo", os.path.join(PROJECT_ROOT, "mlb_sim", "utils", "elo.py")
).load_module()
finance_mod = SourceFileLoader(
    "finance", os.path.join(PROJECT_ROOT, "mlb_sim", "utils", "finance.py")
).load_module()
mr_mod = SourceFileLoader(
    "mean_reversion", os.path.join(PROJECT_ROOT, "mlb_sim", "mean_reversion.py")
).load_module()

# Aliases for the functions under test
expected_result = elo_mod.expected_result
update_ratings = elo_mod.update_ratings
implied_prob_from_moneyline = finance_mod.implied_prob_from_moneyline
remove_vig_additive = finance_mod.remove_vig_additive
kelly_fraction = finance_mod.kelly_fraction
fit_ou = mr_mod.fit_ou
apply_ou = mr_mod.apply_ou

def test_expected_result_equal_ratings():
    assert expected_result(1500, 1500) == pytest.approx(0.5, rel=1e-6)

def test_update_ratings_win():
    new_a, new_b = update_ratings(1500, 1500, score_a=1, k=32)
    # expected change = 32 * (1 - 0.5) = 16
    assert new_a == pytest.approx(1516)
    assert new_b == pytest.approx(1484)

def test_implied_prob_from_moneyline_positive():
    assert implied_prob_from_moneyline(150) == pytest.approx(100 / 250)

def test_implied_prob_from_moneyline_negative():
    assert implied_prob_from_moneyline(-150) == pytest.approx(150 / 250)

def test_remove_vig_additive():
    p1, p2 = remove_vig_additive(0.6, 0.5)
    assert p1 + p2 == pytest.approx(1.0)

def test_kelly_fraction():
    # decimal odds 3.0, p=0.6 => b=2, q=0.4 => (2*0.6-0.4)/2 = 0.4
    assert kelly_fraction(0.6, 3.0) == pytest.approx(0.4)

def test_fit_apply_ou_constant_series():
    series = [100, 100, 100, 100]
    theta, sigma = fit_ou(series)
    # sigma should be zero (no variance) for a constant series
    assert sigma == pytest.approx(0.0, abs=1e-6)
    updated = apply_ou(100, dt=1.0, theta=theta, sigma=sigma, mu=100)
    assert updated == pytest.approx(100.0)
