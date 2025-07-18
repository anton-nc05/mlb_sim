# File: mlb_sim/run_season.py
"""
CLI to orchestrate the full MLB season simulation pipeline
using your actual data filenames and correct ordering.
"""

import argparse
import os
from importlib.machinery import SourceFileLoader

def load_module(name: str, relpath: str):
    """
    Dynamically load a module from a .py file relative to PROJECT_ROOT.
    """
    full_path = os.path.join(PROJECT_ROOT, relpath)
    return SourceFileLoader(name, full_path).load_module()

def run_pipeline(date_str: str):
    global PROJECT_ROOT
    PROJECT_ROOT = os.getcwd()

    # Paths to your actual files
    team_path      = os.path.join(PROJECT_ROOT, "data", "Team_Inputs(1).csv")
    hist_data_path = os.path.join(PROJECT_ROOT, "data", "Historical_data.csv")
    cfg_path       = os.path.join(PROJECT_ROOT, "mlb_sim", "config.yml")
    out_dir        = os.path.join(PROJECT_ROOT, "mlb_sim")

    # Load modules dynamically
    pp        = load_module("preprocessing",    "mlb_sim/preprocessing.py")
    hist_mod  = load_module("historical_elo",   "mlb_sim/historical_elo.py")
    pre_mod   = load_module("preseason_elo",    "mlb_sim/preseason_elo.py")
    blend_mod = load_module("blend_elos",       "mlb_sim/blend_elos.py")
    odds_mod  = load_module("odds_calculation", "mlb_sim/odds_calculation.py")

    # 1) Historical Elo → writes historical_elos.csv & elo_log.csv
    hist_mod.run_historical_elo(
        historical_results_path=hist_data_path,
        config_path=cfg_path,
        output_elos_path=os.path.join(out_dir, "historical_elos.csv"),
        output_log_path=os.path.join(out_dir, "elo_log.csv"),
    )

    # 2) Preseason Elo → reads historical_elos.csv, writes preseason_elos.csv
    pre_mod.compute_weights_ridge(
        team_metrics_path=team_path,
        historical_elos_path=os.path.join(out_dir, "historical_elos.csv"),
        output_path=os.path.join(out_dir, "preseason_elos.csv"),
        alpha=1.0,
    )

    # 3) Blend Elos → writes blended_elos.csv
    blend_mod.blend_elos(
        preseason_elos_path=os.path.join(out_dir, "preseason_elos.csv"),
        historical_elos_path=os.path.join(out_dir, "historical_elos.csv"),
        games_played=0,
        output_path=os.path.join(out_dir, "blended_elos.csv"),
        config_path=cfg_path,
    )

    # 4) Odds & Picks → writes Daily_Picks_YYYYMMDD.csv
    odds_mod.calculate_daily_picks(
        hist_results_path=hist_data_path,
        preseason_elos_path=os.path.join(out_dir, "preseason_elos.csv"),
        historical_elos_path=os.path.join(out_dir, "historical_elos.csv"),
        date=date_str,
        output_path=os.path.join(out_dir, f"Daily_Picks_{date_str.replace('-', '')}.csv"),
        config_path=cfg_path,
    )

    print("✅ Full simulation pipeline complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MLB season sim")
    parser.add_argument("--date", required=True, help="YYYY-MM-DD for daily picks")
    args = parser.parse_args()
    run_pipeline(args.date)
