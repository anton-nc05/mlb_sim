from mlb_sim.historical_elo import run_historical_elo
from mlb_sim.preseason_elo   import compute_weights_ridge

# 1a) Historical Elo
run_historical_elo(
    historical_results_path="data/Historical_data.csv",
    config_path="mlb_sim/config.yml",
    output_elos_path="mlb_sim/historical_elos.csv",
    output_log_path="mlb_sim/elo_log.csv"
)

# 1b) Preseason Elo
compute_weights_ridge(
    team_metrics_path="data/Team_Inputs(1).csv",
    historical_elos_path="mlb_sim/historical_elos.csv",
    output_path="mlb_sim/preseason_elos.csv",
    alpha=1.0
)
print("✅ Elo files created.")
