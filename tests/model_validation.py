#!/usr/bin/env python3
"""
Model Validation and Backtesting Framework
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, brier_score_loss
from scipy import stats

def validate_predictions(predictions_df, results_df):
    """
    Comprehensive model validation
    
    Args:
        predictions_df: DataFrame with columns ['date', 'home_team', 'away_team', 'prob_home', 'pick']
        results_df: DataFrame with actual results ['date', 'home_team', 'away_team', 'home_score', 'away_score']
    """
    
    # Merge predictions with results
    merged = pd.merge(
        predictions_df, 
        results_df, 
        on=['date', 'home_team', 'away_team'],
        how='inner'
    )
    
    if len(merged) == 0:
        print("❌ No matching predictions and results found")
        return None
    
    # Calculate actual outcomes
    merged['home_won'] = (merged['home_score'] > merged['away_score']).astype(int)
    merged['pick_correct'] = (
        ((merged['pick'] == merged['home_team']) & merged['home_won']) |
        ((merged['pick'] == merged['away_team']) & ~merged['home_won'])
    )
    
    print("🔍 MODEL VALIDATION RESULTS")
    print("=" * 50)
    
    # Basic accuracy metrics
    accuracy = merged['pick_correct'].mean()
    total_picks = len(merged)
    correct_picks = merged['pick_correct'].sum()
    
    print(f"Overall Accuracy: {accuracy:.1%} ({correct_picks}/{total_picks})")
    
    # Calibration analysis
    prob_bins = pd.cut(merged['prob_home'], bins=10, labels=False)
    calibration_data = []
    
    for bin_idx in range(10):
        bin_mask = prob_bins == bin_idx
        if bin_mask.sum() == 0:
            continue
            
        bin_data = merged[bin_mask]
        predicted_prob = bin_data['prob_home'].mean()
        actual_rate = bin_data['home_won'].mean()
        count = len(bin_data)
        
        calibration_data.append({
            'bin': bin_idx,
            'predicted_prob': predicted_prob,
            'actual_rate': actual_rate,
            'count': count,
            'calibration_error': abs(predicted_prob - actual_rate)
        })
    
    calib_df = pd.DataFrame(calibration_data)
    mean_calib_error = calib_df['calibration_error'].mean()
    
    print(f"Mean Calibration Error: {mean_calib_error:.3f}")
    
    # Probabilistic scoring
    try:
        brier_score = brier_score_loss(merged['home_won'], merged['prob_home'])
        log_loss_score = log_loss(merged['home_won'], merged['prob_home'])
        print(f"Brier Score: {brier_score:.3f} (lower is better)")
        print(f"Log Loss: {log_loss_score:.3f} (lower is better)")
    except Exception as e:
        print(f"⚠️  Error calculating probabilistic scores: {e}")
    
    # Performance by confidence level
    merged['confidence'] = abs(merged['prob_home'] - 0.5)
    confidence_bins = pd.qcut(merged['confidence'], q=3, labels=['Low', 'Medium', 'High'])
    
    print("\n📊 Performance by Confidence:")
    for conf_level in ['Low', 'Medium', 'High']:
        subset = merged[confidence_bins == conf_level]
        if len(subset) > 0:
            acc = subset['pick_correct'].mean()
            count = len(subset)
            print(f"   {conf_level}: {acc:.1%} ({count} games)")
    
    # ROI calculation (if odds available)
    if 'home_ml' in merged.columns and 'away_ml' in merged.columns:
        merged['bet_return'] = calculate_bet_returns(merged)
        total_roi = merged['bet_return'].sum()
        avg_roi_per_bet = merged['bet_return'].mean()
        
        print(f"\n💰 Betting Performance:")
        print(f"   Total ROI: {total_roi:+.1f} units")
        print(f"   ROI per bet: {avg_roi_per_bet:+.3f} units")
        print(f"   ROI %: {(total_roi/len(merged)*100):+.1f}%")
    
    return {
        'accuracy': accuracy,
        'calibration_error': mean_calib_error,
        'brier_score': brier_score if 'brier_score' in locals() else None,
        'total_games': total_picks,
        'calibration_data': calib_df
    }

def calculate_bet_returns(df):
    """Calculate returns from betting strategy"""
    returns = []
    
    for _, row in df.iterrows():
        if row['pick'] == row['home_team']:
            # Bet on home team
            if row['pick_correct']:
                if row['home_ml'] > 0:
                    return_val = row['home_ml'] / 100  # +150 → 1.5 units profit
                else:
                    return_val = 100 / abs(row['home_ml'])  # -150 → 0.67 units profit
            else:
                return_val = -1  # Lost bet
        else:
            # Bet on away team
            if row['pick_correct']:
                if row['away_ml'] > 0:
                    return_val = row['away_ml'] / 100
                else:
                    return_val = 100 / abs(row['away_ml'])
            else:
                return_val = -1
        
        returns.append(return_val)
    
    return returns

def run_monte_carlo_simulation(model_accuracy, num_bets=100, num_simulations=10000):
    """
    Monte Carlo simulation of betting outcomes
    """
    results = []
    
    for _ in range(num_simulations):
        # Simulate random bet outcomes
        outcomes = np.random.binomial(1, model_accuracy, num_bets)
        # Assume average odds of -110 (need 52.4% to break even)
        wins = outcomes.sum()
        losses = num_bets - wins
        profit = wins * 0.909 - losses * 1.0  # -110 odds
        results.append(profit)
    
    results = np.array(results)
    
    print(f"\n🎲 Monte Carlo Simulation ({num_simulations:,} trials):")
    print(f"   Mean Profit: {results.mean():+.1f} units")
    print(f"   Profit Range: {results.min():+.1f} to {results.max():+.1f} units")
    print(f"   Probability of Profit: {(results > 0).mean():.1%}")
    print(f"   90% Confidence Interval: [{np.percentile(results, 5):+.1f}, {np.percentile(results, 95):+.1f}]")
    
    return results

# Example usage
if __name__ == "__main__":
    # Example validation workflow
    print("📈 Model Validation Framework")
    print("\nTo use this validation:")
    print("1. Collect historical predictions in format:")
    print("   ['date', 'home_team', 'away_team', 'prob_home', 'pick']")
    print("2. Collect actual game results in format:")
    print("   ['date', 'home_team', 'away_team', 'home_score', 'away_score']")
    print("3. Run validate_predictions(predictions_df, results_df)")
    
    # Run example simulation
    run_monte_carlo_simulation(0.565)  # Assume 56.5% accuracy