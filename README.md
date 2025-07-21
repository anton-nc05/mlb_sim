A Python-based MLB betting analysis and prediction system using Elo ratings and statistical modeling.

Overview
This project is a comprehensive baseball analytics tool using 2 different apis and over 40 different metrics that combines Elo rating systems with sportsboook odds analysis to generate MLB game predictions and betting insights. The system processes historical data, calculates team ratings, and provides daily picks based on statistical modeling.

Features
 - Elo Rating System: Dynamic team strength calculations based on historical performance
 - Odds Integration: Real-time betting odds analysis and comparison
 - Game Simulation: Monte Carlo simulation for game outcome predictions
 - Model Validation: Backtesting and performance metrics for prediction accuracy
 - Daily Picks: Automated generation of recommended bets for current games

Files Structure
 - Core Scripts
    - mlb_simulator.py - Main simulation engine for game predictions
    - elo_data_fixer.py - Data preprocessing and Elo rating calculations
    - elo_debug.py - Debugging tools for Elo model validation
    - model_validation.py - Performance testing and accuracy metrics

API Integration
mlb_api_test.py - MLB API connection and data retrieval
odds_api_test.py - Sports betting odds API integration

Data Files
Historical_data.csv - Historical MLB game results and statistics
Team_Inputs(1).csv - Team-specific input parameters and settings
inseason_elos.csv - Current season Elo ratings
inseason_elos_fixed.csv - Processed and validated Elo ratings
mlb_odds_2025-07-21.csv - Daily betting odds data

