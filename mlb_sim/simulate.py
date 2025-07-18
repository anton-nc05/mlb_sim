### simulate.py

import argparse
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from .constants import *
from .data_utils import compute_preseason_elos, compute_historical_elos
from .elo import blend_elos, win_probability_from_elo_diff, update_elo
from .api_calls import get_team_mappings, get_daily_schedule, get_daily_odds, american_odds_to_prob


def main():
    parser = argparse.ArgumentParser(description="MLB Betting Simulator")
    parser.add_argument('--start', required=True, help='YYYY-MM-DD')
    parser.add_argument('--end', required=True, help='YYYY-MM-DD')
    parser.add_argument('--k_metrics', type=int, default=K_METRICS)
    parser.add_argument('--alpha', type=float, default=ALPHA)
    parser.add_argument('--k_factor', type=float, default=K_FACTOR)
    parser.add_argument('--edge_threshold', type=float, default=EDGE_THRESHOLD)
    parser.add_argument('--initial_bankroll', type=float, default=INITIAL_BANKROLL)
    parser.add_argument('--bet_amount', type=float, default=BET_AMOUNT)
    args = parser.parse_args()

    # Compute and blend Elos
    pre_df = compute_preseason_elos(args.k_metrics, METRIC_COLUMNS)
    hist_df = compute_historical_elos(args.k_factor)
    pre_map = dict(zip(pre_df['Team'], pre_df['preseason_elo']))
    hist_map = dict(zip(hist_df['team'], hist_df['historical_elo']))

    code_map = {
        'CHC': 'CHN', 'CHW': 'CHA', 'KCR': 'KCA', 'LAA': 'ANA', 'LAD': 'LAN',
        'NYM': 'NYN', 'NYY': 'NYA', 'SDP': 'SDN', 'SFG': 'SFN', 'STL': 'SLN',
        'TBR': 'TBA', 'WSH': 'WAS', 'WSN': 'WAS'
    }
    # Remap preseason teams to historical keys
    pre_map = {code_map.get(team, team): elo for team, elo in pre_map.items()}

    blended = blend_elos(pre_map, hist_map, args.alpha)
    # Write previous Elos without indentation error
    pd.DataFrame({'team': list(blended.keys()), 'previous_elo': list(blended.values())}).to_csv(PREVIOUS_ELO_PATH, index=False)

    # Setup simulation
    team_map = get_team_mappings()
    current_elos = blended.copy()
    bankroll = args.initial_bankroll
    bets = {}
    summary = []
    cum_bets = cum_wins = 0
    cum_brier = 0.0

    start_date = datetime.fromisoformat(args.start)
    end_date = datetime.fromisoformat(args.end)
    date = start_date

    while date <= end_date:
        ds = date.strftime('%Y-%m-%d')
        prev = (date - timedelta(days=1)).strftime('%Y-%m-%d')

        # Process previous day's results
        for game in get_daily_schedule(prev):
            status = game.get('status', {}).get('detailedState')
            if status == 'Final':
                gid = game['gamePk']
                home_id = game['teams']['home']['team']['id']
                away_id = game['teams']['away']['team']['id']
                home = code_map.get(team_map.get(home_id, ''), team_map.get(home_id, ''))
                away = code_map.get(team_map.get(away_id, ''), team_map.get(away_id, ''))
                runs_h = game['teams']['home']['score']
                runs_a = game['teams']['away']['score']
                out_h = int(runs_h > runs_a)
                # Update Elos
                ra, rb = current_elos[home], current_elos[away]
                na, nb = update_elo(ra, rb, out_h, args.k_factor)
                current_elos[home], current_elos[away] = na, nb
                # Record in-season Elos
                pd.DataFrame([
                    {'date': prev, 'team': home, 'elo': na},
                    {'date': prev, 'team': away, 'elo': nb}
                ]).to_csv(
                    INSEASON_ELO_PATH,
                    mode='a',
                    header=not Path(INSEASON_ELO_PATH).exists(),
                    index=False
                )
                # Settle bets
                if gid in bets:
                    for bet in bets[gid]:
                        outcome = out_h if bet['team'] == home else (1 - out_h)
                        american = bet['american']
                        if outcome == 1:
                            profit = args.bet_amount * (american / 100 if american > 0 else 100 / abs(american))
                        else:
                            profit = -args.bet_amount
                        bankroll += profit
                        cum_bets += 1
                        cum_wins += outcome
                        cum_brier += (bet['prob'] - outcome) ** 2
                    del bets[gid]

        # Place bets for current day
        odds_list = get_daily_odds(ds)
        odds_map = {e['id']: e for e in odds_list}
        for game in get_daily_schedule(ds):
            status = game.get('status', {}).get('detailedState')
            if status in ['Scheduled', 'Pre-Game', 'Preview']:
                gid = game['gamePk']
                home_id = game['teams']['home']['team']['id']
                away_id = game['teams']['away']['team']['id']
                home = code_map.get(team_map.get(home_id, ''), team_map.get(home_id, ''))
                away = code_map.get(team_map.get(away_id, ''), team_map.get(away_id, ''))
                diff = current_elos[home] - current_elos[away]
                p_home = win_probability_from_elo_diff(diff)
                ev = odds_map.get(gid)
                if not ev:
                    continue
                # Extract moneyline odds
                outcomes = ev['markets'][0]['outcomes']
                h_odd = next((o['point'] for o in outcomes if o['name'] == home), None)
                a_odd = next((o['point'] for o in outcomes if o['name'] == away), None)
                if h_odd is None or a_odd is None:
                    continue
                prob_home = american_odds_to_prob(h_odd)
                prob_away = american_odds_to_prob(a_odd)
                # Decide bet
                if (p_home - prob_home) >= args.edge_threshold:
                    bets.setdefault(gid, []).append({'team': home, 'prob': p_home, 'american': h_odd})
                elif ((1 - p_home) - prob_away) >= args.edge_threshold:
                    bets.setdefault(gid, []).append({'team': away, 'prob': 1 - p_home, 'american': a_odd})

        # Daily summary metrics
        roi = (bankroll - args.initial_bankroll) / args.initial_bankroll
        hit_rate = cum_wins / cum_bets if cum_bets else 0.0
        brier_score = cum_brier / cum_bets if cum_bets else 0.0
        summary.append({'date': ds, 'bankroll': bankroll, 'ROI': roi, 'hit_rate': hit_rate, 'brier_score': brier_score})
        pd.DataFrame(summary).to_csv(DAILY_SUMMARY_PATH, index=False)

        date += timedelta(days=1)


if __name__ == "__main__":
    main()
