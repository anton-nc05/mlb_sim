#!/usr/bin/env python3
"""
mlb_simulator.py

Put this file alongside:
  Team_Inputs(1).csv
  Historical_data.csv
  mlb_odds_2025-07-18.csv
  inseason_elos.csv

Then in VSCode/Code Runner just hit Run ▶️
"""

import pandas as pd
import math

# ─── CONFIG: filenames (all in same folder as this script) ─────────────────

TEAM_INPUTS = "Team_Inputs(1).csv"
HISTORICAL  = "Historical_data.csv"
ODDS        = "mlb_odds_2025-07-18.csv"
ELOS        = "inseason_elos.csv"

# ─── HELPERS ────────────────────────────────────────────────────────────────

def poisson_pmf(k, lam):
    return lam**k * math.exp(-lam) / math.factorial(k)

def win_prob_poisson(lam_h, lam_a, max_runs=15):
    pmf_h = [poisson_pmf(i, lam_h) for i in range(max_runs+1)]
    pmf_a = [poisson_pmf(i, lam_a) for i in range(max_runs+1)]
    p = 0.0
    for i in range(max_runs+1):
        p += pmf_h[i] * sum(pmf_a[:i])
    return p

# ─── LOAD & CALIBRATE HOME-FIELD & LEAGUE RUNS ──────────────────────────────

def load_and_calibrate_hfa():
    df = pd.read_csv(HISTORICAL, parse_dates=['date'])
    df = df[df['gametype']=='regular']
    hr = df['hruns'].mean()
    vr = df['vruns'].mean()
    league_rpg = (hr + vr) / 2
    hfa_factor  = hr / vr
    return league_rpg, hfa_factor

# ─── LOAD TEAM OFFENSE/DEFENSE FACTORS ─────────────────────────────────────

def load_team_metrics(league_rpg):
    tm = pd.read_csv(TEAM_INPUTS).rename(columns={'Team':'team'})
    tm['rpg'] = tm['R'] / tm['G']
    tm['off_factor'] = tm['rpg'] / league_rpg
    tm['ERA'] = tm['ERA'].astype(float)
    league_era = tm['ERA'].mean()
    tm['def_factor'] = league_era / tm['ERA']
    return tm[['team','off_factor','def_factor']]

# ─── LOAD LATEST ELO RATINGS ─────────────────────────────────────────────────

def load_elos():
    e = pd.read_csv(ELOS, parse_dates=['date'])
    latest = e['date'].max()
    e = e[e['date']==latest]
    # map any odd codes → your three-letter
    code_map = {'CHN':'CHC','LAN':'LAD','SLN':'STL','WAS':'WSN'}
    e['team'] = e['team'].map(lambda x: code_map.get(x,x))
    avg_elo = e['elo'].mean()
    ed = e.drop_duplicates('team').set_index('team')['elo'].to_dict()
    return ed, avg_elo

# ─── MAIN ───────────────────────────────────────────────────────────────────

def main():
    # 1) Calibrate
    league_rpg, hfa = load_and_calibrate_hfa()

    # 2) Team factors
    tm_dict = load_team_metrics(league_rpg).set_index('team').to_dict('index')

    # 3) Elo
    elo_dict, avg_elo = load_elos()

    # 4) Today's matchups
    odds = pd.read_csv(ODDS, parse_dates=['date'])
    name_to_code = {
      'Arizona Diamondbacks':'ARI','Atlanta Braves':'ATL','Baltimore Orioles':'BAL',
      # … (same mapping as before) …
      'Washington Nationals':'WSN'
    }
    odds['hc'] = odds['home_team'].map(name_to_code)
    odds['ac'] = odds['away_team'].map(name_to_code)

    # 5) Compute picks
    results = []
    for _, r in odds.iterrows():
        # get factors (default to 1.0 if missing)
        h = tm_dict.get(r['hc'], {'off_factor':1,'def_factor':1})
        a = tm_dict.get(r['ac'], {'off_factor':1,'def_factor':1})
        elo_h = elo_dict.get(r['hc'], avg_elo)
        elo_a = elo_dict.get(r['ac'], avg_elo)

        lam_h = h['off_factor'] * a['def_factor'] * league_rpg * hfa
        lam_a = a['off_factor'] * h['def_factor'] * league_rpg
        p_h   = win_prob_poisson(lam_h, lam_a)
        pick  = r['home_team'] if p_h >= 0.5 else r['away_team']

        results.append({
            'date':       r['date'].date(),
            'home':       r['home_team'],
            'away':       r['away_team'],
            'elo_h':      round(elo_h,1),
            'elo_a':      round(elo_a,1),
            'λ_home':     round(lam_h,2),
            'λ_away':     round(lam_a,2),
            'P_home_win': round(p_h,3),
            'pick':       pick
        })

    df = pd.DataFrame(results)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
