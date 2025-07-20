#!/usr/bin/env python3
"""
mlb_simulator.py
Refactored to give every team a unique Elo and print picks just once.
"""

import pandas as pd
import math

# ─── CONFIG ────────────────────────────────────────────────────────────────────

TEAM_INPUTS = "Team_Inputs(1).csv"
HISTORICAL  = "Historical_data.csv"
ODDS        = "mlb_odds_2025-07-20.csv"
ELOS        = "inseason_elos.csv"

# ─── HELPERS ───────────────────────────────────────────────────────────────────

def poisson_pmf(k, lam):
    return lam**k * math.exp(-lam) / math.factorial(k)

def win_prob_poisson(lam_h, lam_a, max_runs=15):
    pmf_h = [poisson_pmf(i, lam_h) for i in range(max_runs+1)]
    pmf_a = [poisson_pmf(i, lam_a) for i in range(max_runs+1)]
    return sum(pmf_h[i] * sum(pmf_a[:i]) for i in range(max_runs+1))

def win_prob_elo(elo_h, elo_a):
    # standard Elo logistic curve
    return 1.0 / (1.0 + 10 ** ((elo_a - elo_h) / 400.0))

# ─── LOAD & CALIBRATE HFA ─────────────────────────────────────────────────────

def load_and_calibrate_hfa():
    df = pd.read_csv(HISTORICAL, parse_dates=['date'])
    df = df[df['gametype']=='regular']
    hr = df['hruns'].mean()
    vr = df['vruns'].mean()
    league_rpg = (hr + vr) / 2.0
    hfa_factor  = hr / vr
    return league_rpg, hfa_factor

# ─── LOAD TEAM OFFENSE/DEFENSE FACTORS ────────────────────────────────────────

def load_team_metrics(league_rpg):
    tm = pd.read_csv(TEAM_INPUTS).rename(columns={'Team':'team'})
    tm['rpg'] = tm['R'] / tm['G']
    tm['off_factor'] = tm['rpg'] / league_rpg
    tm['ERA'] = tm['ERA'].astype(float)
    league_era = tm['ERA'].mean()
    tm['def_factor'] = league_era / tm['ERA']
    return tm.set_index('team')[['off_factor','def_factor']].to_dict('index')

# ─── LOAD LATEST ELO RATINGS ──────────────────────────────────────────────────

def load_elos():
    e = pd.read_csv(ELOS, parse_dates=['date'])
    latest = e['date'].max()
    e = e[e['date']==latest].copy()

    # if your CSV uses odd codes, normalize them here:
    code_map = {'CHN':'CHC','LAN':'LAD','SLN':'STL','WAS':'WSN'}
    e['team'] = e['team'].replace(code_map)

    # average in case you have duplicates
    elo_by_team = e.groupby('team')['elo'].mean().to_dict()
    avg_elo = sum(elo_by_team.values()) / len(elo_by_team)
    return elo_by_team, avg_elo

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    # 1) Calibration
    league_rpg, hfa = load_and_calibrate_hfa()

    # 2) Offense/defense factors
    tm_factors = load_team_metrics(league_rpg)

    # 3) Elo
    elo_dict, avg_elo = load_elos()

    # 4) Today's matchups & code mapping
    odds = pd.read_csv(ODDS, parse_dates=['date'])
    name_to_code = {
        'Arizona Diamondbacks':'ARI', 'Atlanta Braves':'ATL',
        'Baltimore Orioles':'BAL',       'Boston Red Sox':'BOS',
        'Chicago Cubs':'CHC',            'Chicago White Sox':'CWS',
        'Cincinnati Reds':'CIN',         'Cleveland Guardians':'CLE',
        'Colorado Rockies':'COL',        'Detroit Tigers':'DET',
        'Houston Astros':'HOU',          'Kansas City Royals':'KC',
        'Los Angeles Angels':'LAA',      'Los Angeles Dodgers':'LAD',
        'Miami Marlins':'MIA',           'Milwaukee Brewers':'MIL',
        'Minnesota Twins':'MIN',         'New York Mets':'NYM',
        'New York Yankees':'NYY',        'Oakland Athletics':'OAK',
        'Philadelphia Phillies':'PHI',   'Pittsburgh Pirates':'PIT',
        'San Diego Padres':'SDP',        'San Francisco Giants':'SFG',
        'Seattle Mariners':'SEA',        'St. Louis Cardinals':'STL',
        'Tampa Bay Rays':'TB',           'Texas Rangers':'TEX',
        'Toronto Blue Jays':'TOR',       'Washington Nationals':'WSN'
    }
    odds['hc'] = odds['home_team'].map(name_to_code)
    odds['ac'] = odds['away_team'].map(name_to_code)

    # 5) Warn about any missing Elo codes
    missing_elos = set(odds['hc'].unique()) | set(odds['ac'].unique()) - set(elo_dict.keys())
    if missing_elos:
        print(f"⚠️  Missing Elo for: {sorted(missing_elos)} – using league average ({avg_elo:.1f})")

    # 6) Compute picks
    results = []
    for _, r in odds.iterrows():
        off_h = tm_factors.get(r['hc'], {'off_factor':1,'def_factor':1})
        off_a = tm_factors.get(r['ac'], {'off_factor':1,'def_factor':1})
        elo_h = elo_dict.get(r['hc'], avg_elo)
        elo_a = elo_dict.get(r['ac'], avg_elo)

        lam_h = off_h['off_factor'] * off_a['def_factor'] * league_rpg * hfa
        lam_a = off_a['off_factor'] * off_h['def_factor'] * league_rpg
        p_poisson = win_prob_poisson(lam_h, lam_a)
        p_elo     = win_prob_elo(elo_h, elo_a)

        # blend Poisson & Elo (70/30)
        p_h = 0.7 * p_poisson + 0.3 * p_elo
        pick = r['home_team'] if p_h >= 0.5 else r['away_team']

        results.append({
            'home':       r['home_team'],
            'away':       r['away_team'],
            'elo_h':      round(elo_h,1),
            'elo_a':      round(elo_a,1),
            'P_poisson':  round(p_poisson,3),
            'P_elo':      round(p_elo,3),
            'P_combined': round(p_h,3),
            'pick':       pick
        })

    # 7) Print a single summary of all picks
    print("\nToday's Picks:")
    for row in results:
        print(f"  {row['home']} @ {row['away']} → Pick: {row['pick']} "
              f"(Pₚₒᵢₛ = {row['P_poisson']}, Pₑₗₒ = {row['P_elo']}, "
              f"P = {row['P_combined']})")

if __name__ == "__main__":
    main()
