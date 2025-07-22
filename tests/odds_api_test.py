# File: odds_api_test.py
"""
Fetches today’s MLB moneyline and totals odds, computes implied probabilities,
and writes the results to a CSV file.

Usage:
    pip install requests
    ODDS_API_KEY=your_real_key_here python odds_api_test.py
"""

import os
import datetime
import requests
import csv

def get_api_key():
    key = os.getenv("ODDS_API_KEY", "68256bffa0b9127296003dfddb6c8fca")
    if not key or key.startswith("…") or len(key) < 20:
        raise RuntimeError("Please set a valid ODDS_API_KEY environment variable.")
    return key

def american_implied_prob(odds: int) -> float:
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return -odds / (-odds + 100)

def fetch_json(session: requests.Session, url: str, **kwargs):
    resp = session.get(url, **kwargs)
    resp.raise_for_status()
    return resp.json()

def main():
    API_KEY = get_api_key()
    os.environ.pop("HTTP_PROXY", None)
    os.environ.pop("HTTPS_PROXY", None)

    BASE_URL = "https://api.the-odds-api.com/v4"
    session = requests.Session()
    session.trust_env = False

    today = datetime.date.today()
    print(f"Writing MLB odds for {today} to CSV...")

    # 1) Fetch sports list and find MLB
    sports = fetch_json(
        session,
        f"{BASE_URL}/sports/",
        params={"apiKey": API_KEY},
        timeout=10
    )
    mlb = next((s for s in sports if s.get("key") == "baseball_mlb"), None)
    if not mlb:
        print("MLB sport key not found.")
        return

    # 2) Fetch both moneyline (h2h) and totals markets
    events = fetch_json(
        session,
        f"{BASE_URL}/sports/{mlb['key']}/odds/",
        params={
            "apiKey": API_KEY,
            "regions": "us",
            "markets": "h2h,totals",
            "oddsFormat": "american"
        },
        timeout=10
    )
    if not events:
        print("No MLB odds available right now.")
        return

    # 3) Open CSV for writing
    output_file = f"mlb_odds_{today}.csv"
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "date",
            "away_team",
            "home_team",
            "away_ml",
            "home_ml",
            "away_implied_prob",
            "home_implied_prob",
            "total_line",
            "over_odds",
            "under_odds"
        ])

        # 4) Loop through events, filter by today, and write rows
        for event in events:
            ct = event.get("commence_time")
            if not ct:
                continue

            try:
                dt_utc = datetime.datetime.fromisoformat(ct.replace("Z", "+00:00"))
                dt_local = dt_utc.astimezone()
            except Exception:
                continue

            if dt_local.date() != today:
                continue

            away = event.get("away_team", "Away")
            home = event.get("home_team", "Home")
            bk = event.get("bookmakers", [])
            if not bk:
                continue

            markets = {m["key"]: m for m in bk[0].get("markets", [])}
            h2h = markets.get("h2h")
            totals = markets.get("totals")
            if not h2h or not totals:
                continue

            # Moneyline odds
            outcomes_h2h = h2h.get("outcomes", [])
            away_ml = next((o["price"] for o in outcomes_h2h if o["name"] == away), None)
            home_ml = next((o["price"] for o in outcomes_h2h if o["name"] == home), None)
            if away_ml is None or home_ml is None:
                continue
 
            # Implied probabilities
            away_imp = american_implied_prob(away_ml)
            home_imp = american_implied_prob(home_ml)

            # Totals (over/under)
            outcomes_totals = totals.get("outcomes", [])
            over = next((o for o in outcomes_totals if o["name"].startswith("Over")), None)
            under = next((o for o in outcomes_totals if o["name"].startswith("Under")), None)
            if not over or not under:
                continue

            # Parse the total line (e.g. "Over 8.5")
            try:
                total_line = float(over["name"].split(" ")[1])
            except Exception:
                total_line = None

            over_odds = over["price"]
            under_odds = under["price"]

            writer.writerow([
                today,
                away,
                home,
                away_ml,
                home_ml,
                round(away_imp, 4),
                round(home_imp, 4),
                total_line,
                over_odds,
                under_odds
            ])

    print(f"Wrote data for {today} to {output_file}")

if __name__ == "__main__":
    main()
