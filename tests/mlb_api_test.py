# File: mlb_api_test.py
"""
Test environment for MLB-StatsAPI.

Fetches today’s MLB schedule and prints each matchup
with the final box score in a clean, side-by-side format.
If no box score is available (game not yet active), it prints "game not active".

Usage:
    pip install MLB-StatsAPI
    python mlb_api_test.py
"""

import statsapi
import datetime

def main():
    # Today's date for the schedule
    today_str = datetime.date.today().strftime("%m/%d/%Y")
    print(f"MLB Games for {today_str}:\n")

    # Fetch schedule for today
    try:
        games = statsapi.schedule(start_date=today_str, end_date=today_str)
    except Exception as e:
        print(f"Error fetching schedule: {e}")
        return

    if not games:
        print("No games scheduled for today.")
        return

    # For each game, fetch and display box score or "game not active"
    for game in games:
        game_pk = game.get("game_pk") or game.get("game_id")
        away    = game.get("away_name") or game.get("away_team", "Unknown Away")
        home    = game.get("home_name") or game.get("home_team", "Unknown Home")

        try:
            box = statsapi.boxscore_data(game_pk)
            away_score = box["teams"]["away"]["teamStats"]["batting"]["runs"]
            home_score = box["teams"]["home"]["teamStats"]["batting"]["runs"]
            print(f"{away:<20} ({away_score}) at {home:<20} ({home_score})")
        except KeyError:
            # No 'teams' key means the game isn't active yet
            print(f"{away:<20} at {home:<20} — game not active")
        except Exception as e:
            # Any other errors
            print(f"{away:<20} at {home:<20} — error: {e}")

if __name__ == "__main__":
    main()
