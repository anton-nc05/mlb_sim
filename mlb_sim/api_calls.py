### api_calls.py

import requests
from .constants import BASE_URL, ODDS_API_KEY
from typing import Dict, Any, List


def get_with_retries(url: str, params: Dict[str, Any] = None, retries: int = 3) -> Any:
    """GET with retries."""
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception:
            if i == retries - 1:
                return None


def get_team_mappings() -> Dict[int, str]:
    """Map MLB team ID to abbreviation."""
    url = 'https://statsapi.mlb.com/api/v1/teams'
    data = get_with_retries(url, params={'sportId': 1})
    mapping = {}
    if data and 'teams' in data:
        for t in data['teams']:
            mapping[t['id']] = t.get('abbreviation', t['name'])
    return mapping


def get_daily_schedule(date: str) -> List[Dict]:
    """Get MLB schedule for a date."""
    url = 'https://statsapi.mlb.com/api/v1/schedule'
    data = get_with_retries(url, params={'sportId': 1, 'date': date})
    games = []
    if data and 'dates' in data and data['dates']:
        games = data['dates'][0]['games']
    return games


def american_odds_to_prob(odds: int) -> float:
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100 / (odds + 100)
    return -odds / (-odds + 100)


def get_daily_odds(date: str) -> List[Dict]:
    """Get consensus odds for MLB games on a date."""
    url = f"{BASE_URL}/sports/baseball_mlb/odds"
    data = get_with_retries(url, params={'regions': 'us', 'markets': 'h2h,spreads,totals', 'dateFormat': 'iso', 'apiKey': ODDS_API_KEY})
    events = []
    if data:
        for e in data:
            if e.get('commence_time', '').startswith(date):
                events.append(e)
    return events
