#!/usr/bin/env python3
"""
Add 12 new football leagues to football.db.gz for the Rei-Rei ML predictor.
Usage: python3 scripts/add_new_leagues.py <batch>
  batch 1 -> leagues 103, 104, 113, 114
  batch 2 -> leagues 169, 240, 244, 328
  batch 3 -> leagues 329, 479, 564, 596
"""

import gzip
import shutil
import sqlite3
import sys
import time
from pathlib import Path

import requests

API_KEY = 'd440bc0c72f6ef65a024d6bb5483e965'
BASE    = 'https://v3.football.api-sports.io'
SLEEP   = 1.3

DB_GZ   = Path(__file__).parent.parent / 'football.db.gz'
DB_PATH = Path(__file__).parent.parent / 'football.db'

ALL_LEAGUES = {
    103: 'Norway - Eliteserien',
    104: 'Norway - OBOS-ligaen',
    113: 'Sweden - Allsvenskan',
    114: 'Sweden - Superettan',
    169: 'China - Super League',
    240: 'Colombia - Primera B',
    244: 'Finland - Veikkausliiga',
    328: 'Estonia - Esiliiga',
    329: 'Estonia - Meistriliiga',
    479: 'Canada - Premier League',
    564: 'Sweden - Division 1 Sodra',
    596: 'Sweden - Division 2 Vastra Gotaland',
}

BATCHES = {
    '1': [103, 104, 113, 114],
    '2': [169, 240, 244, 328],
    '3': [329, 479, 564, 596],
}

SEASONS = list(range(2010, 2027))

HEADERS = {'x-apisports-key': API_KEY, 'Accept': 'application/json'}


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def api_get(endpoint, params):
    url = f'{BASE}{endpoint}'
    resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
    resp.raise_for_status()
    time.sleep(SLEEP)
    return resp.json()


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def decompress_db():
    if not DB_PATH.exists():
        print(f'Decompressing {DB_GZ} ...')
        with gzip.open(DB_GZ, 'rb') as f_in, open(DB_PATH, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def compress_db():
    print(f'Compressing {DB_PATH} -> {DB_GZ}')
    with open(DB_PATH, 'rb') as f_in, gzip.open(DB_GZ, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)


def ensure_league(conn, league_id, league_name):
    cur = conn.execute('SELECT id FROM leagues WHERE id=?', (league_id,))
    if cur.fetchone() is None:
        conn.execute('INSERT OR IGNORE INTO leagues (id, name) VALUES (?,?)',
                     (league_id, league_name))
        conn.commit()
        print(f'  Inserted league {league_id}: {league_name}')


# ---------------------------------------------------------------------------
# Goal helpers
# ---------------------------------------------------------------------------

def extract_goals(events):
    goals = []
    for ev in events:
        if ev.get('type') != 'Goal':
            continue
        minute  = ev.get('time', {}).get('elapsed', 0) or 0
        extra   = ev.get('time', {}).get('extra') or 0
        team_id = (ev.get('team') or {}).get('id')
        player  = ((ev.get('player') or {}).get('name') or '').strip()
        goals.append((minute, extra, team_id, player))
    goals.sort(key=lambda x: x[0] + x[1])
    return goals


def get_first_goal(goals, home_tid):
    if not goals:
        return None, None
    minute, extra, team_id, _ = goals[0]
    total_min = minute + extra
    side = 'home' if team_id == home_tid else 'away'
    return total_min, side


def build_goals_html(goals, home_tid):
    parts = []
    for minute, extra, team_id, player in goals:
        minute_str = f'{minute}+{extra}' if extra else str(minute)
        if team_id != home_tid:
            parts.append(f'<span class="away-goal">{minute_str}\' </span>')
        else:
            parts.append(f"{minute_str}' {player}")
    return ' '.join(parts)


def build_goals_text(goals, home_tid):
    home_parts, away_parts = [], []
    for minute, extra, team_id, player in goals:
        minute_str = f'{minute}+{extra}' if extra else str(minute)
        label = f"{minute_str}' {player}".strip()
        (home_parts if team_id == home_tid else away_parts).append(label)
    return ' | '.join(home_parts + away_parts)


# ---------------------------------------------------------------------------
# Fixture processing
# ---------------------------------------------------------------------------

def process_fixture(conn, fixture, league_db_id, season):
    fix    = fixture.get('fixture', {})
    fix_id = fix.get('id')
    if fix_id is None:
        return False
    if conn.execute('SELECT id FROM matches WHERE fixture_id=?', (fix_id,)).fetchone():
        return False  # already present

    teams     = fixture.get('teams', {})
    home_info = teams.get('home', {})
    away_info = teams.get('away', {})
    home_tid  = home_info.get('id')
    home_team = home_info.get('name', '')
    away_team = away_info.get('name', '')

    goals_data  = fixture.get('goals', {})
    ft_home     = goals_data.get('home')
    ft_away     = goals_data.get('away')
    if ft_home is None or ft_away is None:
        return False

    total_goals = ft_home + ft_away

    # Optimization: skip events API call if no goals at all
    if total_goals == 0:
        return False

    score   = fixture.get('score', {})
    ht      = score.get('halftime', {})
    ht_home = ht.get('home') or 0
    ht_away = ht.get('away') or 0

    st_home  = ft_home - ht_home
    st_away  = ft_away - ht_away
    ht_goals = ht_home + ht_away
    st_goals = st_home + st_away
    btts     = 1 if ft_home > 0 and ft_away > 0 else 0
    result   = 'H' if ft_home > ft_away else ('A' if ft_away > ft_home else 'D')

    date_full = fix.get('date', '')
    date_str  = date_full[:10]
    sort_date = date_str.replace('-', '')
    time_str  = date_full[11:16]

    # Fetch goal events
    evdata = api_get('/fixtures/events', {'fixture': fix_id})
    events = evdata.get('response', [])
    goals  = extract_goals(events)

    first_min, first_side = get_first_goal(goals, home_tid)
    if first_min is None or first_min > 16:
        return False

    goals_html = build_goals_html(goals, home_tid)
    goals_text = build_goals_text(goals, home_tid)

    conn.execute(
        '''INSERT INTO matches (
            fixture_id, league_id, season, date_str, sort_date, time_str,
            home_team, away_team, ht_home, ht_away, st_home, st_away,
            ft_home, ft_away, total_goals, ht_goals, st_goals,
            btts, result, goals_html, goals_text,
            first_goal_min, first_goal_team, fg_result, is_archived
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)''',
        (fix_id, league_db_id, season, date_str, sort_date, time_str,
         home_team, away_team, ht_home, ht_away, st_home, st_away,
         ft_home, ft_away, total_goals, ht_goals, st_goals,
         btts, result, goals_html, goals_text,
         first_min, first_side, first_side)
    )
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    batch_key = sys.argv[1] if len(sys.argv) > 1 else '1'
    if batch_key not in BATCHES:
        print(f'Invalid batch "{batch_key}". Use 1, 2, or 3.')
        sys.exit(1)

    league_ids = BATCHES[batch_key]
    NEW_LEAGUES = {k: v for k, v in ALL_LEAGUES.items() if k in league_ids}
    print(f'Batch {batch_key}: {list(NEW_LEAGUES.values())}')

    decompress_db()
    conn = sqlite3.connect(DB_PATH)

    try:
        conn.execute('ALTER TABLE matches ADD COLUMN fixture_id INTEGER')
        conn.commit()
        print('Added fixture_id column to matches table')
    except Exception:
        pass  # column already exists

    total_inserted = 0

    try:
        for league_id, league_name in NEW_LEAGUES.items():
            print(f'\n=== {league_name} (API id={league_id}) ===')
            ensure_league(conn, league_id, league_name)

            for season in SEASONS:
                print(f'  {season}...', end=' ', flush=True)
                data = api_get('/fixtures', {
                    'league':  league_id,
                    'season':  season,
                    'status':  'FT',
                })
                fixtures = data.get('response', [])
                if not fixtures:
                    print('none')
                    continue
                print(f'{len(fixtures)} fixtures', end=' ')

                inserted = 0
                for fix in fixtures:
                    try:
                        if process_fixture(conn, fix, league_id, season):
                            inserted += 1
                            conn.commit()
                    except Exception as e:
                        fid = fix.get('fixture', {}).get('id', '?')
                        print(f'\n    [ERR fixture {fid}] {e}')

                total_inserted += inserted
                print(f'-> {inserted} early-goal matches')

    finally:
        conn.close()

    print(f'\nTotal inserted: {total_inserted}')
    compress_db()
    print('Done.')


if __name__ == '__main__':
    main()
