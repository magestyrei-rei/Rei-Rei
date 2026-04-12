#!/usr/bin/env python3
"""
migrate.py  —  Converte i 65 file HTML in un database SQLite
Esegui una volta sola: python migrate.py
Il file football.db generato va committato su GitHub insieme all'app.
"""

import sqlite3, re, sys
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent   # cartella Downloads
DB_PATH  = Path(__file__).parent / 'football.db'
HTML_RE  = re.compile(r'^(\d+)_gol-16min_2010-2024\.html$')

# ── Helpers ──────────────────────────────────────────────────────────────────

def clean(s):
    return re.sub(r'<[^>]+>', '', s or '').strip()

def parse_score(s):
    if s and '-' in s:
        p = s.split('-', 1)
        try:
            return int(p[0].strip()), int(p[1].strip())
        except ValueError:
            pass
    return 0, 0

def parse_sort_date(date_str):
    """dd/mm/yyyy → yyyymmdd per ordinamento"""
    p = date_str.split('/')
    if len(p) == 3:
        return f"{p[2].zfill(4)}{p[1].zfill(2)}{p[0].zfill(2)}"
    return '00000000'

def parse_first_goal(goals_html, goals_text):
    """Ritorna (team='home'|'away'|None, minute=int|None)"""
    if not goals_html and not goals_text:
        return None, None
    # Primo elemento della lista gol
    first_html = re.split(r',', goals_html.strip())[0].strip() if goals_html else ''
    first_text = (goals_text.split(',')[0].strip() if goals_text else '')
    # Team
    team = 'away' if 'away-goal' in first_html else ('home' if first_html or first_text else None)
    # Minuto
    min_str = re.sub(r'<[^>]+>', '', first_text).strip().rstrip("'")
    min_str = re.sub(r'\+.*', '', min_str).strip()
    try:
        minute = int(min_str)
    except (ValueError, TypeError):
        minute = None
    return team, minute

def to_float(s):
    s2 = clean(s) if s else ''
    try:
        return float(s2)
    except (ValueError, TypeError):
        return None

# ── Parser file HTML ─────────────────────────────────────────────────────────

def parse_html_file(filepath, league_id):
    try:
        content = filepath.read_text(encoding='utf-8')
    except Exception as e:
        print(f"  ERRORE lettura {filepath.name}: {e}")
        return None, []

    # Nome lega
    h1 = re.search(r'<h1>(.*?)</h1>', content)
    if h1:
        name = re.sub(r'\s*[-–]\s*Primo gol.*', '', h1.group(1), flags=re.IGNORECASE).strip()
    else:
        t = re.search(r'<title>(.*?)</title>', content)
        raw = t.group(1) if t else f'League {league_id}'
        name = re.sub(r'\s*[-–]\s*Primo gol.*', '', raw, flags=re.IGNORECASE).strip()

    # Righe
    body = re.sub(r'<thead>.*?</thead>', '', content, flags=re.DOTALL)
    rows = re.findall(r'<tr>\s*(.*?)\s*</tr>', body, re.DOTALL)

    matches = []
    for row in rows:
        cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)
        if len(cells) < 10:
            continue

        ht_str = clean(cells[6])
        st_str = clean(cells[7])
        ft_str = clean(cells[8])
        g_html = cells[9].strip()
        g_text = clean(cells[9])

        ht_h, ht_a = parse_score(ht_str)
        st_h, st_a = parse_score(st_str)
        ft_h, ft_a = parse_score(ft_str)

        total_goals = ft_h + ft_a
        ht_goals    = ht_h + ht_a
        st_goals    = st_h + st_a
        btts        = 1 if (ft_h > 0 and ft_a > 0) else 0

        if ft_h > ft_a:   result = '1'
        elif ft_h < ft_a: result = '2'
        else:             result = 'X'

        fg_team, fg_min = parse_first_goal(g_html, g_text)

        if fg_team == 'home':
            fg_result = ('win' if ft_h > ft_a else ('loss' if ft_h < ft_a else 'draw'))
        elif fg_team == 'away':
            fg_result = ('win' if ft_a > ft_h else ('loss' if ft_a < ft_h else 'draw'))
        else:
            fg_result = None

        date_str = clean(cells[1])

        matches.append({
            'league_id':       league_id,
            'season':          clean(cells[0]),
            'date_str':        date_str,
            'sort_date':       parse_sort_date(date_str),
            'time_str':        clean(cells[2]),
            'home_team':       clean(cells[3]),
            'away_team':       clean(cells[4]),
            'ht_home':  ht_h,  'ht_away':  ht_a,
            'st_home':  st_h,  'st_away':  st_a,
            'ft_home':  ft_h,  'ft_away':  ft_a,
            'total_goals':     total_goals,
            'ht_goals':        ht_goals,
            'st_goals':        st_goals,
            'btts':            btts,
            'result':          result,
            'goals_html':      g_html,
            'goals_text':      g_text,
            'q1':  to_float(cells[10] if len(cells) > 10 else None),
            'qx':  to_float(cells[11] if len(cells) > 11 else None),
            'q2':  to_float(cells[12] if len(cells) > 12 else None),
            'first_goal_min':  fg_min,
            'first_goal_team': fg_team,
            'fg_result':       fg_result,
            'is_archived':     0,
        })

    return name, matches

# ── Schema SQLite ────────────────────────────────────────────────────────────

SCHEMA = """
DROP TABLE IF EXISTS leagues;
DROP TABLE IF EXISTS matches;

CREATE TABLE leagues (
    id        INTEGER PRIMARY KEY,
    name      TEXT NOT NULL,
    file_name TEXT
);

CREATE TABLE matches (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    league_id     INTEGER NOT NULL,
    season        TEXT,
    date_str      TEXT,
    sort_date     TEXT,
    time_str      TEXT,
    home_team     TEXT,
    away_team     TEXT,
    ht_home       INTEGER DEFAULT 0,
    ht_away       INTEGER DEFAULT 0,
    st_home       INTEGER DEFAULT 0,
    st_away       INTEGER DEFAULT 0,
    ft_home       INTEGER DEFAULT 0,
    ft_away       INTEGER DEFAULT 0,
    total_goals   INTEGER DEFAULT 0,
    ht_goals      INTEGER DEFAULT 0,
    st_goals      INTEGER DEFAULT 0,
    btts          INTEGER DEFAULT 0,
    result        TEXT,
    goals_html    TEXT,
    goals_text    TEXT,
    q1            REAL,
    qx            REAL,
    q2            REAL,
    first_goal_min  INTEGER,
    first_goal_team TEXT,
    fg_result       TEXT,
    is_archived     INTEGER DEFAULT 0,
    FOREIGN KEY (league_id) REFERENCES leagues(id)
);

CREATE INDEX idx_league_date  ON matches(league_id, sort_date);
CREATE INDEX idx_sort_date    ON matches(sort_date);
CREATE INDEX idx_league_fg    ON matches(league_id, first_goal_min);
"""

INSERT_SQL = """
INSERT INTO matches (
    league_id, season, date_str, sort_date, time_str,
    home_team, away_team,
    ht_home, ht_away, st_home, st_away, ft_home, ft_away,
    total_goals, ht_goals, st_goals, btts, result,
    goals_html, goals_text, q1, qx, q2,
    first_goal_min, first_goal_team, fg_result, is_archived
) VALUES (
    :league_id, :season, :date_str, :sort_date, :time_str,
    :home_team, :away_team,
    :ht_home, :ht_away, :st_home, :st_away, :ft_home, :ft_away,
    :total_goals, :ht_goals, :st_goals, :btts, :result,
    :goals_html, :goals_text, :q1, :qx, :q2,
    :first_goal_min, :first_goal_team, :fg_result, :is_archived
)
"""

# ── Main ─────────────────────────────────────────────────────────────────────

def run():
    print("=" * 65)
    print("  Football Stats — Migrazione HTML → SQLite")
    print("=" * 65)
    print(f"  Cartella dati : {DATA_DIR}")
    print(f"  Output DB     : {DB_PATH}")
    print()

    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)

    files = sorted(DATA_DIR.glob('*_gol-16min_2010-2024.html'))
    total_leagues = total_matches = 0

    for f in files:
        m = HTML_RE.match(f.name)
        if not m:
            continue

        lid  = int(m.group(1))
        name, matches = parse_html_file(f, lid)

        if not matches:
            print(f"  [{lid:4d}]  (vuoto o errore) {f.name}")
            continue

        conn.execute(
            "INSERT OR REPLACE INTO leagues(id, name, file_name) VALUES (?,?,?)",
            (lid, name, f.name)
        )
        conn.executemany(INSERT_SQL, matches)

        total_leagues += 1
        total_matches += len(matches)
        print(f"  [{lid:4d}]  {name:<52s}  {len(matches):5d} partite")

    conn.commit()
    conn.close()

    db_kb = DB_PATH.stat().st_size // 1024
    print()
    print("=" * 65)
    print(f"  ✓ {total_leagues} campionati, {total_matches:,} partite")
    print(f"  ✓ Database: {db_kb} KB  →  {DB_PATH.name}")
    print("=" * 65)

if __name__ == '__main__':
    run()
