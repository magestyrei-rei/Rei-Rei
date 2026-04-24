# odds_logger.py - Logger delle quote live: snapshot periodici per costruire dataset storico
# Complementare a ml_pick.py. Supporta backend multipli:
#   - 'memory': dict in RAM (volatile, per test)
#   - 'supabase': Postgres via API REST (persistente, free tier 500MB)
#   - 'turso': libSQL via API (persistente, free tier 5GB)
#
# Endpoints esposti (registrati da register()):
#   GET /api/odds-logger-tick         -> esegue uno snapshot di tutte le fixture live
#   GET /api/odds-logger-stats        -> conteggi (total rows, unique fixtures, bookies, date range)
#   GET /api/odds-logger-dump?fixture -> tutte le snapshot per una fixture (JSON)
#   GET /api/odds-logger-csv?since=Y  -> export CSV (memory backend)

import os
import json
import time
import csv
import io
import urllib.request
import urllib.parse
from flask import jsonify, request, Response

# Riuso helper da ml_pick per non duplicare
import ml_pick

# ----- Config -----
ODDS_BACKEND = os.getenv('ODDS_BACKEND', 'memory').lower()   # memory | supabase | turso
SUPABASE_URL = os.getenv('SUPABASE_URL', '').rstrip('/')
SUPABASE_KEY = os.getenv('SUPABASE_KEY', '')
TURSO_URL = os.getenv('TURSO_URL', '').rstrip('/')
TURSO_TOKEN = os.getenv('TURSO_TOKEN', '')
TICK_AUTH_TOKEN = os.getenv('TICK_AUTH_TOKEN', '')   # condiviso col cron esterno per proteggere /tick
LOGGER_MAX_FIXTURES = int(os.getenv('LOGGER_MAX_FIXTURES', '40'))   # safety: max fixture per tick

# ----- In-memory ring buffer (fallback / test) -----
# Lista di dict: {fixture_id, league_id, league_name, country, minute, score_h, score_a,
#                 market, bookmaker, quota, ts}
_MEM_ROWS = []
_MEM_CAP = int(os.getenv('LOGGER_MEM_CAP', '200000'))
_LAST_TICK = {'ts': 0, 'rows_added': 0, 'fixtures_scanned': 0, 'errors': []}


def _mem_insert(rows):
    _MEM_ROWS.extend(rows)
    # ring buffer semplice
    if len(_MEM_ROWS) > _MEM_CAP:
        del _MEM_ROWS[:len(_MEM_ROWS) - _MEM_CAP]


def _mem_stats():
    n = len(_MEM_ROWS)
    fixtures = set()
    bookies = set()
    markets = set()
    ts_min = None
    ts_max = None
    for r in _MEM_ROWS:
        fixtures.add(r['fixture_id'])
        bookies.add(r['bookmaker'])
        markets.add(r['market'])
        ts = r['ts']
        if ts_min is None or ts < ts_min: ts_min = ts
        if ts_max is None or ts > ts_max: ts_max = ts
    return {
        'backend': 'memory',
        'total_rows': n,
        'unique_fixtures': len(fixtures),
        'unique_bookmakers': len(bookies),
        'unique_markets': len(markets),
        'bookmakers': sorted(list(bookies)),
        'markets': sorted(list(markets)),
        'ts_min': ts_min,
        'ts_max': ts_max,
        'cap': _MEM_CAP,
    }


def _mem_dump_fixture(fixture_id):
    return [r for r in _MEM_ROWS if r['fixture_id'] == fixture_id]


def _mem_csv(since_ts=0):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(['ts', 'fixture_id', 'league_id', 'league_name', 'country',
                'minute', 'score_h', 'score_a', 'market', 'bookmaker', 'quota'])
    for r in _MEM_ROWS:
        if r['ts'] >= since_ts:
            w.writerow([r['ts'], r['fixture_id'], r.get('league_id'), r.get('league_name'),
                        r.get('country'), r['minute'], r.get('score_h'), r.get('score_a'),
                        r['market'], r['bookmaker'], r['quota']])
    return buf.getvalue()


# ----- Supabase backend (PostgREST) -----

def _sb_headers():
    return {
        'apikey': SUPABASE_KEY,
        'Authorization': 'Bearer ' + SUPABASE_KEY,
        'Content-Type': 'application/json',
        'Prefer': 'return=minimal',
    }


def _sb_insert(rows):
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError('SUPABASE_URL / SUPABASE_KEY not configured')
    url = SUPABASE_URL + '/rest/v1/odds_snapshots'
    # Batch insert in chunk da 500 righe per evitare 413
    for i in range(0, len(rows), 500):
        chunk = rows[i:i+500]
        body = json.dumps(chunk).encode('utf-8')
        req = urllib.request.Request(url, data=body, method='POST', headers=_sb_headers())
        with urllib.request.urlopen(req, timeout=20) as r:
            if r.status not in (200, 201, 204):
                raise RuntimeError('supabase insert status ' + str(r.status))


def _sb_stats():
    url = SUPABASE_URL + '/rest/v1/odds_snapshots?select=count'
    req = urllib.request.Request(url, method='GET', headers={
        'apikey': SUPABASE_KEY,
        'Authorization': 'Bearer ' + SUPABASE_KEY,
        'Prefer': 'count=exact',
        'Range': '0-0',
    })
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            total = r.headers.get('Content-Range', '').split('/')[-1]
            return {'backend': 'supabase', 'total_rows': int(total) if total.isdigit() else None}
    except Exception as e:
        return {'backend': 'supabase', 'error': str(e)}


# ----- Turso backend (libSQL HTTP) -----

def _turso_insert(rows):
    if not TURSO_URL or not TURSO_TOKEN:
        raise RuntimeError('TURSO_URL / TURSO_TOKEN not configured')
    stmts = []
    for r in rows:
        stmts.append({
            'q': '''INSERT INTO odds_snapshots
                    (ts, fixture_id, league_id, league_name, country, minute,
                     score_h, score_a, market, bookmaker, quota) VALUES
                    (?,?,?,?,?,?,?,?,?,?,?)''',
            'params': [r['ts'], r['fixture_id'], r.get('league_id'), r.get('league_name'),
                       r.get('country'), r['minute'], r.get('score_h'), r.get('score_a'),
                       r['market'], r['bookmaker'], r['quota']],
        })
    body = json.dumps({'statements': stmts}).encode('utf-8')
    req = urllib.request.Request(TURSO_URL + '/v2/pipeline', data=body, method='POST', headers={
        'Authorization': 'Bearer ' + TURSO_TOKEN,
        'Content-Type': 'application/json',
    })
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.status


# ----- Dispatcher -----

def _insert_rows(rows):
    if not rows:
        return 0
    if ODDS_BACKEND == 'memory':
        _mem_insert(rows)
    elif ODDS_BACKEND == 'supabase':
        _sb_insert(rows)
    elif ODDS_BACKEND == 'turso':
        _turso_insert(rows)
    else:
        raise RuntimeError('unknown ODDS_BACKEND: ' + ODDS_BACKEND)
    return len(rows)


def _stats():
    if ODDS_BACKEND == 'memory':
        return _mem_stats()
    if ODDS_BACKEND == 'supabase':
        return _sb_stats()
    if ODDS_BACKEND == 'turso':
        return {'backend': 'turso', 'note': 'stats endpoint TODO'}
    return {'backend': 'unknown'}


# ----- Core tick -----

def do_tick():
    """Snapshot: fetch live fixtures + odds, inserisci nel backend."""
    t0 = time.time()
    ts = int(t0)
    result = {
        'ts': ts,
        'fixtures_scanned': 0,
        'rows_added': 0,
        'markets_seen': 0,
        'bookies_seen': 0,
        'errors': [],
    }

    live = ml_pick._get_live_fixtures_af()
    if not isinstance(live, dict) or 'response' not in live:
        result['errors'].append('live fixtures bad response: ' + str(live)[:200])
        _LAST_TICK.update(result)
        return result

    fixtures = live.get('response') or []
    # Safety cap per non consumare troppa quota API in un colpo
    fixtures = fixtures[:LOGGER_MAX_FIXTURES]
    result['fixtures_scanned'] = len(fixtures)

    all_rows = []
    markets_set = set()
    bookies_set = set()

    for fx in fixtures:
        ctx = ml_pick._fixture_to_model_ctx(fx)
        if not ctx or ctx.get('fixture_id') is None:
            continue
        # Solo fixture con minuto > 0 (live vero) e < 100 (no halftime pausa stretta)
        minute = ctx.get('minute')
        if minute is None or minute < 1 or minute > 120:
            continue
        try:
            odds = ml_pick._get_live_odds(ctx['fixture_id'])
        except Exception as e:
            result['errors'].append('odds fetch %s: %s' % (ctx['fixture_id'], str(e)[:100]))
            continue
        parsed = ml_pick._parse_odds_payload(odds) if isinstance(odds, dict) else {}
        for market, bm_map in parsed.items():
            markets_set.add(market)
            for bm, quota in bm_map.items():
                bookies_set.add(bm)
                all_rows.append({
                    'ts': ts,
                    'fixture_id': ctx['fixture_id'],
                    'league_id': ctx.get('league_id'),
                    'league_name': ctx.get('league_name'),
                    'country': ctx.get('country'),
                    'minute': minute,
                    'score_h': ctx.get('score_home'),
                    'score_a': ctx.get('score_away'),
                    'market': market,
                    'bookmaker': bm,
                    'quota': float(quota),
                })

    try:
        added = _insert_rows(all_rows)
    except Exception as e:
        result['errors'].append('insert: ' + str(e)[:200])
        added = 0

    result['rows_added'] = added
    result['markets_seen'] = len(markets_set)
    result['bookies_seen'] = len(bookies_set)
    result['elapsed_s'] = round(time.time() - t0, 2)
    _LAST_TICK.update(result)
    return result


# ----- DDL helper (Supabase / Turso) -----

SUPABASE_DDL = '''
create table if not exists odds_snapshots (
  id bigserial primary key,
  ts bigint not null,
  fixture_id bigint not null,
  league_id bigint,
  league_name text,
  country text,
  minute integer not null,
  score_h integer,
  score_a integer,
  market text not null,
  bookmaker text not null,
  quota double precision not null
);
create index if not exists idx_odds_fix on odds_snapshots(fixture_id);
create index if not exists idx_odds_ts  on odds_snapshots(ts);
create index if not exists idx_odds_market on odds_snapshots(market);
'''.strip()

TURSO_DDL = '''
CREATE TABLE IF NOT EXISTS odds_snapshots (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts INTEGER NOT NULL,
  fixture_id INTEGER NOT NULL,
  league_id INTEGER,
  league_name TEXT,
  country TEXT,
  minute INTEGER NOT NULL,
  score_h INTEGER,
  score_a INTEGER,
  market TEXT NOT NULL,
  bookmaker TEXT NOT NULL,
  quota REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_odds_fix ON odds_snapshots(fixture_id);
CREATE INDEX IF NOT EXISTS idx_odds_ts  ON odds_snapshots(ts);
CREATE INDEX IF NOT EXISTS idx_odds_market ON odds_snapshots(market);
'''.strip()


# ----- Route registration -----

def register(app):
    """Registra le route del logger sull'app Flask."""

    @app.route('/api/odds-logger-tick')
    def api_odds_tick():
        # Protezione: se TICK_AUTH_TOKEN è settato, richiede ?token=X
        if TICK_AUTH_TOKEN:
            tok = request.args.get('token', '')
            if tok != TICK_AUTH_TOKEN:
                return jsonify({'error': 'unauthorized'}), 401
        result = do_tick()
        return jsonify(result)

    @app.route('/api/odds-logger-stats')
    def api_odds_stats():
        return jsonify({
            'backend': ODDS_BACKEND,
            'last_tick': _LAST_TICK,
            'stats': _stats(),
            'config': {
                'supabase_configured': bool(SUPABASE_URL and SUPABASE_KEY),
                'turso_configured': bool(TURSO_URL and TURSO_TOKEN),
                'tick_auth_required': bool(TICK_AUTH_TOKEN),
                'max_fixtures_per_tick': LOGGER_MAX_FIXTURES,
            },
        })

    @app.route('/api/odds-logger-dump')
    def api_odds_dump():
        fixture = request.args.get('fixture', type=int)
        if not fixture:
            return jsonify({'error': 'missing fixture param'}), 400
        if ODDS_BACKEND == 'memory':
            return jsonify({'fixture': fixture, 'rows': _mem_dump_fixture(fixture)})
        return jsonify({'error': 'dump only supported for memory backend'}), 501

    @app.route('/api/odds-logger-csv')
    def api_odds_csv():
        since = request.args.get('since', default=0, type=int)
        if ODDS_BACKEND != 'memory':
            return jsonify({'error': 'csv only supported for memory backend'}), 501
        csv_text = _mem_csv(since)
        return Response(csv_text, mimetype='text/csv',
                        headers={'Content-Disposition': 'attachment; filename=odds_snapshots.csv'})

    @app.route('/api/odds-logger-ddl')
    def api_odds_ddl():
        """Ritorna il DDL del backend configurato, per setup one-time."""
        return jsonify({
            'backend': ODDS_BACKEND,
            'supabase_ddl': SUPABASE_DDL,
            'turso_ddl': TURSO_DDL,
        })
