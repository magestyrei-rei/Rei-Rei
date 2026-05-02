# odds_logger.py - Logger delle quote live: snapshot periodici per costruire dataset storico
# Complementare a ml_pick.py. Supporta backend multipli:
#   - 'memory': dict in RAM (volatile, per test)
#   - 'supabase': Postgres via API REST (persistente, free tier 500MB)
#   - 'turso': libSQL via API v2/pipeline (persistente, free tier 5GB)
#
# Endpoints esposti (registrati da register()):
#   GET /api/odds-logger-tick         -> esegue uno snapshot di tutte le fixture live
#   GET /api/odds-logger-stats        -> conteggi (total rows, unique fixtures, bookies, date range)
#   GET /api/odds-logger-dump?fixture -> tutte le snapshot per una fixture (JSON)
#   GET /api/odds-logger-csv?since=Y  -> export CSV completo (tutti i backend)
#   GET /api/odds-logger-ddl          -> DDL setup one-time

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
# Accept libsql:// scheme (native Turso) by mapping to https:// for HTTP /v2/pipeline API
if TURSO_URL.startswith('libsql://'):
    TURSO_URL = 'https://' + TURSO_URL[len('libsql://'):]
TURSO_TOKEN = os.getenv('TURSO_TOKEN', '')
TICK_AUTH_TOKEN = os.getenv('TICK_AUTH_TOKEN', '')   # condiviso col cron esterno per proteggere /tick
LOGGER_MAX_FIXTURES = int(os.getenv('LOGGER_MAX_FIXTURES', '40'))   # safety: max fixture per tick

# Whitelist leghe da seguire (set vuoto = nessun filtro). Allineato al workflow n8n / dataset gol-16min.
LEAGUE_WHITELIST = {
    # Top europee
    2, 3, 39, 40, 41, 42, 43, 61, 62, 71, 72, 78, 79,
    88, 89, 94, 95, 98, 106, 107, 110, 119, 120,
    129, 131, 135, 136, 138, 140, 141, 144, 145,
    162, 172, 179, 180, 181, 182, 183, 184, 188,
    # Aggiunte da n8n workflow (27 leghe extra)
    197, 203, 204, 207, 208, 210, 218, 219,
    239, 253, 265, 268, 270, 281, 283, 286,
    292, 293, 299, 345, 346, 373, 383, 408,
    942, 943,
}

# ----- In-memory ring buffer (fallback / test) -----
_MEM_ROWS = []
_MEM_CAP = int(os.getenv('LOGGER_MEM_CAP', '200000'))
_LAST_TICK = {'ts': 0, 'rows_added': 0, 'fixtures_scanned': 0, 'errors': []}


def _mem_insert(rows):
    _MEM_ROWS.extend(rows)
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


def _sb_dump_fixture(fixture_id):
    url = (SUPABASE_URL + '/rest/v1/odds_snapshots'
           '?select=ts,fixture_id,league_id,league_name,country,minute,score_h,score_a,market,bookmaker,quota'
           '&fixture_id=eq.' + str(int(fixture_id)) + '&order=ts.asc')
    req = urllib.request.Request(url, method='GET', headers={
        'apikey': SUPABASE_KEY,
        'Authorization': 'Bearer ' + SUPABASE_KEY,
    })
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read().decode('utf-8'))
    except Exception as e:
        return {'error': str(e)[:200]}


def _sb_iter_csv(since_ts=0, page=1000):
    """Generator: stream CSV da Supabase con paginazione."""
    yield 'ts,fixture_id,league_id,league_name,country,minute,score_h,score_a,market,bookmaker,quota\n'
    offset = 0
    while True:
        url = (SUPABASE_URL + '/rest/v1/odds_snapshots'
               '?select=ts,fixture_id,league_id,league_name,country,minute,'
               'score_h,score_a,market,bookmaker,quota'
               '&ts=gte.' + str(int(since_ts)) + '&order=id.asc'
               '&limit=' + str(page) + '&offset=' + str(offset))
        req = urllib.request.Request(url, method='GET', headers={
            'apikey': SUPABASE_KEY,
            'Authorization': 'Bearer ' + SUPABASE_KEY,
        })
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                rows = json.loads(r.read().decode('utf-8'))
        except Exception as e:
            yield '# error: ' + str(e)[:200] + '\n'
            return
        if not rows:
            return
        buf = io.StringIO()
        w = csv.writer(buf)
        for d in rows:
            w.writerow([d.get('ts'), d.get('fixture_id'), d.get('league_id'),
                        d.get('league_name'), d.get('country'), d.get('minute'),
                        d.get('score_h'), d.get('score_a'), d.get('market'),
                        d.get('bookmaker'), d.get('quota')])
        yield buf.getvalue()
        if len(rows) < page:
            return
        offset += page


# ----- Turso backend (libSQL HTTP v2/pipeline) -----

def _turso_arg(value):
    """Converte un valore Python in arg tipizzato Turso v2."""
    if value is None:
        return {'type': 'null'}
    if isinstance(value, bool):
        return {'type': 'integer', 'value': '1' if value else '0'}
    if isinstance(value, int):
        return {'type': 'integer', 'value': str(value)}
    if isinstance(value, float):
        return {'type': 'float', 'value': value}
    return {'type': 'text', 'value': str(value)}


def _turso_value(typed):
    """Decodifica un valore tipizzato Turso v2 in valore Python."""
    if not isinstance(typed, dict):
        return typed
    t = typed.get('type')
    v = typed.get('value')
    if t == 'null' or v is None:
        return None
    if t == 'integer':
        try: return int(v)
        except: return None
    if t == 'float':
        try: return float(v)
        except: return None
    return v


def _turso_pipeline(reqs, timeout=60):
    if not TURSO_URL or not TURSO_TOKEN:
        raise RuntimeError('TURSO_URL / TURSO_TOKEN not configured')
    body = json.dumps({'requests': reqs}).encode('utf-8')
    req = urllib.request.Request(TURSO_URL + '/v2/pipeline', data=body, method='POST', headers={
        'Authorization': 'Bearer ' + TURSO_TOKEN,
        'Content-Type': 'application/json',
    })
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode('utf-8'))


def _turso_execute(sql, args=None, timeout=60):
    """Esegue una singola SQL e ritorna il result dict (cols + rows)."""
    stmt = {'sql': sql}
    if args is not None:
        stmt['args'] = [_turso_arg(a) for a in args]
    resp = _turso_pipeline([
        {'type': 'execute', 'stmt': stmt},
        {'type': 'close'},
    ], timeout=timeout)
    results = resp.get('results') or []
    if not results:
        raise RuntimeError('turso: empty results')
    first = results[0]
    if first.get('type') == 'error':
        raise RuntimeError('turso error: ' + str(first.get('error'))[:300])
    return first.get('response', {}).get('result', {})


def _turso_select_rows(sql, args=None, timeout=60):
    """Esegue SELECT e ritorna lista di dict (cols->valore python)."""
    result = _turso_execute(sql, args, timeout=timeout)
    cols = [c.get('name') for c in result.get('cols', [])]
    rows = []
    for row_arr in result.get('rows', []):
        rows.append({cols[i]: _turso_value(v) for i, v in enumerate(row_arr)})
    return rows


def _turso_insert(rows):
    """Insert batch usando v2/pipeline. Chunked per evitare payload eccessivi."""
    if not rows:
        return 0
    if not TURSO_URL or not TURSO_TOKEN:
        raise RuntimeError('TURSO_URL / TURSO_TOKEN not configured')
    SQL = ('INSERT INTO odds_snapshots '
           '(ts, fixture_id, league_id, league_name, country, minute, '
           'score_h, score_a, market, bookmaker, quota) '
           'VALUES (?,?,?,?,?,?,?,?,?,?,?)')
    BATCH = 200
    for i in range(0, len(rows), BATCH):
        chunk = rows[i:i+BATCH]
        reqs = []
        for r in chunk:
            reqs.append({
                'type': 'execute',
                'stmt': {
                    'sql': SQL,
                    'args': [
                        _turso_arg(r['ts']),
                        _turso_arg(r['fixture_id']),
                        _turso_arg(r.get('league_id')),
                        _turso_arg(r.get('league_name')),
                        _turso_arg(r.get('country')),
                        _turso_arg(r['minute']),
                        _turso_arg(r.get('score_h')),
                        _turso_arg(r.get('score_a')),
                        _turso_arg(r['market']),
                        _turso_arg(r['bookmaker']),
                        _turso_arg(r['quota']),
                    ],
                },
            })
        reqs.append({'type': 'close'})
        _turso_pipeline(reqs)
    return len(rows)


def _turso_stats():
    try:
        rows = _turso_select_rows(
            'SELECT COUNT(*) as n, MIN(ts) as ts_min, MAX(ts) as ts_max, '
            'COUNT(DISTINCT fixture_id) as nfix, COUNT(DISTINCT bookmaker) as nbook, '
            'COUNT(DISTINCT market) as nmark FROM odds_snapshots'
        )
        if not rows:
            return {'backend': 'turso', 'total_rows': 0}
        r = rows[0]
        bms = _turso_select_rows('SELECT DISTINCT bookmaker FROM odds_snapshots LIMIT 50')
        mks = _turso_select_rows('SELECT DISTINCT market FROM odds_snapshots LIMIT 50')
        return {
            'backend': 'turso',
            'total_rows': r.get('n') or 0,
            'ts_min': r.get('ts_min'),
            'ts_max': r.get('ts_max'),
            'unique_fixtures': r.get('nfix') or 0,
            'unique_bookmakers': r.get('nbook') or 0,
            'unique_markets': r.get('nmark') or 0,
            'bookmakers': sorted([x.get('bookmaker') for x in bms if x.get('bookmaker')]),
            'markets': sorted([x.get('market') for x in mks if x.get('market')]),
        }
    except Exception as e:
        return {'backend': 'turso', 'error': str(e)[:300]}


def _turso_dump_fixture(fixture_id):
    return _turso_select_rows(
        'SELECT ts, fixture_id, league_id, league_name, country, minute, '
        'score_h, score_a, market, bookmaker, quota '
        'FROM odds_snapshots WHERE fixture_id = ? ORDER BY ts ASC',
        [fixture_id],
    )


def _turso_iter_csv(since_ts=0, page=2000):
    """Generator: stream CSV in chunk paginati per evitare timeout su grossi dataset."""
    yield 'ts,fixture_id,league_id,league_name,country,minute,score_h,score_a,market,bookmaker,quota\n'
    last_id = 0
    while True:
        rows = _turso_select_rows(
            'SELECT id, ts, fixture_id, league_id, league_name, country, minute, '
            'score_h, score_a, market, bookmaker, quota '
            'FROM odds_snapshots WHERE ts >= ? AND id > ? ORDER BY id ASC LIMIT ?',
            [since_ts, last_id, page],
        )
        if not rows:
            return
        buf = io.StringIO()
        w = csv.writer(buf)
        for r in rows:
            w.writerow([r.get('ts'), r.get('fixture_id'), r.get('league_id'),
                        r.get('league_name'), r.get('country'), r.get('minute'),
                        r.get('score_h'), r.get('score_a'), r.get('market'),
                        r.get('bookmaker'), r.get('quota')])
        yield buf.getvalue()
        last_id = rows[-1].get('id') or last_id
        if len(rows) < page:
            return


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
        return _turso_stats()
    return {'backend': 'unknown'}


def _dump_fixture_any(fixture_id):
    if ODDS_BACKEND == 'memory':
        return _mem_dump_fixture(fixture_id)
    if ODDS_BACKEND == 'supabase':
        return _sb_dump_fixture(fixture_id)
    if ODDS_BACKEND == 'turso':
        return _turso_dump_fixture(fixture_id)
    return []


def _iter_csv_any(since_ts=0):
    """Ritorna un generator che yields chunk CSV per il backend attivo."""
    if ODDS_BACKEND == 'memory':
        def gen():
            yield _mem_csv(since_ts)
        return gen()
    if ODDS_BACKEND == 'supabase':
        return _sb_iter_csv(since_ts)
    if ODDS_BACKEND == 'turso':
        return _turso_iter_csv(since_ts)
    def empty():
        yield ''
    return empty()


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
    if LEAGUE_WHITELIST:
        fixtures = [f for f in fixtures if (f.get('league') or {}).get('id') in LEAGUE_WHITELIST]
    fixtures = fixtures[:LOGGER_MAX_FIXTURES]
    result['fixtures_scanned'] = len(fixtures)

    all_rows = []
    markets_set = set()
    bookies_set = set()

    for fx in fixtures:
        ctx = ml_pick._fixture_to_model_ctx(fx)
        if not ctx or ctx.get('fixture_id') is None:
            continue
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

SUPABASE_DDL = """
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
""".strip()

TURSO_DDL = """
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
""".strip()


# ----- Route registration -----

def register(app):
    """Registra le route del logger sull'app Flask."""

    @app.route('/api/odds-logger-tick')
    def api_odds_tick():
        if TICK_AUTH_TOKEN:
            tok = request.args.get('token', '')
            if tok != TICK_AUTH_TOKEN:
                return jsonify({'error': 'unauthorized'}), 401
        result = do_tick()
        # Auto-trigger recalibrazione Poisson (best-effort, non blocca il tick)
        try:
            import ml_poisson
            ml_poisson.maybe_recalibrate(min_interval_hours=6)
        except Exception:
            pass
        # Auto-trigger settlement (best-effort, non blocca il tick)
        try:
            import predictions_settlement
            predictions_settlement.maybe_settle(min_interval_min=30, limit=20)
        except Exception:
            pass
        return jsonify(result)

    @app.route('/api/odds-logger-probe')
    def api_odds_probe():
        """Debug: ispeziona la chain live-fixtures \u2192 live-odds \u2192 parser per UN fixture."""
        try:
            import ml_pick
            live_bet_full = request.args.get('live_bet_full')
            if live_bet_full:
                # Dump full value entries for one specific bet name across all live fixtures
                ld = ml_pick._apisports_get('/odds/live', {})
                resp = (ld or {}).get('response') if isinstance(ld, dict) else []
                samples = []
                for r0 in (resp or []):
                    for bet in (r0.get('odds') or []):
                        if bet.get('name') == live_bet_full:
                            samples.append({'fid': (r0.get('fixture') or {}).get('id'), 'values': bet.get('values')})
                            break
                    if len(samples) >= 5:
                        break
                return jsonify({'mode': 'live_bet_full', 'name': live_bet_full, 'samples': samples})
            live_bets_full = request.args.get('live_bets_full', '0') == '1'
            if live_bets_full:
                # Iterate /odds/live, dump UNION of all bet names + sample value per bet
                ld = ml_pick._apisports_get('/odds/live', {})
                resp = (ld or {}).get('response') if isinstance(ld, dict) else []
                bet_map = {}  # name -> set of value labels
                fixtures_seen = 0
                for r0 in (resp or []):
                    fixtures_seen += 1
                    for bet in (r0.get('odds') or []):
                        bn = bet.get('name', '')
                        if bn not in bet_map:
                            bet_map[bn] = set()
                        for vd in (bet.get('values') or []):
                            v = str(vd.get('value', ''))[:40]
                            bet_map[bn].add(v)
                out_bets = sorted([
                    {'name': k, 'values': sorted(list(v))[:8]} for k, v in bet_map.items()
                ], key=lambda x: x['name'])
                return jsonify({'mode': 'live_bets_full', 'fixtures_seen': fixtures_seen, 'unique_bets': len(bet_map), 'bets': out_bets})
            bare = request.args.get('bare', '0') == '1'
            status_only = request.args.get('status', '0') == '1'
            if status_only:
                st = ml_pick._apisports_get('/status', {})
                return jsonify({'mode': 'status', 'status_response': st})
            league_lookup = request.args.get('league_lookup')
            if league_lookup:
                ids = []
                for x in league_lookup.split(','):
                    x = x.strip()
                    if x.isdigit():
                        ids.append(int(x))
                out = []
                live_resp = ml_pick._apisports_get('/odds/live/leagues', {})
                live_set = set()
                if isinstance(live_resp, dict):
                    for r in (live_resp.get('response') or []):
                        lid = (r.get('league') or {}).get('id') or r.get('id')
                        if lid is not None:
                            live_set.add(int(lid))
                for lid in ids:
                    info = ml_pick._apisports_get('/leagues', {'id': lid})
                    rsp = (info or {}).get('response') if isinstance(info, dict) else None
                    name = None; country = None; ltype = None
                    if isinstance(rsp, list) and rsp:
                        lg = rsp[0].get('league') or {}
                        ct = rsp[0].get('country') or {}
                        name = lg.get('name'); country = ct.get('name'); ltype = lg.get('type')
                    out.append({'id': lid, 'name': name, 'country': country, 'type': ltype, 'live_odds_covered': lid in live_set})
                return jsonify({'mode': 'league_lookup', 'live_leagues_total': len(live_set), 'leagues': out})
            auto_premkt = request.args.get('auto_premkt', '0') == '1'
            if auto_premkt:
                # Find an upcoming fixture today and run /odds + parser end-to-end
                import datetime
                today = datetime.datetime.utcnow().strftime('%Y-%m-%d')
                fx = ml_pick._apisports_get('/fixtures', {'date': today})
                fxs = (fx or {}).get('response') if isinstance(fx, dict) else None
                if not fxs:
                    return jsonify({'mode': 'auto_premkt', 'error': 'no fixtures today', 'date': today})
                # pick first popular fixture (top-tier league) or just first
                target = fxs[0]
                pf = (target.get('fixture') or {}).get('id')
                pm = ml_pick._apisports_get('/odds', {'fixture': pf})
                pm_resp = (pm or {}).get('response') if isinstance(pm, dict) else None
                parsed = ml_pick._parse_odds_payload(pm) if isinstance(pm, dict) else {}
                bmset = set(); betset = set()
                if isinstance(pm_resp, list):
                    for r0 in pm_resp[:3]:
                        for bm in (r0.get('bookmakers') or []):
                            bmset.add(bm.get('name'))
                            for bet in (bm.get('bets') or []):
                                betset.add(bet.get('name'))
                return jsonify({
                    'mode': 'auto_premkt',
                    'fid': pf,
                    'league': (target.get('league') or {}).get('name'),
                    'home': (target.get('teams') or {}).get('home', {}).get('name'),
                    'away': (target.get('teams') or {}).get('away', {}).get('name'),
                    'pm_results': (pm or {}).get('results') if isinstance(pm, dict) else None,
                    'response_len': len(pm_resp) if isinstance(pm_resp, list) else None,
                    'bookmakers': sorted(list(bmset))[:30],
                    'bet_names_in_odds': sorted(list(betset))[:60],
                    'parsed_markets': list((parsed or {}).keys()),
                    'parsed_count_per_market': {k: len(v or {}) for k, v in (parsed or {}).items()},
                })
            bets_only = request.args.get('bets', '0') == '1'
            if bets_only:
                bb = ml_pick._apisports_get('/odds/bets', {})
                resp = (bb or {}).get('response') if isinstance(bb, dict) else None
                names = [r.get('name') for r in (resp or [])]
                return jsonify({'mode': 'bets', 'count': len(names), 'names': names})
            premkt_fid = request.args.get('premkt_fixture')
            if premkt_fid:
                try: pf = int(premkt_fid)
                except Exception: return jsonify({'error': 'invalid premkt_fixture'})
                pm = ml_pick._apisports_get('/odds', {'fixture': pf})
                resp = (pm or {}).get('response') if isinstance(pm, dict) else None
                bmset = set(); betset = set()
                for r0 in (resp or [])[:3]:
                    for bm in (r0.get('bookmakers') or []):
                        bmset.add(bm.get('name'))
                        for bet in (bm.get('bets') or []):
                            betset.add(bet.get('name'))
                return jsonify({'mode': 'premkt', 'fid': pf, 'pm_keys': list((pm or {}).keys()) if isinstance(pm, dict) else None, 'pm_results': (pm or {}).get('results') if isinstance(pm, dict) else None, 'bookmakers': sorted(list(bmset))[:30], 'bet_names': sorted(list(betset))[:60]})
            if bare:
                # Diagnostica plan API-Football: chiama /odds/live SENZA filtro fixture
                bare_data = ml_pick._apisports_get('/odds/live', {})
                # Dump first item raw shape to understand
                if isinstance(bare_data, dict):
                    rsp0 = (bare_data.get('response') or [None])[0]
                    if rsp0 is not None:
                        return jsonify({'mode': 'bare_dump', 'first_item_keys': list(rsp0.keys()), 'first_item_sample': {k: (str(rsp0.get(k))[:300] if not isinstance(rsp0.get(k), (list, dict)) else (rsp0.get(k)[:1] if isinstance(rsp0.get(k), list) else rsp0.get(k))) for k in rsp0.keys()}, 'results': bare_data.get('results')})
                if not isinstance(bare_data, dict):
                    return jsonify({'bare_error': str(bare_data)[:300]})
                resp = bare_data.get('response') or []
                bm_set = set()
                bet_set = set()
                for r0 in (resp[:5] if isinstance(resp, list) else []):
                    for bm in (r0.get('bookmakers') or []):
                        bm_set.add(bm.get('name'))
                        for bet in (bm.get('bets') or []):
                            bet_set.add(bet.get('name'))
                return jsonify({
                    'mode': 'bare',
                    'bare_keys': list(bare_data.keys()),
                    'bare_results': bare_data.get('results'),
                    'bare_errors': bare_data.get('errors'),
                    'response_len': len(resp) if isinstance(resp, list) else None,
                    'first5_bookmakers': sorted(list(bm_set))[:20],
                    'first5_bet_names': sorted(list(bet_set))[:30],
                })
            live = ml_pick._get_live_fixtures_af()
            fixtures = (live.get('response') if isinstance(live, dict) else None) or []
            fid_arg = request.args.get('fixture')
            target = None
            if fid_arg:
                try:
                    fid_q = int(fid_arg)
                    target = next((f for f in fixtures if (f.get('fixture') or {}).get('id') == fid_q), None)
                except Exception:
                    target = None
            if target is None:
                if not fixtures:
                    if fid_arg:
                        try:
                            fid = int(fid_arg)
                        except Exception:
                            return jsonify({'error': 'invalid fixture arg'})
                        odds = ml_pick._get_live_odds(fid)
                        parsed = ml_pick._parse_odds_payload(odds) if isinstance(odds, dict) else {}
                        response = (odds or {}).get('response') if isinstance(odds, dict) else None
                        shape = {
                            'odds_keys': list((odds or {}).keys()) if isinstance(odds, dict) else None,
                            'odds_results': (odds or {}).get('results') if isinstance(odds, dict) else None,
                            'odds_errors': (odds or {}).get('errors') if isinstance(odds, dict) else None,
                            'response_len': len(response) if isinstance(response, list) else None,
                        }
                        sample_raw = None
                        if isinstance(response, list) and response:
                            r0 = response[0] or {}
                            bms = r0.get('bookmakers') or []
                            shape['bookmakers_count'] = len(bms)
                            shape['bm_names'] = [b.get('name') for b in bms[:6]]
                            if bms:
                                bm0 = bms[0]
                                bets = bm0.get('bets') or []
                                shape['first_bm_bets_count'] = len(bets)
                                shape['first_bm_bet_names'] = [b.get('name') for b in bets[:30]]
                                if bets:
                                    sample_raw = {
                                        'bm': bm0.get('name'),
                                        'bet0_name': bets[0].get('name'),
                                        'bet0_values': bets[0].get('values'),
                                    }
                        return jsonify({
                            'fid': fid,
                            'mode': 'forced',
                            'shape': shape,
                            'parsed_markets': list(parsed.keys()) if isinstance(parsed, dict) else None,
                            'parsed_count': sum(len(v or {}) for v in (parsed or {}).values()) if isinstance(parsed, dict) else 0,
                            'sample_raw': sample_raw,
                        })
                    return jsonify({
                        'error': 'no live fixtures',
                        'live_keys': list((live or {}).keys()) if isinstance(live, dict) else None,
                        'live_results': (live or {}).get('results') if isinstance(live, dict) else None,
                        'live_errors': (live or {}).get('errors') if isinstance(live, dict) else None,
                        'hint': 'pass ?fixture=NNN per forzare probe, oppure ?bare=1 per /odds/live globale',
                    })
                target = fixtures[0]
            fid = (target.get('fixture') or {}).get('id')
            try:
                ctx = ml_pick._fixture_to_model_ctx(target)
            except Exception as e:
                ctx = {'_ctx_err': str(e)[:200]}
            try:
                odds = ml_pick._get_live_odds(fid)
            except Exception as e:
                return jsonify({'fid': fid, 'odds_err': str(e)[:200]})
            parsed = ml_pick._parse_odds_payload(odds) if isinstance(odds, dict) else {}
            response = (odds or {}).get('response') if isinstance(odds, dict) else None
            shape = {
                'odds_keys': list((odds or {}).keys()) if isinstance(odds, dict) else None,
                'odds_results': (odds or {}).get('results') if isinstance(odds, dict) else None,
                'odds_errors': (odds or {}).get('errors') if isinstance(odds, dict) else None,
                'response_len': len(response) if isinstance(response, list) else None,
            }
            sample_raw = None
            if isinstance(response, list) and response:
                r0 = response[0] or {}
                bms = r0.get('bookmakers') or []
                shape['bookmakers_count'] = len(bms)
                shape['bm_names'] = [b.get('name') for b in bms[:6]]
                if bms:
                    bm0 = bms[0]
                    bets = bm0.get('bets') or []
                    shape['first_bm_bets_count'] = len(bets)
                    shape['first_bm_bet_names'] = [b.get('name') for b in bets[:30]]
                    if bets:
                        sample_raw = {
                            'bm': bm0.get('name'),
                            'bet0_name': bets[0].get('name'),
                            'bet0_values': bets[0].get('values'),
                        }
            return jsonify({
                'fid': fid,
                'ctx_minute': ctx.get('minute') if isinstance(ctx, dict) else None,
                'ctx_keys': list(ctx.keys()) if isinstance(ctx, dict) else None,
                'shape': shape,
                'parsed_markets': list(parsed.keys()) if isinstance(parsed, dict) else None,
                'parsed_count': sum(len(v or {}) for v in (parsed or {}).values()) if isinstance(parsed, dict) else 0,
                'sample_raw': sample_raw,
            })
        except Exception as e:
            return jsonify({'error': str(e)[:300]}), 500


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
        rows = _dump_fixture_any(fixture)
        return jsonify({'backend': ODDS_BACKEND, 'fixture': fixture, 'rows': rows})

    @app.route('/api/odds-logger-csv')
    def api_odds_csv():
        """Export CSV completo (o filtrato per ts >= since). Funziona su tutti i backend."""
        since = request.args.get('since', default=0, type=int)
        gen = _iter_csv_any(since)
        fname = 'odds_snapshots_' + ODDS_BACKEND + '_' + str(int(time.time())) + '.csv'
        return Response(gen, mimetype='text/csv',
                        headers={'Content-Disposition': 'attachment; filename=' + fname})

    @app.route('/api/odds-logger-ddl')
    def api_odds_ddl():
        """Ritorna il DDL. Con ?execute=1 esegue il DDL su Turso (one-time setup)."""
        execute = request.args.get('execute', default=0, type=int)
        if execute and ODDS_BACKEND == 'turso':
            if TICK_AUTH_TOKEN:
                tok = request.args.get('token', '')
                if tok != TICK_AUTH_TOKEN:
                    return jsonify({'error': 'unauthorized'}), 401
            results = []
            errors = []
            stmts = [s.strip() for s in TURSO_DDL.split(';') if s.strip()]
            for sql in stmts:
                try:
                    _turso_execute(sql)
                    results.append({'sql': sql[:80], 'ok': True})
                except Exception as e:
                    errors.append({'sql': sql[:80], 'error': str(e)[:300]})
            return jsonify({
                'backend': ODDS_BACKEND,
                'executed': len(results),
                'errors': errors,
                'results': results,
            })
        return jsonify({
            'backend': ODDS_BACKEND,
            'supabase_ddl': SUPABASE_DDL,
            'turso_ddl': TURSO_DDL,
        })
