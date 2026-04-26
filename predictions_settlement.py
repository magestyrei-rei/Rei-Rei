# predictions_settlement.py - Settlement pipeline: popola predictions_log con i risultati FT
# delle partite per cui abbiamo loggato quote live in odds_snapshots.
#
# Flusso:
#   1) trova fixture_id distinti in odds_snapshots non ancora settlati (ft_home IS NULL)
#   2) per ognuno chiama API-Football /fixtures?id=N -> stato + score + teams
#   3) se status in {FT, AET, PEN} -> chiama /fixtures/events?fixture=N -> minuto primo gol
#   4) INSERT OR REPLACE INTO predictions_log
#
# Endpoints (registrati da register()):
#   GET /api/predictions-log-ddl     -> DDL one-time (token)
#   GET /api/predictions-settle      -> esegue settlement (token), param ?limit=N&max_age_days=D
#   GET /api/predictions-log-stats   -> conteggi (open)
#
# Auto-trigger: maybe_settle() chiamato da odds_logger tick (best-effort, ogni 30 min).

import os
import json
import time
import urllib.request
import urllib.parse
from flask import jsonify, request, current_app


# ---------- config ----------
TURSO_URL = os.getenv('TURSO_URL', '').rstrip('/')
TURSO_TOKEN = os.getenv('TURSO_TOKEN', '')
INGEST_TOKEN = os.getenv('INGEST_TOKEN', '')
APISPORTS_KEY = os.getenv('APISPORTS_KEY', '')
APISPORTS_HOST = 'v3.football.api-sports.io'

# Stato in-memory per maybe_settle
_SETTLE_STATE = {'last_run_ts': 0, 'last_settled': 0, 'last_seen': 0, 'last_error': None}


# ---------- turso helpers (mirror of odds_logger style) ----------
def _turso_arg(v):
    if v is None:
        return {'type': 'null', 'value': None}
    if isinstance(v, bool):
        return {'type': 'integer', 'value': '1' if v else '0'}
    if isinstance(v, int):
        return {'type': 'integer', 'value': str(v)}
    if isinstance(v, float):
        return {'type': 'float', 'value': v}
    return {'type': 'text', 'value': str(v)}


def _turso_value(v):
    if v is None or v.get('type') == 'null':
        return None
    t = v.get('type')
    val = v.get('value')
    if t == 'integer':
        try:
            return int(val)
        except Exception:
            return val
    if t == 'float':
        try:
            return float(val)
        except Exception:
            return val
    return val


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
    result = _turso_execute(sql, args, timeout=timeout)
    cols = [c.get('name') for c in result.get('cols', [])]
    rows = []
    for row_arr in result.get('rows', []):
        rows.append({cols[i]: _turso_value(v) for i, v in enumerate(row_arr)})
    return rows


# ---------- API-Football helpers ----------
def _af_get(path, params=None, timeout=20):
    if not APISPORTS_KEY:
        raise RuntimeError('APISPORTS_KEY not configured')
    qs = ('?' + urllib.parse.urlencode(params)) if params else ''
    url = 'https://' + APISPORTS_HOST + path + qs
    req = urllib.request.Request(url, headers={
        'x-apisports-key': APISPORTS_KEY,
    })
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode('utf-8'))


def _fetch_fixture(fid):
    """Ritorna (settled_dict_or_None, raw_status). settled_dict ha tutti i campi necessari."""
    data = _af_get('/fixtures', params={'id': fid})
    resp = (data or {}).get('response') or []
    if not resp:
        return None, 'no_response'
    f = resp[0]
    fixture = f.get('fixture') or {}
    league = f.get('league') or {}
    teams = f.get('teams') or {}
    goals = f.get('goals') or {}
    score = f.get('score') or {}
    status = (fixture.get('status') or {}).get('short') or ''
    if status not in ('FT', 'AET', 'PEN'):
        return None, status
    home = teams.get('home') or {}
    away = teams.get('away') or {}
    ht = score.get('halftime') or {}
    return {
        'fixture_id': fid,
        'league_id': league.get('id'),
        'league_name': league.get('name'),
        'country': league.get('country'),
        'season': league.get('season'),
        'date_utc': fixture.get('date'),
        'home_team_id': home.get('id'),
        'home_team_name': home.get('name'),
        'away_team_id': away.get('id'),
        'away_team_name': away.get('name'),
        'ft_home': goals.get('home'),
        'ft_away': goals.get('away'),
        'ht_home': ht.get('home'),
        'ht_away': ht.get('away'),
        'status': status,
    }, status


def _fetch_first_goal(fid):
    """Ritorna (minute_int_or_None, team_id_or_None). None se 0-0 o errore."""
    try:
        data = _af_get('/fixtures/events', params={'fixture': fid})
        events = (data or {}).get('response') or []
        goals = [e for e in events if (e.get('type') or '').lower() == 'goal']
        if not goals:
            return None, None
        goals.sort(key=lambda e: ((e.get('time') or {}).get('elapsed') or 999,
                                   (e.get('time') or {}).get('extra') or 0))
        first = goals[0]
        t = first.get('time') or {}
        elapsed = t.get('elapsed')
        extra = t.get('extra') or 0
        team = (first.get('team') or {}).get('id')
        if elapsed is None:
            return None, team
        return int(elapsed) + int(extra or 0), team
    except Exception:
        return None, None


# ---------- DDL ----------
DDL = """
CREATE TABLE IF NOT EXISTS predictions_log (
  fixture_id INTEGER PRIMARY KEY,
  league_id INTEGER,
  league_name TEXT,
  country TEXT,
  season INTEGER,
  date_utc TEXT,
  home_team_id INTEGER,
  home_team_name TEXT,
  away_team_id INTEGER,
  away_team_name TEXT,
  ft_home INTEGER,
  ft_away INTEGER,
  ht_home INTEGER,
  ht_away INTEGER,
  status TEXT,
  first_goal_minute INTEGER,
  first_goal_team_id INTEGER,
  settled_ts INTEGER,
  created_ts INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER))
)
""".strip()

DDL_INDEX_LEAGUE = "CREATE INDEX IF NOT EXISTS idx_pl_league ON predictions_log(league_id)"
DDL_INDEX_SETTLED = "CREATE INDEX IF NOT EXISTS idx_pl_settled ON predictions_log(settled_ts)"
DDL_INDEX_FGM = "CREATE INDEX IF NOT EXISTS idx_pl_fgm ON predictions_log(first_goal_minute)"


def _ensure_ddl():
    _turso_execute(DDL)
    _turso_execute(DDL_INDEX_LEAGUE)
    _turso_execute(DDL_INDEX_SETTLED)
    _turso_execute(DDL_INDEX_FGM)


# ---------- core: settle ----------
def _candidate_fixtures(limit=30, max_age_days=7):
    """Trova fixture_id presenti in odds_snapshots ma NON in predictions_log con ft_home valorizzato.
    Limita a partite recenti (max_age_days) per evitare match troppo vecchi.
    """
    cutoff_ts = int(time.time()) - int(max_age_days) * 86400
    sql = (
        "SELECT DISTINCT s.fixture_id FROM odds_snapshots s "
        "LEFT JOIN predictions_log p ON p.fixture_id = s.fixture_id "
        "WHERE s.ts >= ? AND (p.fixture_id IS NULL OR p.ft_home IS NULL) "
        "ORDER BY s.fixture_id DESC LIMIT ?"
    )
    rows = _turso_select_rows(sql, [cutoff_ts, int(limit)])
    return [r['fixture_id'] for r in rows if r.get('fixture_id') is not None]


def _upsert(rec):
    sql = (
        "INSERT OR REPLACE INTO predictions_log "
        "(fixture_id, league_id, league_name, country, season, date_utc, "
        " home_team_id, home_team_name, away_team_id, away_team_name, "
        " ft_home, ft_away, ht_home, ht_away, status, "
        " first_goal_minute, first_goal_team_id, settled_ts) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )
    args = [
        rec['fixture_id'], rec.get('league_id'), rec.get('league_name'), rec.get('country'),
        rec.get('season'), rec.get('date_utc'),
        rec.get('home_team_id'), rec.get('home_team_name'),
        rec.get('away_team_id'), rec.get('away_team_name'),
        rec.get('ft_home'), rec.get('ft_away'), rec.get('ht_home'), rec.get('ht_away'),
        rec.get('status'),
        rec.get('first_goal_minute'), rec.get('first_goal_team_id'),
        int(time.time()),
    ]
    _turso_execute(sql, args)


def settle_batch(limit=30, max_age_days=7):
    """Esegue il settlement di un batch. Ritorna dict con stats."""
    started = time.time()
    try:
        _ensure_ddl()
    except Exception as e:
        return {'error': 'ddl: ' + str(e)[:200]}
    try:
        candidates = _candidate_fixtures(limit=limit, max_age_days=max_age_days)
    except Exception as e:
        return {'error': 'candidates: ' + str(e)[:200]}

    settled = 0
    not_finished = 0
    errors = 0
    skipped = []
    settled_list = []

    for fid in candidates:
        try:
            rec, status = _fetch_fixture(fid)
            if rec is None:
                not_finished += 1
                skipped.append({'fid': fid, 'status': status})
                continue
            fgm, fgt = _fetch_first_goal(fid)
            rec['first_goal_minute'] = fgm
            rec['first_goal_team_id'] = fgt
            _upsert(rec)
            settled += 1
            settled_list.append({'fid': fid, 'score': '%s-%s' % (rec.get('ft_home'), rec.get('ft_away')),
                                 'fgm': fgm, 'league': rec.get('league_name')})
        except Exception as e:
            errors += 1
            skipped.append({'fid': fid, 'err': str(e)[:120]})

    elapsed = round(time.time() - started, 2)
    return {
        'ok': True,
        'candidates': len(candidates),
        'settled': settled,
        'not_finished': not_finished,
        'errors': errors,
        'elapsed_s': elapsed,
        'settled_list': settled_list[:20],
        'skipped_sample': skipped[:10],
    }


def maybe_settle(min_interval_min=30, limit=20, max_age_days=7):
    """Auto-trigger best-effort dal tick di odds_logger.
    Esegue settle_batch solo se sono passati >= min_interval_min minuti dall'ultima esecuzione.
    Cattura ogni eccezione: NON blocca il caller.
    """
    try:
        last_ts = _SETTLE_STATE.get('last_run_ts', 0) or 0
        elapsed = time.time() - last_ts
        if last_ts > 0 and elapsed < min_interval_min * 60:
            return {'skipped': True, 'reason': 'too soon', 'elapsed_min': round(elapsed / 60, 1)}
        _SETTLE_STATE['last_run_ts'] = int(time.time())
        res = settle_batch(limit=limit, max_age_days=max_age_days)
        _SETTLE_STATE['last_settled'] = res.get('settled', 0)
        _SETTLE_STATE['last_seen'] = res.get('candidates', 0)
        if res.get('error'):
            _SETTLE_STATE['last_error'] = res['error']
        else:
            _SETTLE_STATE['last_error'] = None
        return res
    except Exception as e:
        _SETTLE_STATE['last_error'] = str(e)[:200]
        return {'error': str(e)[:200]}


# ---------- routes ----------
def register(app):
    @app.route('/api/predictions-log-ddl')
    def predictions_log_ddl():
        token = request.args.get('token', '')
        if not INGEST_TOKEN or token != INGEST_TOKEN:
            return jsonify({'error': 'forbidden'}), 403
        try:
            _ensure_ddl()
            return jsonify({'ok': True, 'ddl': 'applied'})
        except Exception as e:
            return jsonify({'error': str(e)[:300]}), 500

    @app.route('/api/predictions-settle')
    def predictions_settle():
        token = request.args.get('token', '')
        if not INGEST_TOKEN or token != INGEST_TOKEN:
            return jsonify({'error': 'forbidden'}), 403
        try:
            limit = int(request.args.get('limit', '30'))
        except Exception:
            limit = 30
        try:
            max_age = int(request.args.get('max_age_days', '7'))
        except Exception:
            max_age = 7
        res = settle_batch(limit=limit, max_age_days=max_age)
        return jsonify(res)

    @app.route('/api/predictions-log-stats')
    def predictions_log_stats():
        try:
            _ensure_ddl()
            tot = _turso_select_rows("SELECT COUNT(*) AS n FROM predictions_log")
            settled = _turso_select_rows("SELECT COUNT(*) AS n FROM predictions_log WHERE ft_home IS NOT NULL")
            filtered = _turso_select_rows(
                "SELECT COUNT(*) AS n FROM predictions_log "
                "WHERE ft_home IS NOT NULL AND first_goal_minute IS NOT NULL AND first_goal_minute <= 16"
            )
            by_league = _turso_select_rows(
                "SELECT league_id, league_name, COUNT(*) AS n FROM predictions_log "
                "WHERE ft_home IS NOT NULL GROUP BY league_id ORDER BY n DESC LIMIT 20"
            )
            recent = _turso_select_rows(
                "SELECT fixture_id, league_name, home_team_name, away_team_name, "
                "ft_home, ft_away, first_goal_minute, settled_ts "
                "FROM predictions_log WHERE ft_home IS NOT NULL "
                "ORDER BY settled_ts DESC LIMIT 10"
            )
            return jsonify({
                'ok': True,
                'total': (tot or [{}])[0].get('n', 0),
                'settled': (settled or [{}])[0].get('n', 0),
                'first_goal_le16': (filtered or [{}])[0].get('n', 0),
                'by_league_top20': by_league,
                'recent': recent,
                'auto_state': _SETTLE_STATE,
            })
        except Exception as e:
            return jsonify({'error': str(e)[:300]}), 500
