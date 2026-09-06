# predictions_settlement.py - Settlement pipeline: popola predictions_log con i risultati FT
# delle partite per cui abbiamo loggato quote live in odds_snapshots.
#
# Flusso:
# 1) trova fixture_id distinti in odds_snapshots non ancora settlati (ft_home IS NULL)
# 2) per ognuno chiama API-Football /fixtures?id=N -> stato + score + teams
# 3) se status in {FT, AET, PEN} -> chiama /fixtures/events?fixture=N -> minuto primo gol
# 4) INSERT OR REPLACE INTO predictions_log
#
# Endpoints (registrati da register()):
#   GET /api/predictions-log-ddl        -> DDL one-time (token)
#   GET /api/predictions-settle         -> esegue settlement (token), param ?limit=N&max_age_days=D
#   GET /api/predictions-log-stats      -> conteggi (open)
#   GET /api/ml-picks-accuracy          -> accuratezza per mercato, campionato, mercatoÃÂÃÂcampionato
#   GET /api/ml-accuracy-trend          -> curva di apprendimento settimanale (param: ?market=&league=)
#
# Auto-trigger: maybe_settle() chiamato da odds_logger tick (best-effort, ogni 30 min).

import os
import json
import time
import urllib.request
import urllib.parse
from flask import jsonify, request, current_app

# ---------- config ----------

def _normalize_turso_url(u):
    if not u:
        return ''
    u = u.rstrip('/')
    if u.startswith('libsql://'):
        u = 'https://' + u[len('libsql://'):]
    return u

TURSO_URL   = _normalize_turso_url(os.getenv('TURSO_URL', ''))
TURSO_TOKEN = os.getenv('TURSO_TOKEN', '')
INGEST_TOKEN  = os.getenv('INGEST_TOKEN', '')
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
    t   = v.get('type')
    val = v.get('value')
    if t == 'integer':
        try:    return int(val)
        except: return val
    if t == 'float':
        try:    return float(val)
        except: return val
    return val

def _turso_pipeline(reqs, timeout=60):
    if not TURSO_URL or not TURSO_TOKEN:
        raise RuntimeError('TURSO_URL / TURSO_TOKEN not configured')
    body = json.dumps({'requests': reqs}).encode('utf-8')
    req  = urllib.request.Request(TURSO_URL + '/v2/pipeline', data=body, method='POST', headers={
        'Authorization': 'Bearer ' + TURSO_TOKEN,
        'Content-Type':  'application/json',
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
    cols   = [c.get('name') for c in result.get('cols', [])]
    rows   = []
    for row_arr in result.get('rows', []):
        rows.append({cols[i]: _turso_value(v) for i, v in enumerate(row_arr)})
    return rows

# ---------- API-Football helpers ----------

def _af_get(path, params=None, timeout=20):
    if not APISPORTS_KEY:
        raise RuntimeError('APISPORTS_KEY not configured')
    qs  = ('?' + urllib.parse.urlencode(params)) if params else ''
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
    f       = resp[0]
    fixture = f.get('fixture') or {}
    league  = f.get('league')  or {}
    teams   = f.get('teams')   or {}
    goals   = f.get('goals')   or {}
    score   = f.get('score')   or {}
    status  = (fixture.get('status') or {}).get('short') or ''
    if status not in ('FT', 'AET', 'PEN'):
        return None, status
    home = teams.get('home') or {}
    away = teams.get('away') or {}
    ht   = score.get('halftime') or {}
    return {
        'fixture_id':      fid,
        'league_id':       league.get('id'),
        'league_name':     league.get('name'),
        'country':         league.get('country'),
        'season':          league.get('season'),
        'date_utc':        fixture.get('date'),
        'home_team_id':    home.get('id'),
        'home_team_name':  home.get('name'),
        'away_team_id':    away.get('id'),
        'away_team_name':  away.get('name'),
        'ft_home':         goals.get('home'),
        'ft_away':         goals.get('away'),
        'ht_home':         ht.get('home'),
        'ht_away':         ht.get('away'),
        'status':          status,
    }, status

def _fetch_first_goal(fid):
    """Ritorna (minute_int_or_None, team_id_or_None). None se 0-0 o errore."""
    try:
        data   = _af_get('/fixtures/events', params={'fixture': fid})
        events = (data or {}).get('response') or []
        goals  = [e for e in events if (e.get('type') or '').lower() == 'goal']
        if not goals:
            return None, None
        goals.sort(key=lambda e: ((e.get('time') or {}).get('elapsed') or 999,
                                   (e.get('time') or {}).get('extra')   or 0))
        first   = goals[0]
        t       = first.get('time') or {}
        elapsed = t.get('elapsed')
        extra   = t.get('extra') or 0
        team    = (first.get('team') or {}).get('id')
        if elapsed is None:
            return None, team
        return int(elapsed) + int(extra or 0), team
    except Exception:
        return None, None

# ---------- DDL ----------

DDL = """
CREATE TABLE IF NOT EXISTS predictions_log (
  fixture_id          INTEGER PRIMARY KEY,
  league_id           INTEGER,
  league_name         TEXT,
  country             TEXT,
  season              INTEGER,
  date_utc            TEXT,
  home_team_id        INTEGER,
  home_team_name      TEXT,
  away_team_id        INTEGER,
  away_team_name      TEXT,
  ft_home             INTEGER,
  ft_away             INTEGER,
  ht_home             INTEGER,
  ht_away             INTEGER,
  status              TEXT,
  first_goal_minute   INTEGER,
  first_goal_team_id  INTEGER,
  settled_ts          INTEGER,
  created_ts          INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER))
)
""".strip()

DDL_INDEX_LEAGUE  = "CREATE INDEX IF NOT EXISTS idx_pl_league  ON predictions_log(league_id)"
DDL_INDEX_SETTLED = "CREATE INDEX IF NOT EXISTS idx_pl_settled ON predictions_log(settled_ts)"
DDL_INDEX_FGM     = "CREATE INDEX IF NOT EXISTS idx_pl_fgm     ON predictions_log(first_goal_minute)"

# ---------- ml_picks_log: traccia i pick effettivi del modello ML ----------

DDL_PICKS_LOG = """
CREATE TABLE IF NOT EXISTS ml_picks_log (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  fixture_id   INTEGER NOT NULL,
  league_name  TEXT,
  country      TEXT,
  home_team    TEXT,
  away_team    TEXT,
  market       TEXT NOT NULL,
  model_prob   REAL,
  bookie_quota REAL,
  edge_pct     REAL,
  logged_at    INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER)),
  ft_home      INTEGER,
  ft_away      INTEGER,
  result       TEXT,
  settled_at   INTEGER,
  UNIQUE(fixture_id, market)
)
""".strip()

def _ensure_picks_ddl():
    _turso_execute(DDL_PICKS_LOG)
    try:
        _turso_execute("ALTER TABLE ml_picks_log ADD COLUMN country TEXT")
    except Exception:
        pass  # colonna gia' presente
    # Backfill country dai dati predictions_log (idempotente - solo righe con country NULL)
    try:
        _turso_execute(
            "UPDATE ml_picks_log SET country = ("
            "  SELECT pl.country FROM predictions_log pl"
            "  WHERE pl.fixture_id = ml_picks_log.fixture_id LIMIT 1"
            ") WHERE country IS NULL"
        )
    except Exception:
        pass

def log_picks(fixture_id, ctx, picks):
    """Salva pick ML per una fixture. UNIQUE(fixture_id,market) ignora duplicati."""
    try:
        _ensure_picks_ddl()
        for p in (picks or []):
            _turso_execute(
                "INSERT OR IGNORE INTO ml_picks_log "
                "(fixture_id, league_name, country, home_team, away_team, market, model_prob, bookie_quota, edge_pct) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                [fixture_id, ctx.get('league_name'), ctx.get('country'), ctx.get('home'), ctx.get('away'),
                 p.get('market'), p.get('model_prob'),
                 p.get('quota') or p.get('bookie_quota'), p.get('edge_pct')]
            )
    except Exception:
        pass

def _settle_picks(fixture_id, ft_home, ft_away):
    """Marca WIN/LOSS i pick di questa fixture dopo il risultato FT."""
    if ft_home is None or ft_away is None:
        return
    try:
        _ensure_picks_ddl()
        picks = _turso_select_rows(
            "SELECT id, market FROM ml_picks_log WHERE fixture_id=? AND result IS NULL",
            [fixture_id]
        )
        if not picks:
            return
        total = ft_home + ft_away
        btts  = ft_home > 0 and ft_away > 0
        now   = int(time.time())
        wins_map = {
            'over_1_5':  total > 1,  'over_2_5':  total > 2,  'over_3_5':  total > 3,
            'under_1_5': total < 2,  'under_2_5': total < 3,  'under_3_5': total < 4,
            'btts_si': btts, 'btts_no': not btts,
            '1': ft_home > ft_away, 'X': ft_home == ft_away, '2': ft_away > ft_home,
        }
        for p in picks:
            mkt = (p.get('market') or '')
            if mkt not in wins_map:
                continue
            res = 'WIN' if wins_map[mkt] else 'LOSS'
            _turso_execute(
                "UPDATE ml_picks_log SET result=?,ft_home=?,ft_away=?,settled_at=? WHERE id=?",
                [res, ft_home, ft_away, now, p['id']]
            )
    except Exception:
        pass

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
    # Settle picks per questa fixture
    _settle_picks(rec['fixture_id'], rec.get('ft_home'), rec.get('ft_away'))

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

    settled      = 0
    not_finished = 0
    errors       = 0
    skipped      = []
    settled_list = []

    for fid in candidates:
        try:
            rec, status = _fetch_fixture(fid)
            if rec is None:
                not_finished += 1
                skipped.append({'fid': fid, 'status': status})
                continue
            fgm, fgt = _fetch_first_goal(fid)
            rec['first_goal_minute']  = fgm
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
        'ok':           True,
        'candidates':   len(candidates),
        'settled':      settled,
        'not_finished': not_finished,
        'errors':       errors,
        'elapsed_s':    elapsed,
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
        # Settla anche pick ML orfani (non in odds_snapshots)
        try:
            settle_ml_picks_orphan(limit=30, max_age_days=14)
        except Exception:
            pass
        _SETTLE_STATE['last_settled'] = res.get('settled', 0)
        _SETTLE_STATE['last_seen']    = res.get('candidates', 0)
        if res.get('error'):
            _SETTLE_STATE['last_error'] = res['error']
        else:
            _SETTLE_STATE['last_error'] = None
        return res
    except Exception as e:
        _SETTLE_STATE['last_error'] = str(e)[:200]
        return {'error': str(e)[:200]}

def settle_ml_picks_orphan(limit=50, max_age_days=14):
    """Settla pick in ml_picks_log rimasti orfani (fixture_id non in odds_snapshots).
    Chiama direttamente l'API per ogni fixture_id pendente.
    """
    try:
        _ensure_picks_ddl()
        cutoff = int(time.time()) - int(max_age_days) * 86400
        rows = _turso_select_rows(
            "SELECT DISTINCT fixture_id FROM ml_picks_log "
            "WHERE result IS NULL AND logged_at >= ? LIMIT ?",
            [cutoff, int(limit)]
        )
        if not rows:
            return {'settled': 0, 'not_finished': 0, 'n': 0}
        settled      = 0
        not_finished = 0
        for row in rows:
            fid = row.get('fixture_id')
            if not fid:
                continue
            try:
                rec, status = _fetch_fixture(fid)
                if rec is None:
                    not_finished += 1
                    continue
                _settle_picks(fid, rec.get('ft_home'), rec.get('ft_away'))
                settled += 1
            except Exception:
                pass
            time.sleep(0.3)
        return {'settled': settled, 'not_finished': not_finished, 'n': len(rows)}
    except Exception as e:
        return {'error': str(e)[:300]}


# ---------- early_goal_log: match con primo gol <=16' da n8n ----------

DDL_EGL = """
CREATE TABLE IF NOT EXISTS early_goal_log (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  fixture_id      INTEGER UNIQUE,
  league_id       INTEGER,
  league_name     TEXT,
  country         TEXT,
  home_team       TEXT,
  away_team       TEXT,
  ht_home         INTEGER,
  ht_away         INTEGER,
  ft_home         INTEGER,
  ft_away         INTEGER,
  first_goal_min  INTEGER,
  match_date      TEXT,
  source          TEXT DEFAULT 'n8n',
  logged_ts       INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER))
)
""".strip()

def _ensure_egl():
    _turso_execute(DDL_EGL)
    _turso_execute("CREATE INDEX IF NOT EXISTS idx_egl_league ON early_goal_log(league_id)")
    _turso_execute("CREATE INDEX IF NOT EXISTS idx_egl_date   ON early_goal_log(match_date)")

# ---------- user_bets: paper trading dell'utente (con edge vs base-rate) ----------

import sqlite3 as _sqlite3
import re as _re_bets
import threading as _threading

_LOCAL_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'football.db')

DDL_USER_BETS = """
CREATE TABLE IF NOT EXISTS user_bets (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  created_ts   INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER)),
  fixture_id   INTEGER,
  league_id    INTEGER,
  league_name  TEXT,
  home_team    TEXT,
  away_team    TEXT,
  match_date   TEXT,
  market       TEXT,
  minute       INTEGER,
  score_home   INTEGER,
  score_away   INTEGER,
  odds         REAL,
  stake        REAL DEFAULT 1,
  base_rate    REAL,
  fair_odds    REAL,
  edge_pct     REAL,
  sample_n     INTEGER,
  status       TEXT DEFAULT 'pending',
  ft_home      INTEGER,
  ft_away      INTEGER,
  result       TEXT,
  pnl          REAL,
  settled_ts   INTEGER
)
""".strip()

def _ensure_bets_ddl():
    _turso_execute(DDL_USER_BETS)
    try:
        _turso_execute("ALTER TABLE user_bets ADD COLUMN placed INTEGER DEFAULT 1")
    except Exception:
        pass  # colonna gia' presente

def _bet_market_win(market, fh, fa):
    """Esito di un mercato dato il risultato FT. None se non valutabile."""
    if fh is None or fa is None:
        return None
    tot = fh + fa
    btts = (fh > 0 and fa > 0)
    table = {
        'over_0_5': tot > 0, 'over_1_5': tot > 1, 'over_2_5': tot > 2,
        'over_3_5': tot > 3, 'over_4_5': tot > 4,
        'under_1_5': tot < 2, 'under_2_5': tot < 3, 'under_3_5': tot < 4, 'under_4_5': tot < 5,
        'over_1_5_ht': tot > 1, 'over_2_5_ht': tot > 2, 'under_1_5_ht': tot < 2, 'under_2_5_ht': tot < 3,
        'over_0_5_st': tot > 0, 'over_1_5_st': tot > 1, 'over_2_5_st': tot > 2,
        'under_1_5_st': tot < 2, 'under_2_5_st': tot < 3,
        'gol_casa_st': fh > 0, 'gol_ospite_st': fa > 0,
        'gol_casa': fh > 0, 'gol_ospite': fa > 0,
        'btts_si': btts, 'btts_no': not btts,
        '1': fh > fa, 'X': fh == fa, '2': fa > fh,
    }
    return table.get(market)

def _bet_timeline(goals_html, goals_text):
    """Lista ordinata (minuto, is_away|None) dei gol di una partita."""
    out = []
    gh = goals_html or ''
    if gh.strip():
        for part in _re_bets.split(r'(<span[^>]*away-goal[^>]*>.*?</span>)', gh, flags=_re_bets.DOTALL | _re_bets.IGNORECASE):
            if not part or not part.strip():
                continue
            away = bool(_re_bets.match(r'<span[^>]*away-goal', part, _re_bets.IGNORECASE))
            for tok in part.split(','):
                m = _re_bets.search(r'\d+', tok)
                if m:
                    out.append((int(m.group()), away))
    else:
        for tok in (goals_text or '').split(','):
            m = _re_bets.search(r'\d+', tok)
            if m:
                out.append((int(m.group()), None))
    out.sort(key=lambda x: x[0])
    return out

def _bet_base_rate(league_id, market, minute, sh, sa):
    """% storica del mercato dato lo stato (minuto, punteggio) nella lega. Ritorna (pct|None, n)."""
    try:
        con = _sqlite3.connect(_LOCAL_DB)
        con.row_factory = _sqlite3.Row
        rows = con.execute(
            "SELECT goals_html, goals_text, ft_home, ft_away FROM matches WHERE league_id=?",
            (int(league_id),)
        ).fetchall()
        con.close()
    except Exception:
        return None, 0
    cur_total = (sh or 0) + (sa or 0)
    is_total = market.startswith('over_') or market.startswith('under_')
    matched = won = 0
    for r in rows:
        fh, fa = r['ft_home'], r['ft_away']
        if fh is None or fa is None:
            continue
        tl = _bet_timeline(r['goals_html'], r['goals_text'])
        hM = sum(1 for (mn, aw) in tl if mn <= minute and aw is False)
        aM = sum(1 for (mn, aw) in tl if mn <= minute and aw is True)
        unkM = sum(1 for (mn, aw) in tl if mn <= minute and aw is None)
        totM = hM + aM + unkM
        if is_total:
            if totM != cur_total:
                continue
        elif market in ('btts_si', 'btts_no'):
            if unkM > 0:
                continue
            if (hM > 0) != ((sh or 0) > 0) or (aM > 0) != ((sa or 0) > 0):
                continue
        else:  # 1 / X / 2
            if unkM > 0:
                continue
            if hM != (sh or 0) or aM != (sa or 0):
                continue
        if market.endswith('_ht'):
            htM = sum(1 for (mn, aw) in tl if mn <= 45)
            _hthr = {'over_1_5_ht': 2, 'over_2_5_ht': 3, 'under_1_5_ht': 2, 'under_2_5_ht': 3}.get(market)
            w = None if _hthr is None else ((htM >= _hthr) if market.startswith('over_') else (htM < _hthr))
        else:
            w = _bet_market_win(market, fh, fa)
        if w is None:
            continue
        matched += 1
        if w:
            won += 1
    if matched == 0:
        return None, 0
    return round(100.0 * won / matched, 1), matched

def _norm_toks(s):
    """Token normalizzati di un nome squadra (minuscolo, senza accenti, >=3 lettere)."""
    import unicodedata as _ud, re as _re2
    s = (s or '').lower()
    s = ''.join(c for c in _ud.normalize('NFKD', s) if not _ud.combining(c))
    return set(t for t in _re2.findall(r'[a-z0-9]+', s) if len(t) >= 3)

_GENERIC_TEAM = {'deportivo', 'atletico', 'club', 'sportivo', 'sporting', 'real',
                 'afc', 'san', 'del', 'dos', 'das', 'futbol', 'calcio', 'sport'}

def _name_match(a, b):
    """True se i nomi squadra condividono almeno un token DISTINTIVO (non generico),
    cosi' 'Deportivo Moron' non combacia con 'Deportivo Riestra'."""
    ta, tb = _norm_toks(a), _norm_toks(b)
    if not ta or not tb:
        return False
    common = (ta & tb) - _GENERIC_TEAM
    return bool(common)

def _settle_user_bets(limit=80):
    """Settla le giocate pending: pass 1 quelle con fixture_id; pass 2 quelle
    scritte a mano (senza fixture_id) agganciandole per lega+data+squadre."""
    _ensure_bets_ddl()
    rows = _turso_select_rows(
        "SELECT id, fixture_id, market, odds, stake FROM user_bets "
        "WHERE result IS NULL AND fixture_id IS NOT NULL LIMIT ?", [int(limit)]
    )
    settled = 0
    for r in rows:
        try:
            rec, status = _fetch_fixture(r['fixture_id'])
            if rec is None:
                continue
            fh, fa = rec.get('ft_home'), rec.get('ft_away')
            _mk = r.get('market')
            if _mk and _mk.endswith('_ht'):
                w = _bet_market_win(_mk, rec.get('ht_home'), rec.get('ht_away'))
            else:
                w = _bet_market_win(_mk, fh, fa)
            if w is None:
                continue
            stake = r.get('stake') or 1.0
            odds = r.get('odds') or 0.0
            pnl = stake * (odds - 1.0) if w else -stake
            _turso_execute(
                "UPDATE user_bets SET status='settled', ft_home=?, ft_away=?, result=?, pnl=?, settled_ts=? WHERE id=?",
                [fh, fa, ('WIN' if w else 'LOSS'), round(pnl, 2), int(time.time()), r['id']]
            )
            settled += 1
            time.sleep(0.2)
        except Exception:
            pass
    # --- pass 2: giocate scritte a mano (senza fixture_id) -> aggancio per lega+data+squadre ---
    rows2 = _turso_select_rows(
        "SELECT id, league_id, match_date, home_team, away_team, market, odds, stake "
        "FROM user_bets WHERE result IS NULL AND fixture_id IS NULL "
        "AND league_id IS NOT NULL AND match_date IS NOT NULL LIMIT ?", [int(limit)]
    )
    for r in rows2:
        try:
            date = (r.get('match_date') or '')
            lid = r.get('league_id')
            if not lid or len(date) < 10:
                continue
            yr = int(date[:4])
            fx = None
            for season in (yr, yr - 1):
                data = _af_get('/fixtures', params={'league': int(lid), 'season': season, 'date': date})
                for f in (data.get('response') or []):
                    t = f.get('teams') or {}
                    if _name_match(r.get('home_team'), (t.get('home') or {}).get('name')) and \
                       _name_match(r.get('away_team'), (t.get('away') or {}).get('name')):
                        fx = f
                        break
                if fx:
                    break
            if not fx:
                continue
            fid = (fx.get('fixture') or {}).get('id')
            if fid:
                _turso_execute("UPDATE user_bets SET fixture_id=? WHERE id=?", [fid, r['id']])
            st = ((fx.get('fixture') or {}).get('status') or {}).get('short')
            if st not in ('FT', 'AET', 'PEN'):
                continue  # non ancora finita: si settla al prossimo giro via fixture_id
            g = fx.get('goals') or {}
            fh, fa = g.get('home'), g.get('away')
            _mk = r.get('market')
            if _mk and _mk.endswith('_ht'):
                _hts = (fx.get('score') or {}).get('halftime') or {}
                w = _bet_market_win(_mk, _hts.get('home'), _hts.get('away'))
            else:
                w = _bet_market_win(_mk, fh, fa)
            if w is None:
                continue
            stake = r.get('stake') or 1.0
            odds = r.get('odds') or 0.0
            pnl = stake * (odds - 1.0) if w else -stake
            _turso_execute(
                "UPDATE user_bets SET status='settled', ft_home=?, ft_away=?, result=?, pnl=?, settled_ts=? WHERE id=?",
                [fh, fa, ('WIN' if w else 'LOSS'), round(pnl, 2), int(time.time()), r['id']]
            )
            settled += 1
            time.sleep(0.2)
        except Exception:
            pass
    return settled

# ---------- live early-goal (Match in Play + push Telegram, sostituisce n8n) ----------

_LIVE_EG = {'ts': 0, 'matches': []}   # cache in-memory per il tab Match in Play
_SCANNER_CACHE = {}   # cache scanner nicchie: (minute, market) -> (ts, full_rows)
_LIVE_FG = {}                          # fixture_id -> minuto del 1o gol (fisso, cache)

DDL_LIVE_SEEN = """
CREATE TABLE IF NOT EXISTS live_alert_seen (
  fixture_id INTEGER PRIMARY KEY,
  ts INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER))
)
""".strip()

def _monitored_leagues():
    try:
        con = _sqlite3.connect(_LOCAL_DB)
        ids = [r[0] for r in con.execute("SELECT DISTINCT league_id FROM matches").fetchall()]
        con.close()
        return set(ids)
    except Exception:
        return set()

def _league_name(lid):
    try:
        con = _sqlite3.connect(_LOCAL_DB)
        r = con.execute("SELECT name FROM leagues WHERE id=?", (lid,)).fetchone()
        con.close()
        return r[0] if r else None
    except Exception:
        return None

def _send_telegram(text):
    import urllib.request as _u, urllib.parse as _up
    token = os.getenv('TELEGRAM_TOKEN', '')
    chat = os.getenv('TELEGRAM_CHAT_ID', '')
    if not token or not chat:
        return False
    try:
        body = _up.urlencode({'chat_id': chat, 'text': text, 'parse_mode': 'HTML',
                              'disable_web_page_preview': 'true'}).encode()
        req = _u.Request('https://api.telegram.org/bot' + token + '/sendMessage', data=body)
        with _u.urlopen(req, timeout=15) as r:
            return getattr(r, 'status', 200) == 200
    except Exception:
        return False

def _refresh_live_eg(send_telegram=False):
    monitored = _monitored_leagues()
    resp = (_af_get('/fixtures', {'live': 'all'}) or {}).get('response') or []
    out = []
    for f in resp:
        lg = f.get('league') or {}
        lid = lg.get('id')
        if lid not in monitored:
            continue
        g = f.get('goals') or {}
        gh, ga = g.get('home'), g.get('away')
        if (gh or 0) + (ga or 0) == 0:
            continue
        fid = (f.get('fixture') or {}).get('id')
        if fid is None:
            continue
        fgm = _LIVE_FG.get(fid)
        if fgm is None:
            ev = (_af_get('/fixtures/events', {'fixture': fid}) or {}).get('response') or []
            mins = [((e.get('time') or {}).get('elapsed') or 0) + ((e.get('time') or {}).get('extra') or 0)
                    for e in ev if (e.get('type') or '') == 'Goal']
            if mins:
                fgm = min(mins)
                _LIVE_FG[fid] = fgm   # cache solo il minuto reale
            else:
                fgm = 999             # eventi non ancora disponibili: NON cachare, ricontrolla al prossimo tick
        if fgm > 16:
            continue
        teams = f.get('teams') or {}
        out.append({
            'fixture_id': fid,
            'league_id': lid,
            'league': _league_name(lid) or lg.get('name'),
            'home': (teams.get('home') or {}).get('name'),
            'away': (teams.get('away') or {}).get('name'),
            'score': '%s-%s' % (gh, ga),
            'minute': ((f.get('fixture') or {}).get('status') or {}).get('elapsed'),
            'first_goal_min': fgm,
        })
    out.sort(key=lambda e: (e.get('minute') or 0))
    _LIVE_EG['ts'] = int(time.time())
    _LIVE_EG['matches'] = out
    sent = 0
    if send_telegram and out:
        try:
            _turso_execute(DDL_LIVE_SEEN)
            for e in out:
                if _turso_select_rows("SELECT 1 FROM live_alert_seen WHERE fixture_id=?", [e['fixture_id']]):
                    continue
                msg = ("⚽ <b>EARLY GOAL</b> — 1° gol al %d'\n%s\n<b>%s</b> vs <b>%s</b>\nRisultato: %s  (%s')" %
                       (e['first_goal_min'], e['league'], e['home'], e['away'], e['score'], e.get('minute') or '?'))
                if _send_telegram(msg):
                    _turso_execute("INSERT OR IGNORE INTO live_alert_seen (fixture_id) VALUES (?)", [e['fixture_id']])
                    sent += 1
        except Exception:
            pass
    return {'live_earlygoal': len(out), 'telegram_sent': sent}

def _ro_con(timeout=15.0):
    """Connessione al DB locale con busy_timeout: sotto contesa (es. archiver in
    scrittura) aspetta invece di dare subito 'database is locked'. Con WAL attivo
    i lock sono comunque rari."""
    con = _sqlite3.connect(_LOCAL_DB, timeout=timeout)
    con.row_factory = _sqlite3.Row
    try:
        con.execute("PRAGMA busy_timeout=15000")
    except Exception:
        pass
    return con


# === Gol tardivi: cache in-memory dei minuti-gol per match (build una volta) ===
_LATE_CACHE = {'built': False, 'rows': None, 'names': None}
_LATE_LOCK = _threading.Lock()   # una sola build anche con richieste concorrenti


def _late_build():
    """Costruisce UNA volta la lista (league_id, tuple(minuti_gol)) di tutti i
    match giocati, cosi' /api/late-goals conta i gol dopo un minuto qualsiasi
    senza ri-parsare le timeline ad ogni richiesta."""
    con = _ro_con()
    names = {r['id']: r['name'] for r in con.execute('SELECT id, name FROM leagues')}
    data = []
    for r in con.execute("SELECT league_id, goals_html, goals_text FROM matches WHERE ft_home IS NOT NULL"):
        tl = _bet_timeline(r['goals_html'], r['goals_text'])
        if not tl:
            continue
        mins = tuple(mn for (mn, aw) in tl if mn is not None)
        data.append((r['league_id'], mins))
    con.close()
    return data, names


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
        try:    limit   = int(request.args.get('limit', '30'))
        except: limit   = 30
        try:    max_age = int(request.args.get('max_age_days', '7'))
        except: max_age = 7
        res = settle_batch(limit=limit, max_age_days=max_age)
        return jsonify(res)

    @app.route('/api/settle-ml-picks-now')
    def api_settle_ml_picks_now():
        token = request.args.get('token', '')
        if not INGEST_TOKEN or token != INGEST_TOKEN:
            return jsonify({'error': 'forbidden'}), 403
        res = settle_ml_picks_orphan(limit=100, max_age_days=30)
        return jsonify(res)

    @app.route('/api/ml-picks-accuracy')
    def api_ml_picks_accuracy():
        """
        Accuratezza pick ML reali (WIN/LOSS) da ml_picks_log.
        Ritorna:
          by_market        - accuratezza globale per mercato
          by_league        - accuratezza globale per campionato
          by_market_league - accuratezza per coppia (mercato, campionato) [NUOVO]
          recent           - ultimi 30 pick
          pending          - pick non ancora settlati
          total_logged     - totale pick loggati
        """
        try:
            _ensure_picks_ddl()

            # --- Accuratezza per mercato (globale) ---
            by_market = _turso_select_rows(
                "SELECT market, "
                "COUNT(*) as total, "
                "SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) as wins, "
                "ROUND(100.0 * SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) / COUNT(*), 1) as accuracy_pct "
                "FROM ml_picks_log WHERE result IS NOT NULL "
                "GROUP BY market ORDER BY total DESC"
            )

            # --- Accuratezza per campionato (globale) ---
            by_league = _turso_select_rows(
                "SELECT league_name, MIN(country) as country, "
                "COUNT(*) as total, "
                "SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) as wins, "
                "ROUND(100.0 * SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) / COUNT(*), 1) as accuracy_pct "
                "FROM ml_picks_log WHERE result IS NOT NULL AND league_name IS NOT NULL "
                "GROUP BY league_name ORDER BY total DESC LIMIT 20"
            )

            # --- [NUOVO] Accuratezza per mercato ÃÂÃÂ campionato ---
            # Solo coppie con almeno 3 predizioni (per evitare rumore statistico)
            by_market_league = _turso_select_rows(
                "SELECT league_name, MIN(country) as country, market, "
                "COUNT(*) as total, "
                "SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) as wins, "
                "ROUND(100.0 * SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) / COUNT(*), 1) as accuracy_pct "
                "FROM ml_picks_log "
                "WHERE result IS NOT NULL AND league_name IS NOT NULL "
                "GROUP BY league_name, market "
                "HAVING COUNT(*) >= 3 "
                "ORDER BY accuracy_pct DESC, total DESC "
                "LIMIT 200"
            )

            # --- Ultimi 30 pick (con risultato) ---
            recent = _turso_select_rows(
                "SELECT fixture_id, league_name, home_team, away_team, market, "
                "model_prob, edge_pct, result, ft_home, ft_away, logged_at "
                "FROM ml_picks_log ORDER BY logged_at DESC LIMIT 30"
            )

            pending_r = _turso_select_rows("SELECT COUNT(*) as n FROM ml_picks_log WHERE result IS NULL")
            total_r   = _turso_select_rows("SELECT COUNT(*) as n FROM ml_picks_log")

            return jsonify({
                'by_market':        by_market        or [],
                'by_league':        by_league        or [],
                'by_market_league': by_market_league or [],   # <-- NUOVO
                'recent':           recent           or [],
                'pending':          (pending_r or [{}])[0].get('n', 0),
                'total_logged':     (total_r   or [{}])[0].get('n', 0),
            })
        except Exception as e:
            return jsonify({'error': str(e)[:300]}), 500

    @app.route('/api/ml-accuracy-trend')
    def api_ml_accuracy_trend():
        """
        [NUOVO] Curva di apprendimento ML: accuratezza per settimana.

        Parametri opzionali:
          ?market=over_2_5      -> filtra per mercato specifico
          ?league=Bundesliga    -> filtra per campionato
          ?mode=cumulative      -> accuratezza cumulativa (default: settimanale)

        Risposta:
          trend    - lista [{week, total, wins, accuracy_pct, cumulative_total, cumulative_accuracy_pct}]
          by_market_weekly - lista [{week, market, total, wins, accuracy_pct}] (solo se no filtro mercato)
          market   - mercato filtrato ('all' se nessuno)
          league   - campionato filtrato ('all' se nessuno)
        """
        try:
            _ensure_picks_ddl()
            market = request.args.get('market', '').strip()
            league = request.args.get('league', '').strip()

            # Costruisce WHERE clause dinamicamente
            where_parts = ["result IS NOT NULL"]
            args = []
            if market:
                where_parts.append("market = ?")
                args.append(market)
            if league:
                where_parts.append("league_name = ?")
                args.append(league)
            where = " AND ".join(where_parts)

            # --- Trend settimanale (per mercato se filtrato, altrimenti globale) ---
            trend_rows = _turso_select_rows(
                "SELECT strftime('%Y-%W', datetime(logged_at, 'unixepoch')) as week, "
                "COUNT(*) as total, "
                "SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) as wins, "
                "ROUND(100.0 * SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) / COUNT(*), 1) as accuracy_pct "
                "FROM ml_picks_log WHERE " + where +
                " GROUP BY week ORDER BY week ASC",
                args if args else None
            )

            # Aggiunge accuratezza cumulativa (rolling) a ogni settimana
            cum_total = 0
            cum_wins  = 0
            for row in trend_rows:
                cum_total += row.get('total', 0) or 0
                cum_wins  += row.get('wins', 0)  or 0
                row['cumulative_total']        = cum_total
                row['cumulative_wins']         = cum_wins
                row['cumulative_accuracy_pct'] = round(100.0 * cum_wins / cum_total, 1) if cum_total else None

            # --- Trend per singolo mercato (solo se non filtrato) ---
            by_market_weekly = []
            if not market:
                bm_where_parts = ["result IS NOT NULL"]
                bm_args = []
                if league:
                    bm_where_parts.append("league_name = ?")
                    bm_args.append(league)
                bm_where = " AND ".join(bm_where_parts)
                by_market_weekly = _turso_select_rows(
                    "SELECT strftime('%Y-%W', datetime(logged_at, 'unixepoch')) as week, "
                    "market, "
                    "COUNT(*) as total, "
                    "SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) as wins, "
                    "ROUND(100.0 * SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) / COUNT(*), 1) as accuracy_pct "
                    "FROM ml_picks_log WHERE " + bm_where +
                    " GROUP BY week, market ORDER BY week ASC, market",
                    bm_args if bm_args else None
                )

            # --- Riepilogo per campionato (solo se non filtrato per campionato) ---
            league_summary = []
            if not league:
                lg_where_parts = ["result IS NOT NULL", "league_name IS NOT NULL"]
                lg_args = []
                if market:
                    lg_where_parts.append("market = ?")
                    lg_args.append(market)
                lg_where = " AND ".join(lg_where_parts)
                league_summary = _turso_select_rows(
                    "SELECT league_name, MIN(country) as country, "
                    "COUNT(*) as total, "
                    "SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) as wins, "
                    "ROUND(100.0 * SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) / COUNT(*), 1) as accuracy_pct "
                    "FROM ml_picks_log WHERE " + lg_where +
                    " GROUP BY league_name HAVING COUNT(*) >= 3 ORDER BY total DESC LIMIT 30",
                    lg_args if lg_args else None
                )

            return jsonify({
                'trend':             trend_rows       or [],
                'by_market_weekly':  by_market_weekly or [],
                'league_summary':    league_summary   or [],
                'market':            market or 'all',
                'league':            league or 'all',
                'weeks_tracked':     len(trend_rows),
            })
        except Exception as e:
            return jsonify({'error': str(e)[:300]}), 500

    @app.route('/api/recent-matches')
    def api_recent_matches():
        """Ultimi N match con primo gol entro 16'. Filtra per league + country per evitare omonimi."""
        try:
            _ensure_ddl()
            limit   = min(int(request.args.get('limit', 30)), 100)
            league  = request.args.get('league', '').strip()
            country = request.args.get('country', '').strip()
            base_sel = (
                "SELECT fixture_id, league_name, country, league_id, "
                "home_team_name, away_team_name, "
                "ft_home, ft_away, ht_home, ht_away, first_goal_minute, date_utc, settled_ts "
                "FROM predictions_log WHERE ft_home IS NOT NULL "
                "AND first_goal_minute IS NOT NULL AND first_goal_minute <= 16"
            )
            if league and country:
                rows = _turso_select_rows(base_sel + " AND league_name = ? AND country = ? ORDER BY settled_ts DESC LIMIT ?", [league, country, limit])
            elif league:
                rows = _turso_select_rows(base_sel + " AND league_name = ? ORDER BY settled_ts DESC LIMIT ?", [league, limit])
            else:
                rows = _turso_select_rows(base_sel + " ORDER BY settled_ts DESC LIMIT ?", [limit])
            return jsonify({'matches': rows or [], 'n': len(rows or [])})
        except Exception as e:
            return jsonify({'error': str(e)[:300]}), 500

    @app.route('/api/predictions-log-stats')
    def predictions_log_stats():
        try:
            _ensure_ddl()
            tot      = _turso_select_rows("SELECT COUNT(*) AS n FROM predictions_log")
            settled  = _turso_select_rows("SELECT COUNT(*) AS n FROM predictions_log WHERE ft_home IS NOT NULL")
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
                'ok':               True,
                'total':            (tot      or [{}])[0].get('n', 0),
                'settled':          (settled  or [{}])[0].get('n', 0),
                'first_goal_le16':  (filtered or [{}])[0].get('n', 0),
                'by_league_top20':  by_league,
                'recent':           recent,
                'auto_state':       _SETTLE_STATE,
            })
        except Exception as e:
            return jsonify({'error': str(e)[:300]}), 500

    @app.route('/api/recent-predictions')
    def api_recent_predictions():
        """Ultime 30 predizioni (un pick per fixture, dedup) con esito, da ml_picks_log (Turso)."""
        try:
            _ensure_picks_ddl()
            rows = _turso_select_rows(
                "SELECT fixture_id, league_name, country, home_team, away_team, market, "
                "model_prob, bookie_quota, edge_pct, result, ft_home, ft_away, logged_at, settled_at "
                "FROM ml_picks_log WHERE id IN (SELECT MIN(id) FROM ml_picks_log GROUP BY fixture_id) "
                "ORDER BY logged_at DESC LIMIT 30"
            )
            return jsonify({'predictions': rows or []})
        except Exception as e:
            return jsonify({'error': str(e)[:300], 'predictions': []}), 500

    @app.route('/api/eg-ingest', methods=['POST'])
    def api_eg_ingest():
        """Ingest match early-goal da n8n. Accetta lista o singolo oggetto. Token-protected."""
        tok = request.args.get('token','') or request.headers.get('X-Ingest-Token','')
        if not INGEST_TOKEN or tok != INGEST_TOKEN:
            return jsonify({'error':'forbidden'}), 403
        try:
            _ensure_egl()
            data  = request.get_json(force=True) or {}
            items = data if isinstance(data, list) else [data]
            ins   = 0
            for item in items:
                _turso_execute(
                    "INSERT OR REPLACE INTO early_goal_log "
                    "(fixture_id,league_id,league_name,country,home_team,away_team,"
                    " ht_home,ht_away,ft_home,ft_away,first_goal_min,match_date,source) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    [item.get('fixture_id'), item.get('league_id'), item.get('league_name'),
                     item.get('country'),    item.get('home_team'), item.get('away_team'),
                     item.get('ht_home'),    item.get('ht_away'),
                     item.get('ft_home'),    item.get('ft_away'),
                     item.get('first_goal_min'), item.get('match_date'),
                     item.get('source','n8n')]
                )
                ins += 1
            return jsonify({'ok': True, 'inserted': ins})
        except Exception as e:
            return jsonify({'error': str(e)[:300]}), 500

    @app.route('/api/eg-matches')
    def api_eg_matches():
        """Ultimi N match con primo gol <=16' da early_goal_log (fonte n8n)."""
        try:
            _ensure_egl()
            limit   = min(int(request.args.get('limit', 30)), 100)
            league  = request.args.get('league', '').strip()
            country = request.args.get('country','').strip()
            base = ("SELECT fixture_id,league_id,league_name,country,"
                    "home_team,away_team,ht_home,ht_away,ft_home,ft_away,"
                    "first_goal_min,match_date,logged_ts "
                    "FROM early_goal_log WHERE first_goal_min IS NOT NULL AND first_goal_min<=16")
            if league and country:
                rows = _turso_select_rows(base+" AND league_name=? AND country=? ORDER BY match_date DESC LIMIT ?",[league,country,limit])
            elif league:
                rows = _turso_select_rows(base+" AND league_name=? ORDER BY match_date DESC LIMIT ?",[league,limit])
            else:
                rows = _turso_select_rows(base+" ORDER BY match_date DESC LIMIT ?",[limit])
            return jsonify({'matches': rows or [], 'n': len(rows or [])})
        except Exception as e:
            return jsonify({'error': str(e)[:300]}), 500

    @app.route('/api/eg-log-leagues')
    def api_eg_log_leagues():
        """Campionati distinti in early_goal_log con almeno 1 match early goal."""
        try:
            _ensure_egl()
            rows = _turso_select_rows(
                "SELECT league_name,country,league_id,COUNT(*) as n "
                "FROM early_goal_log WHERE first_goal_min IS NOT NULL AND first_goal_min<=16 "
                "GROUP BY league_name,country,league_id ORDER BY n DESC LIMIT 100"
            )
            return jsonify({'leagues': rows or []})
        except Exception as e:
            return jsonify({'error': str(e)[:300]}), 500


    # ---------- Google Sheets proxy (legge dal foglio n8n Goal Alert) ----------

    _GSHEETS_ID = '1OXax8q6A06bj_ZhDyj8P6bfecbELZ4dhrDXQQWnRexU'

    def _gsheets_csv(sheet_name):
        url = ('https://docs.google.com/spreadsheets/d/'
               + _GSHEETS_ID + '/gviz/tq?tqx=out:csv&sheet='
               + urllib.parse.quote(sheet_name))
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode('utf-8')

    def _parse_gs_csv(text):
        import csv, io
        reader = csv.reader(io.StringIO(text))
        rows = list(reader)
        if not rows:
            return []
        headers = [h.strip() for h in rows[0]]
        return [dict(zip(headers, row)) for row in rows[1:] if any(row)]

    @app.route('/api/gs-leagues')
    def api_gs_leagues():
        """Campionati distinti dal foglio Google Sheets _tracking (n8n Goal Alert)."""
        try:
            text = _gsheets_csv('_tracking')
            rows = _parse_gs_csv(text)
            seen = {}
            for row in rows:
                ln  = row.get('league_name','').strip()
                lid = row.get('league_id','').strip()
                if ln and ln not in seen:
                    seen[ln] = lid
            leagues = [{'league_name': k, 'league_id': v} for k, v in seen.items()]
            leagues.sort(key=lambda x: x['league_name'])
            return jsonify({'leagues': leagues})
        except Exception as e:
            return jsonify({'error': str(e)[:300]}), 500

    @app.route('/api/gs-matches')
    def api_gs_matches():
        """Ultimi 30 match da Google Sheets per campionato (n8n Goal Alert)."""
        try:
            league = request.args.get('league','').strip()
            if not league:
                return jsonify({'error': 'league parameter required'}), 400
            text = _gsheets_csv(league)
            rows = _parse_gs_csv(text)
            out = []
            for row in rows:
                # Primo gol: primo valore in minuti_gol (es. "5', 47'" -> 5)
                mg = row.get('minuti_gol','').strip()
                first_min = 99
                if mg:
                    import re as _re
                    m = _re.search(r'(\d+)', mg)
                    if m:
                        try: first_min = int(m.group(1))
                        except: pass
                out.append({
                    'data':          row.get('data',''),
                    'campionato':    row.get('campionato', league),
                    'casa':          row.get('casa',''),
                    'ospite':        row.get('ospite',''),
                    'ris_1t':        row.get('ris_1t',''),
                    'ris_2t':        row.get('ris_2t',''),
                    'ris_finale':    row.get('ris_finale',''),
                    'minuti_gol':    mg,
                    'first_goal_min': first_min,
                    'fixture_id':    row.get('fixture_id',''),
                })
            out.sort(key=lambda x: x['data'], reverse=True)
            return jsonify({'matches': out[:30], 'n': len(out)})
        except Exception as e:
            return jsonify({'error': str(e)[:300]}), 500

    # ---------- user_bets (paper trading) ----------

    @app.route('/api/bets/fixtures')
    def api_bets_fixtures():
        """Partite del giorno per un campionato (per agganciare il fixture_id)."""
        try:
            league = int(request.args.get('league') or 0)
            date   = (request.args.get('date') or '').strip()  # YYYY-MM-DD
            if not league or len(date) < 10:
                return jsonify({'fixtures': []})
            year = int(date[:4])
            seen = {}
            for season in (year, year - 1):
                try:
                    data = _af_get('/fixtures', params={'league': league, 'season': season, 'date': date})
                    for f in (data.get('response') or []):
                        fx = f.get('fixture') or {}
                        t  = f.get('teams') or {}
                        fid = fx.get('id')
                        if fid and fid not in seen:
                            seen[fid] = {
                                'fixture_id': fid,
                                'home': (t.get('home') or {}).get('name'),
                                'away': (t.get('away') or {}).get('name'),
                                'time': (fx.get('date') or '')[11:16],
                            }
                except Exception:
                    pass
            return jsonify({'fixtures': list(seen.values())})
        except Exception as e:
            return jsonify({'error': str(e)[:300], 'fixtures': []}), 500

    @app.route('/api/bets', methods=['POST'])
    def api_bets_add():
        """Registra una giocata (locked) + calcola edge vs base-rate del database."""
        try:
            _ensure_bets_ddl()
            d = request.get_json(force=True) or {}
            league_id = int(d.get('league_id') or 0)
            market    = (d.get('market') or '').strip()
            minute    = int(d.get('minute') or 0)
            sh        = int(d.get('score_home') or 0)
            sa        = int(d.get('score_away') or 0)
            odds      = float(d.get('odds') or 0)
            stake     = float(d.get('stake') or 1)
            if not league_id or not market or odds <= 1.0:
                return jsonify({'error': 'Dati mancanti: campionato, mercato e quota (>1) sono obbligatori'}), 400
            br, n = _bet_base_rate(league_id, market, minute, sh, sa)
            fair    = round(100.0 / br, 2) if (br and br > 0) else None
            implied = round(100.0 / odds, 1)
            edge    = round(br - implied, 1) if (br is not None) else None
            placed = 0 if str(d.get('placed', 1)).lower() in ('0', 'false', 'no') else 1
            _turso_execute(
                "INSERT INTO user_bets "
                "(fixture_id,league_id,league_name,home_team,away_team,match_date,market,minute,"
                " score_home,score_away,odds,stake,base_rate,fair_odds,edge_pct,sample_n,placed) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [d.get('fixture_id'), league_id, d.get('league_name'), d.get('home_team'),
                 d.get('away_team'), d.get('match_date'), market, minute, sh, sa, odds, stake,
                 br, fair, edge, n, placed]
            )
            return jsonify({'ok': True, 'base_rate': br, 'sample': n,
                            'fair_odds': fair, 'implied_pct': implied, 'edge_pct': edge})
        except Exception as e:
            return jsonify({'error': str(e)[:300]}), 500

    @app.route('/api/bets')
    def api_bets_list():
        """Lista giocate + statistiche aggregate (edge, ROI, win%, calibrazione)."""
        try:
            _ensure_bets_ddl()
            rows = _turso_select_rows("SELECT * FROM user_bets ORDER BY created_ts DESC LIMIT 300")
            def _is_placed(r):
                return (r.get('placed') is None) or (r.get('placed') == 1)
            real  = [r for r in rows if _is_placed(r)]
            nobet = [r for r in rows if not _is_placed(r)]
            rset = [r for r in real  if r.get('result') in ('WIN', 'LOSS')]
            nset = [r for r in nobet if r.get('result') in ('WIN', 'LOSS')]
            n = len(rset)
            wins = sum(1 for r in rset if r.get('result') == 'WIN')
            staked = sum((r.get('stake') or 0) for r in rset)
            pnl = sum((r.get('pnl') or 0) for r in rset)
            edges = [r.get('edge_pct') for r in real if r.get('edge_pct') is not None]
            base_settled = [r.get('base_rate') for r in rset if r.get('base_rate') is not None]
            nwins = sum(1 for r in nset if r.get('result') == 'WIN')
            nhyp = sum((r.get('pnl') or 0) for r in nset)
            stats = {
                'total': len(real),
                'settled': n,
                'pending': len(real) - n,
                'win_rate': round(100.0 * wins / n, 1) if n else None,
                'roi_pct': round(100.0 * pnl / staked, 1) if staked else None,
                'pnl': round(pnl, 2),
                'staked': round(staked, 2),
                'avg_edge': round(sum(edges) / len(edges), 1) if edges else None,
                'avg_base_rate': round(sum(base_settled) / len(base_settled), 1) if base_settled else None,
                'nobet_total': len(nobet),
                'nobet_settled': len(nset),
                'nobet_win_rate': round(100.0 * nwins / len(nset), 1) if nset else None,
                'nobet_hyp_pnl': round(nhyp, 2),
            }
            return jsonify({'bets': rows, 'stats': stats})
        except Exception as e:
            return jsonify({'error': str(e)[:300]}), 500

    @app.route('/api/bets/settle')
    def api_bets_settle():
        """Settla le giocate pending (via fixture_id + API-Football)."""
        try:
            try:    lim = int(request.args.get('limit', '80'))
            except: lim = 80
            n = _settle_user_bets(limit=lim)
            return jsonify({'ok': True, 'settled': n})
        except Exception as e:
            return jsonify({'error': str(e)[:300]}), 500

    @app.route('/api/bets/<int:bet_id>', methods=['DELETE'])
    def api_bets_delete(bet_id):
        try:
            _ensure_bets_ddl()
            _turso_execute("DELETE FROM user_bets WHERE id=?", [bet_id])
            return jsonify({'ok': True})
        except Exception as e:
            return jsonify({'error': str(e)[:300]}), 500

    @app.route('/api/bets/preview')
    def api_bets_preview():
        """Anteprima edge SENZA registrare nulla."""
        try:
            league_id = int(request.args.get('league_id') or 0)
            market    = (request.args.get('market') or '').strip()
            minute    = int(request.args.get('minute') or 0)
            sh        = int(request.args.get('score_home') or 0)
            sa        = int(request.args.get('score_away') or 0)
            odds      = float(request.args.get('odds') or 0)
            if not league_id or not market:
                return jsonify({'error': 'campionato e mercato richiesti'}), 400
            br, n = _bet_base_rate(league_id, market, minute, sh, sa)
            fair    = round(100.0 / br, 2) if (br and br > 0) else None
            implied = round(100.0 / odds, 1) if odds > 1 else None
            edge    = round(br - implied, 1) if (br is not None and implied is not None) else None
            return jsonify({'base_rate': br, 'sample': n, 'fair_odds': fair,
                            'implied_pct': implied, 'edge_pct': edge})
        except Exception as e:
            return jsonify({'error': str(e)[:300]}), 500

    @app.route('/api/bets/<int:bet_id>/result', methods=['POST'])
    def api_bets_result(bet_id):
        """Imposta a mano il risultato finale di una giocata -> settla."""
        try:
            _ensure_bets_ddl()
            d = request.get_json(force=True) or {}
            fh = int(d.get('ft_home'))
            fa = int(d.get('ft_away'))
            rows = _turso_select_rows("SELECT market, odds, stake FROM user_bets WHERE id=?", [bet_id])
            if not rows:
                return jsonify({'error': 'giocata non trovata'}), 404
            b = rows[0]
            w = _bet_market_win(b.get('market'), fh, fa)
            if w is None:
                return jsonify({'error': 'mercato non valutabile'}), 400
            stake = b.get('stake') or 1.0
            odds  = b.get('odds') or 0.0
            pnl = stake * (odds - 1.0) if w else -stake
            _turso_execute(
                "UPDATE user_bets SET status='settled', ft_home=?, ft_away=?, result=?, pnl=?, settled_ts=? WHERE id=?",
                [fh, fa, ('WIN' if w else 'LOSS'), round(pnl, 2), int(time.time()), bet_id]
            )
            return jsonify({'ok': True, 'result': ('WIN' if w else 'LOSS')})
        except Exception as e:
            return jsonify({'error': str(e)[:300]}), 500

    @app.route('/api/live-earlygoal-tick')
    def api_live_eg_tick():
        """Cron (ogni 5 min): aggiorna la lista live early-goal + invia Telegram per le nuove."""
        token = request.args.get('token', '') or request.headers.get('X-Tick-Token', '')
        exp = os.getenv('TICK_AUTH_TOKEN', '')
        if not exp or token != exp:
            return jsonify({'error': 'unauthorized'}), 401
        try:
            return jsonify(_refresh_live_eg(send_telegram=True))
        except Exception as e:
            return jsonify({'error': str(e)[:300]}), 500

    @app.route('/api/live-earlygoal')
    def api_live_eg():
        """Lista delle partite in corso col 1o gol <=16' (per il tab Match in Play).
        Legge la cache; la rinfresca se piu' vecchia di 4 min (senza Telegram)."""
        try:
            if time.time() - _LIVE_EG.get('ts', 0) > 240:
                _refresh_live_eg(send_telegram=False)
            return jsonify({'matches': _LIVE_EG.get('matches', []), 'updated_ts': _LIVE_EG.get('ts', 0)})
        except Exception as e:
            return jsonify({'error': str(e)[:300], 'matches': []}), 500

    @app.route('/api/segments')
    def api_segments():
        """Base rate di un mercato, per campionato, spezzata per STATO al minuto
        (es. 1-0, 0-1 = anche chi ha segnato il 1o gol), con n e quota equa.
        Serve a trovare le nicchie piu' affidabili."""
        try:
            lid = int(request.args.get('league') or 0)
            minute = int(request.args.get('minute') or 65)
            market = (request.args.get('market') or 'over_1_5').strip()
            first = (request.args.get('first') or '').strip()  # '', 'home', 'away' = chi ha segnato il 1o gol
            period = (request.args.get('period') or '').strip().upper()
            if not lid:
                return jsonify({'error': 'parametro league richiesto'}), 400
            # periodo dal parametro (FT/HT/ST); retro-compat: dedotto dal suffisso _ht/_st
            if period not in ('FT', 'HT', 'ST'):
                period = 'HT' if market.endswith('_ht') else ('ST' if market.endswith('_st') else 'FT')
            base = market[:-3] if (market.endswith('_ht') or market.endswith('_st')) else market
            OVER = {'over_0_5': 1, 'over_1_5': 2, 'over_2_5': 3, 'over_3_5': 4, 'over_4_5': 5}
            UNDER = {'under_1_5': 2, 'under_2_5': 3, 'under_3_5': 4, 'under_4_5': 5}
            con = _sqlite3.connect(_LOCAL_DB)
            con.row_factory = _sqlite3.Row
            rows = con.execute(
                "SELECT goals_html, goals_text, ft_home, ft_away, ht_home, ht_away, st_home, st_away FROM matches WHERE league_id=?",
                (lid,)).fetchall()
            con.close()
            groups = {}
            for r in rows:
                fh, fa = r['ft_home'], r['ft_away']
                if fh is None or fa is None:
                    continue
                # punteggio finale del periodo scelto
                if period == 'HT':
                    pa, pb = r['ht_home'], r['ht_away']
                elif period == 'ST':
                    pa, pb = r['st_home'], r['st_away']
                else:
                    pa, pb = fh, fa
                if pa is None or pb is None:
                    continue
                tl = _bet_timeline(r['goals_html'], r['goals_text'])
                if first:
                    if not tl or tl[0][1] is None:
                        continue
                    if (first == 'away') != bool(tl[0][1]):  # tl[0] = 1o gol; True = ospite
                        continue
                if any(aw is None for (mn, aw) in tl if mn <= minute):
                    continue
                # stato al minuto (scoreline pieno) -> chiave di raggruppamento
                hM = sum(1 for (mn, aw) in tl if mn <= minute and aw is False)
                aM = sum(1 for (mn, aw) in tl if mn <= minute and aw is True)
                # gol del PERIODO gia' segnati entro il minuto -> per lo skip "gia' deciso"
                if period == 'HT':
                    lim = min(minute, 45)
                    psa = sum(1 for (mn, aw) in tl if mn <= lim and aw is False)
                    psb = sum(1 for (mn, aw) in tl if mn <= lim and aw is True)
                elif period == 'ST':
                    psa = sum(1 for (mn, aw) in tl if 45 < mn <= minute and aw is False)
                    psb = sum(1 for (mn, aw) in tl if 45 < mn <= minute and aw is True)
                else:
                    psa, psb = hM, aM
                sofar = psa + psb
                ptot = pa + pb
                if base in OVER:
                    thr = OVER[base]
                    if sofar >= thr:
                        continue
                    win = ptot >= thr
                elif base in UNDER:
                    thr = UNDER[base]
                    if sofar >= thr:
                        continue
                    win = ptot < thr
                elif base == 'gol_casa':
                    if psa > 0:
                        continue
                    win = pa > 0
                elif base == 'gol_ospite':
                    if psb > 0:
                        continue
                    win = pb > 0
                elif base == 'btts_si':
                    if psa > 0 and psb > 0:
                        continue
                    win = (pa > 0 and pb > 0)
                elif base == 'btts_no':
                    if psa > 0 and psb > 0:
                        continue
                    win = not (pa > 0 and pb > 0)
                elif base in ('1', 'X', '2'):
                    win = {'1': pa > pb, 'X': pa == pb, '2': pb > pa}[base]
                else:
                    continue
                st = "%d-%d" % (hM, aM)
                g = groups.setdefault(st, [0, 0])
                g[0] += 1
                if win:
                    g[1] += 1
            out = []
            for st, (n, w) in groups.items():
                br = round(100.0 * w / n, 1) if n else 0.0
                out.append({'state': st, 'n': n, 'wins': w, 'base_rate': br,
                            'fair_odds': (round(100.0 / br, 2) if br > 0 else None)})
            out.sort(key=lambda x: -x['n'])
            return jsonify({'league_id': lid, 'minute': minute, 'market': market, 'period': period,
                            'first': first or 'qualsiasi', 'segments': out})
        except Exception as e:
            return jsonify({'error': str(e)[:300]}), 500

    @app.route('/api/comebacks')
    def api_comebacks():
        """Traiettoria del vantaggio: per i match early-goal di una lega, quante volte
        chi ha segnato per primo (andando avanti) ha poi SUBITO una rimonta (e' finito
        sotto) e come e' finita: ri-vinta / pareggiata / persa. Filtrabile per chi ha
        segnato il 1o gol (first=home|away). Serve la sequenza gol completa."""
        try:
            lid = int(request.args.get('league') or 0)
            first = (request.args.get('first') or '').strip()
            if not lid:
                return jsonify({'error': 'parametro league richiesto'}), 400
            con = _sqlite3.connect(_LOCAL_DB)
            con.row_factory = _sqlite3.Row
            rows = con.execute(
                "SELECT date_str, home_team, away_team, goals_html, goals_text, ft_home, ft_away "
                "FROM matches WHERE league_id=? ORDER BY sort_date DESC, time_str DESC",
                (lid,)).fetchall()
            con.close()
            cat = {'held_won': 0, 'never_behind_draw': 0,
                   'overturned_won': 0, 'overturned_draw': 0, 'overturned_lost': 0}
            peg = {'tot': 0, '1': 0, 'X': 0, '2': 0}   # vantaggio annullato (parita' >0-0), split esito finale
            peg2 = {'tot': 0, '1': 0, 'X': 0, '2': 0}  # ...partendo da un doppio vantaggio (>=2)
            peg_list = []                              # ultime 30 a vantaggio annullato (piu' recenti), esito 1/X/2
            peg_signs = []                             # esito (1/X/2) di TUTTE le gare a vantaggio annullato, per le strisce
            total = 0
            for r in rows:
                fh, fa = r['ft_home'], r['ft_away']
                if fh is None or fa is None:
                    continue
                tl = _bet_timeline(r['goals_html'], r['goals_text'])
                if not tl or any(aw is None for (mn, aw) in tl):
                    continue  # serve la sequenza completa per la traiettoria
                fa_first = tl[0][1]  # True = ospite ha segnato per primo
                if first == 'home' and fa_first:
                    continue
                if first == 'away' and not fa_first:
                    continue
                h = a = 0
                behind = False
                had_lead = had_two = tie_after_lead = tie_after_two = False
                for (mn, aw) in tl:
                    if aw:
                        a += 1
                    else:
                        h += 1
                    if fa_first and h > a:          # 1o marcatore = ospite -> sotto se casa supera
                        behind = True
                    elif (not fa_first) and a > h:  # 1o marcatore = casa -> sotto se ospite supera
                        behind = True
                    d = h - a
                    if d != 0:                      # qualcuno e' in vantaggio
                        had_lead = True
                        if abs(d) >= 2:
                            had_two = True
                    elif (h + a) > 0:               # parita' non 0-0 -> vantaggio annullato
                        if had_lead:
                            tie_after_lead = True
                        if had_two:
                            tie_after_two = True
                if fa_first:
                    fs_win, draw = fa > fh, fa == fh
                else:
                    fs_win, draw = fh > fa, fh == fa
                total += 1
                if not behind:
                    cat['held_won' if fs_win else 'never_behind_draw'] += 1
                else:
                    cat['overturned_won' if fs_win else ('overturned_draw' if draw else 'overturned_lost')] += 1
                sign = '1' if fh > fa else ('2' if fa > fh else 'X')
                if tie_after_lead:
                    peg['tot'] += 1
                    peg[sign] += 1
                if tie_after_two:
                    peg2['tot'] += 1
                    peg2[sign] += 1
                if tie_after_lead:
                    peg_signs.append(sign)             # in ordine DESC (piu' recente prima)
                    if len(peg_list) < 30:
                        hh = aa = 0
                        prog = []
                        for (mn, aw) in tl:
                            if aw:
                                aa += 1
                            else:
                                hh += 1
                            prog.append('%d-%d' % (hh, aa))
                        peg_list.append({
                            'date': r['date_str'], 'home': r['home_team'],
                            'away': r['away_team'], 'final': '%d-%d' % (fh, fa),
                            'sign': sign, 'two': tie_after_two, 'seq': prog})
            def pct(n):
                return round(100.0 * n / total, 1) if total else 0.0
            def pct_of(n, den):
                return round(100.0 * n / den, 1) if den else 0.0
            overturned = cat['overturned_won'] + cat['overturned_draw'] + cat['overturned_lost']
            recovered = cat['overturned_won'] + cat['overturned_draw']

            def _max_run(signs, target):
                best = cur = 0
                for s in signs:
                    cur = cur + 1 if s == target else 0
                    if cur > best:
                        best = cur
                return best

            def _cur_run(signs, target):     # signs[0] = piu' recente
                n = 0
                for s in signs:
                    if s == target:
                        n += 1
                    else:
                        break
                return n
            peg_streak = {
                'x_cur': _cur_run(peg_signs, 'X'), 'x_max': _max_run(peg_signs, 'X'),
                'h_max': _max_run(peg_signs, '1'), 'a_max': _max_run(peg_signs, '2'),
                'n': len(peg_signs),
            }
            return jsonify({
                'league_id': lid, 'first': first or 'qualsiasi', 'total': total,
                'categories': cat, 'pct': {k: pct(v) for k, v in cat.items()},
                'overturned': overturned, 'overturned_pct': pct(overturned),
                'recovered': recovered, 'recovered_pct': pct(recovered),
                'pegged': {
                    'tot': peg['tot'], 'tot_pct': pct(peg['tot']),
                    'home': peg['1'], 'draw': peg['X'], 'away': peg['2'],
                    'home_pct': pct_of(peg['1'], peg['tot']),
                    'draw_pct': pct_of(peg['X'], peg['tot']),
                    'away_pct': pct_of(peg['2'], peg['tot']),
                    'decisive': peg['1'] + peg['2'],
                    'decisive_pct': pct_of(peg['1'] + peg['2'], peg['tot']),
                },
                'pegged_two': {
                    'tot': peg2['tot'], 'tot_pct': pct(peg2['tot']),
                    'home': peg2['1'], 'draw': peg2['X'], 'away': peg2['2'],
                    'decisive': peg2['1'] + peg2['2'],
                    'decisive_pct': pct_of(peg2['1'] + peg2['2'], peg2['tot']),
                },
                'pegged_list': peg_list,
                'pegged_streak': peg_streak,
            })
        except Exception as e:
            return jsonify({'error': str(e)[:300]}), 500

    @app.route('/api/late-goals')
    def api_late_goals():
        """Per (minuto): istogramma dei gol segnati DAL minuto in poi (>=minuto)
        per ogni campionato -> il frontend deriva sia >=N sia =N gol. Universo:
        la nicchia early-goal (1o gol <=16')."""
        try:
            minute = int(request.args.get('minute') or 65)
            minute = max(0, min(120, minute))
            if not _LATE_CACHE['built']:
                with _LATE_LOCK:                       # evita build concorrenti multiple
                    if not _LATE_CACHE['built']:
                        _LATE_CACHE['rows'], _LATE_CACHE['names'] = _late_build()
                        _LATE_CACHE['built'] = True
            rows = _LATE_CACHE['rows']
            names = _LATE_CACHE['names']
            CAP = 6
            agg = {}
            tot_hist = [0] * (CAP + 1)
            tot_n = 0
            for (lid, mins) in rows:
                late = 0
                for mn in mins:
                    if mn >= minute:
                        late += 1
                idx = late if late < CAP else CAP
                a = agg.get(lid)
                if a is None:
                    a = agg[lid] = {'n': 0, 'hist': [0] * (CAP + 1)}
                a['n'] += 1
                a['hist'][idx] += 1
                tot_n += 1
                tot_hist[idx] += 1
            out = [{'league_id': lid, 'league': names.get(lid, str(lid)),
                    'n': a['n'], 'hist': a['hist']} for lid, a in agg.items()]
            out.sort(key=lambda x: -x['n'])
            return jsonify({'minute': minute, 'total_n': tot_n,
                            'total_hist': tot_hist, 'cap': CAP, 'leagues': out})
        except Exception as e:
            return jsonify({'error': str(e)[:300]}), 500

    @app.route('/api/late-goals-matches')
    def api_late_goals_matches():
        """Ultime N partite di UN campionato: parziale al minuto scelto, finale e
        gol segnati DAL minuto in poi (>=minuto). Filtrabile per chi ha segnato
        il 1o gol (first=home|away)."""
        try:
            lid = int(request.args.get('league') or 0)
            minute = int(request.args.get('minute') or 65)
            minute = max(0, min(120, minute))
            first = (request.args.get('first') or '').strip()
            limit = min(int(request.args.get('limit') or 30), 100)
            if not lid:
                return jsonify({'error': 'parametro league richiesto'}), 400
            con = _ro_con()
            sql = ("SELECT date_str, home_team, away_team, ft_home, ft_away, "
                   "first_goal_team, goals_html, goals_text FROM matches "
                   "WHERE league_id=? AND ft_home IS NOT NULL")
            params = [lid]
            if first in ('home', 'away'):
                sql += " AND first_goal_team = ?"
                params.append(first)
            sql += " ORDER BY sort_date DESC, time_str DESC LIMIT ?"
            params.append(limit)
            rows = con.execute(sql, params).fetchall()
            con.close()
            out = []
            for r in rows:
                tl = _bet_timeline(r['goals_html'], r['goals_text'])
                pre_h = pre_a = 0
                late_mins = []
                for (mn, aw) in tl:
                    if mn is None:
                        continue
                    if mn < minute:
                        if aw:
                            pre_a += 1
                        else:
                            pre_h += 1
                    else:
                        late_mins.append(mn)
                out.append({
                    'date': r['date_str'], 'home': r['home_team'], 'away': r['away_team'],
                    'pre': '%d-%d' % (pre_h, pre_a),
                    'final': '%d-%d' % (r['ft_home'], r['ft_away']),
                    'late': len(late_mins), 'late_mins': late_mins,
                    'fg_team': r['first_goal_team'],
                })
            return jsonify({'league_id': lid, 'minute': minute,
                            'first': first or 'qualsiasi', 'n': len(out), 'matches': out})
        except Exception as e:
            return jsonify({'error': str(e)[:300]}), 500

    @app.route('/api/live-read')
    def api_live_read():
        """Cruscotto live: per (campionato, minuto, punteggio attuale) restituisce
        TUTTI i mercati insieme con base rate storico + quota equa + n, cosi' in
        diretta vedi subito dove c'e' valore. Salta i mercati gia' decisi dallo
        stato corrente (es. Over 1.5 se sono gia' 2 gol)."""
        try:
            lid = int(request.args.get('league') or 0)
            minute = int(request.args.get('minute') or 65)
            sh = int(request.args.get('sh') or 0)
            sa = int(request.args.get('sa') or 0)
            if not lid:
                return jsonify({'error': 'parametro league richiesto'}), 400
            total = sh + sa
            markets = [('over_1_5', 'Over 1.5'), ('over_2_5', 'Over 2.5'),
                       ('over_3_5', 'Over 3.5'), ('over_4_5', 'Over 4.5'),
                       ('under_1_5', 'Under 1.5'), ('under_2_5', 'Under 2.5'),
                       ('under_3_5', 'Under 3.5'), ('under_4_5', 'Under 4.5'),
                       ('over_1_5_ht', 'Over 1.5 HT'), ('over_2_5_ht', 'Over 2.5 HT'),
                       ('under_1_5_ht', 'Under 1.5 HT'), ('under_2_5_ht', 'Under 2.5 HT'),
                       ('btts_si', 'BTTS Si'), ('btts_no', 'BTTS No'),
                       ('1', '1 (Casa)'), ('X', 'X (Pari)'), ('2', '2 (Ospite)')]
            OVER_THR = {'over_1_5': 2, 'over_2_5': 3, 'over_3_5': 4, 'over_4_5': 5,
                        'under_1_5': 2, 'under_2_5': 3, 'under_3_5': 4, 'under_4_5': 5,
                        'over_1_5_ht': 2, 'over_2_5_ht': 3, 'under_1_5_ht': 2, 'under_2_5_ht': 3}
            out = []
            for mk, label in markets:
                if mk.endswith('_ht') and minute >= 45:
                    continue
                if mk in OVER_THR and total >= OVER_THR[mk]:
                    continue
                if mk in ('btts_si', 'btts_no') and sh > 0 and sa > 0:
                    continue
                br, n = _bet_base_rate(lid, mk, minute, sh, sa)
                if not n:
                    continue
                out.append({'market': mk, 'label': label, 'base_rate': br, 'n': n,
                            'fair_odds': (round(100.0 / br, 2) if br > 0 else None)})
            out.sort(key=lambda x: -x['base_rate'])
            return jsonify({'league_id': lid, 'minute': minute, 'state': '%d-%d' % (sh, sa),
                            'total': total, 'markets': out})
        except Exception as e:
            return jsonify({'error': str(e)[:300]}), 500

    @app.route('/api/live-matches')  # nudge redeploy 2026-07-29
    def api_live_matches():
        """Per (campionato, minuto, punteggio attuale): restituisce l'ELENCO delle
        partite storiche di quella lega che erano in quello stato a quel minuto (come
        sono finite) + le % di uscita dei mercati da li in poi. Una sola scansione."""
        try:
            lid = int(request.args.get('league') or 0)
            minute = int(request.args.get('minute') or 65)
            sh = int(request.args.get('sh') or 0)
            sa = int(request.args.get('sa') or 0)
            if not lid:
                return jsonify({'error': 'parametro league richiesto'}), 400
            total = sh + sa
            markets = [('over_1_5', 'Over 1.5', 2), ('over_2_5', 'Over 2.5', 3),
                       ('over_3_5', 'Over 3.5', 4), ('over_4_5', 'Over 4.5', 5),
                       ('under_1_5', 'Under 1.5', 2), ('under_2_5', 'Under 2.5', 3),
                       ('under_3_5', 'Under 3.5', 4), ('under_4_5', 'Under 4.5', 5),
                       ('btts_si', 'BTTS Si', None), ('btts_no', 'BTTS No', None),
                       ('1', '1 (Casa)', None), ('X', 'X (Pareggio)', None), ('2', '2 (Ospite)', None)]
            cnt = {mk: [0, 0] for mk, _, _ in markets}
            matched = []
            n_matched = 0
            con = _sqlite3.connect(_LOCAL_DB)
            con.row_factory = _sqlite3.Row
            rows = con.execute(
                "SELECT date_str, home_team, away_team, ft_home, ft_away, goals_html, goals_text "
                "FROM matches WHERE league_id=? ORDER BY sort_date DESC, time_str DESC", (lid,)).fetchall()
            con.close()
            for r in rows:
                fh, fa = r['ft_home'], r['ft_away']
                if fh is None or fa is None:
                    continue
                tl = _bet_timeline(r['goals_html'], r['goals_text'])
                if any(aw is None for (mn, aw) in tl if mn <= minute):
                    continue
                hM = sum(1 for (mn, aw) in tl if mn <= minute and aw is False)
                aM = sum(1 for (mn, aw) in tl if mn <= minute and aw is True)
                if hM != sh or aM != sa:
                    continue
                n_matched += 1
                tot_ft = fh + fa
                btts = (fh > 0 and fa > 0)
                for mk, label, thr in markets:
                    if thr is not None:
                        if total >= thr:
                            continue
                        cnt[mk][0] += 1
                        if mk.startswith('over_'):
                            if tot_ft >= thr:
                                cnt[mk][1] += 1
                        else:
                            if tot_ft < thr:
                                cnt[mk][1] += 1
                    elif mk in ('btts_si', 'btts_no'):
                        if sh > 0 and sa > 0:
                            continue
                        cnt[mk][0] += 1
                        if (mk == 'btts_si' and btts) or (mk == 'btts_no' and not btts):
                            cnt[mk][1] += 1
                    else:
                        cnt[mk][0] += 1
                        if {'1': fh > fa, 'X': fh == fa, '2': fa > fh}[mk]:
                            cnt[mk][1] += 1
                if len(matched) < 30:
                    fgm = tl[0][0] if tl else None
                    fgt = (('away' if tl[0][1] else 'home') if (tl and tl[0][1] is not None) else None)
                    matched.append({'date': r['date_str'], 'home': r['home_team'], 'away': r['away_team'],
                                    'final': '%d-%d' % (fh, fa), 'fg_min': fgm, 'fg_team': fgt})
            out_m = []
            for mk, label, thr in markets:
                n, w = cnt[mk]
                if not n:
                    continue
                br = round(100.0 * w / n, 1)
                out_m.append({'market': mk, 'label': label, 'base_rate': br, 'n': n,
                              'fair_odds': (round(100.0 / br, 2) if br > 0 else None)})
            out_m.sort(key=lambda x: -x['base_rate'])
            return jsonify({'league_id': lid, 'minute': minute, 'state': '%d-%d' % (sh, sa),
                            'n': n_matched, 'matches': matched, 'markets': out_m})
        except Exception as e:
            return jsonify({'error': str(e)[:300]}), 500

    @app.route('/api/scanner')
    def api_scanner():
        """Scanner nicchie: per un minuto + mercato, scorre TUTTI i campionati e
        classifica le combinazioni (campionato x stato) col base rate piu' alto,
        cosi' vedi DOVE storicamente il niche rende di piu'. Filtrabile per stato/n.
        Cache 10 min per (minuto, mercato): i filtri stato/n sono istantanei."""
        try:
            minute = int(request.args.get('minute') or 65)
            market = (request.args.get('market') or 'over_1_5').strip()
            state_filter = (request.args.get('state') or '').strip()
            min_n = int(request.args.get('min_n') or 50)
            ckey = (minute, market)
            now = time.time()
            cached = _SCANNER_CACHE.get(ckey)
            if cached and now - cached[0] < 600:
                full = cached[1]
            else:
                OVER = {'over_0_5': 1, 'over_1_5': 2, 'over_2_5': 3, 'over_3_5': 4, 'over_4_5': 5}
                UNDER = {'under_1_5': 2, 'under_2_5': 3, 'under_3_5': 4, 'under_4_5': 5}
                HT = {'over_1_5_ht': 2, 'over_2_5_ht': 3, 'under_1_5_ht': 2, 'under_2_5_ht': 3}
                con = _sqlite3.connect(_LOCAL_DB)
                con.row_factory = _sqlite3.Row
                names = {r['id']: r['name'] for r in con.execute("SELECT id, name FROM leagues")}
                groups = {}
                # streaming: una riga alla volta dal cursore (niente .fetchall(): non carica le 93k partite in RAM insieme)
                for r in con.execute("SELECT league_id, goals_html, goals_text, ft_home, ft_away FROM matches"):
                    fh, fa = r['ft_home'], r['ft_away']
                    if fh is None or fa is None:
                        continue
                    tl = _bet_timeline(r['goals_html'], r['goals_text'])
                    if any(aw is None for (mn, aw) in tl if mn <= minute):
                        continue
                    hM = sum(1 for (mn, aw) in tl if mn <= minute and aw is False)
                    aM = sum(1 for (mn, aw) in tl if mn <= minute and aw is True)
                    totM = hM + aM
                    if market in OVER:
                        thr = OVER[market]
                        if totM >= thr:
                            continue
                        win = (fh + fa) >= thr
                    elif market in UNDER:
                        thr = UNDER[market]
                        if totM >= thr:
                            continue
                        win = (fh + fa) < thr
                    elif market in HT:
                        thr = HT[market]
                        if totM >= thr:
                            continue
                        htM = sum(1 for (mn, aw) in tl if mn <= 45)
                        win = (htM >= thr) if market.startswith('over_') else (htM < thr)
                    elif market == 'btts_si':
                        if hM > 0 and aM > 0:
                            continue
                        win = (fh > 0 and fa > 0)
                    elif market == 'btts_no':
                        if hM > 0 and aM > 0:
                            continue
                        win = not (fh > 0 and fa > 0)
                    elif market in ('1', 'X', '2'):
                        win = {'1': fh > fa, 'X': fh == fa, '2': fa > fh}[market]
                    else:
                        continue
                    key = (r['league_id'], "%d-%d" % (hM, aM))
                    g = groups.setdefault(key, [0, 0])
                    g[0] += 1
                    if win:
                        g[1] += 1
                con.close()
                full = []
                for (lid, st), (n, w) in groups.items():
                    br = round(100.0 * w / n, 1) if n else 0.0
                    full.append({'league_id': lid, 'league': names.get(lid, str(lid)),
                                 'state': st, 'n': n, 'base_rate': br,
                                 'fair_odds': (round(100.0 / br, 2) if br > 0 else None)})
                _SCANNER_CACHE[ckey] = (now, full)
                # eviction: libera le voci scadute (>600s) e tieni al massimo le 12 piu' recenti
                for _k in [k for k, v in list(_SCANNER_CACHE.items()) if now - v[0] > 600]:
                    _SCANNER_CACHE.pop(_k, None)
                if len(_SCANNER_CACHE) > 12:
                    for _k in sorted(_SCANNER_CACHE, key=lambda k: _SCANNER_CACHE[k][0])[:len(_SCANNER_CACHE) - 12]:
                        _SCANNER_CACHE.pop(_k, None)
            out = [x for x in full if x['n'] >= min_n and (not state_filter or x['state'] == state_filter)]
            out.sort(key=lambda x: -x['base_rate'])
            out = out[:60]
            return jsonify({'minute': minute, 'market': market,
                            'state': state_filter or 'tutti', 'min_n': min_n, 'rows': out})
        except Exception as e:
            return jsonify({'error': str(e)[:300]}), 500

    @app.route('/api/market-trend')
    def api_market_trend():
        """Andamento storico di un mercato per un campionato: base rate PER STAGIONE
        (media grezza sul totale delle partite early-goal della lega, non condizionata a
        minuto/stato) + media complessiva. Per il grafico a barre per stagione."""
        try:
            lid = int(request.args.get('league') or 0)
            market = (request.args.get('market') or 'over_2_5').strip()
            if not lid:
                return jsonify({'error': 'parametro league richiesto'}), 400
            # periodo dal parametro (FT/HT/ST); retro-compat: dedotto dal suffisso del market
            period = (request.args.get('period') or '').strip().upper()
            if period not in ('FT', 'HT', 'ST'):
                period = 'HT' if market.endswith('_ht') else ('ST' if market.endswith('_st') else 'FT')
            # market base: togli l'eventuale suffisso di periodo -> stessa chiave in ogni periodo
            base = market[:-3] if (market.endswith('_ht') or market.endswith('_st')) else market
            con = _sqlite3.connect(_LOCAL_DB)
            con.row_factory = _sqlite3.Row
            dates = []
            wins = []
            tot_w = 0
            for r in con.execute(
                    "SELECT date_str, ft_home, ft_away, ht_home, ht_away, st_home, st_away FROM matches WHERE league_id=? "
                    "ORDER BY sort_date ASC, time_str ASC", (lid,)):
                fh, fa = r['ft_home'], r['ft_away']
                if fh is None or fa is None:
                    continue
                if period == 'HT':
                    a, b = r['ht_home'], r['ht_away']
                elif period == 'ST':
                    a, b = r['st_home'], r['st_away']
                else:
                    a, b = fh, fa
                if a is None or b is None:
                    continue
                w = _bet_market_win(base, a, b)
                if w is None:
                    continue
                dates.append(r['date_str'])
                wins.append(1 if w else 0)
                if w:
                    tot_w += 1
            con.close()
            n = len(wins)
            avg = round(100.0 * tot_w / n, 1) if n else 0.0
            return jsonify({'league_id': lid, 'market': market, 'period': period, 'avg': avg,
                            'total_n': n, 'dates': dates, 'wins': wins})
        except Exception as e:
            return jsonify({'error': str(e)[:300]}), 500
