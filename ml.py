# ml.py - ML predictor: (1) early-goal legacy + (2) minute+score live-betting
# Hierarchical Laplace smoothing with back-off (alpha=3)
from flask import jsonify, send_from_directory, request
from datetime import datetime
import time
import re
import ml_pick  # live betting picks: API-Football odds + Kelly
import odds_logger  # live odds snapshot logger for historical dataset
import ml_poisson  # Poisson bivariate live engine: correct score, HTS/ATS, mercati gol
import predictions_settlement  # Settlement pipeline: popola predictions_log con risultati FT

_ML_CACHE = {'eg_data': None, 'eg_ts': 0, 'adv_data': None, 'adv_ts': 0}
_ML_TTL = 600  # 10 minuti

# Mercati tempo pieno
_MK_FT = ['1', 'X', '2',
         'over_1_5', 'over_2_5', 'over_3_5',
         'under_1_5', 'under_2_5', 'under_3_5',
         'btts_si', 'btts_no']

# Mercati secondo tempo (solo goal 2T)
_MK_2H = ['2h_1', '2h_X', '2h_2',
          '2h_over_0_5', '2h_over_1_5', '2h_over_2_5',
          '2h_under_0_5', '2h_under_1_5', '2h_under_2_5',
          '2h_btts_si', '2h_btts_no',
          '2h_home_scores', '2h_away_scores']

_MK_ALL = _MK_FT + _MK_2H

# Minuti snapshot per advanced predictor
_SNAP_MINUTES = [45, 60, 70, 80]

# Regex per parsare goals_html
# Match: <span class="away-goal">MIN'</span>  OPPURE  MIN'
_GOAL_RE = re.compile(
    r'<span[^>]*class="[^"]*away-goal[^"]*"[^>]*>\s*(\d+(?:\+\d+)?)\s*\'?\s*</span>'
    r'|(\d+(?:\+\d+)?)\s*\''
)


def _parse_min(s):
    if '+' in s:
        parts = s.split('+')
        return int(parts[0]) + int(parts[1])
    return int(s)


def _parse_goals(html):
    """Ritorna lista di (minuto, team) dai goals_html."""
    if not html:
        return []
    out = []
    for m in _GOAL_RE.finditer(html):
        away, home = m.group(1), m.group(2)
        if away is not None:
            out.append((_parse_min(away), 'away'))
        elif home is not None:
            out.append((_parse_min(home), 'home'))
    return out


def _score_at(goals, minute):
    """Score (home, away) considerando i goal con minuto <= `minute`."""
    h = sum(1 for (mn, t) in goals if mn <= minute and t == 'home')
    a = sum(1 for (mn, t) in goals if mn <= minute and t == 'away')
    return (h, a)


def _ft_metrics(r):
    th = r.get('ft_home')
    ta = r.get('ft_away')
    if th is None or ta is None:
        return None
    tg = r.get('total_goals')
    if tg is None:
        tg = th + ta
    btts = 1 if (th > 0 and ta > 0) else 0
    res = r.get('result')
    if res not in ('1', 'X', '2'):
        if th > ta:
            res = '1'
        elif ta > th:
            res = '2'
        else:
            res = 'X'
    return {
        '1': 1 if res == '1' else 0,
        'X': 1 if res == 'X' else 0,
        '2': 1 if res == '2' else 0,
        'over_1_5': 1 if tg > 1 else 0,
        'over_2_5': 1 if tg > 2 else 0,
        'over_3_5': 1 if tg > 3 else 0,
        'under_1_5': 1 if tg < 2 else 0,
        'under_2_5': 1 if tg < 3 else 0,
        'under_3_5': 1 if tg < 4 else 0,
        'btts_si': btts,
        'btts_no': 1 - btts,
    }


def _2h_metrics(r):
    sh = r.get('st_home')
    sa = r.get('st_away')
    if sh is None or sa is None:
        return None
    sg = sh + sa
    btts2 = 1 if (sh > 0 and sa > 0) else 0
    if sh > sa:
        res2 = '1'
    elif sa > sh:
        res2 = '2'
    else:
        res2 = 'X'
    return {
        '2h_1': 1 if res2 == '1' else 0,
        '2h_X': 1 if res2 == 'X' else 0,
        '2h_2': 1 if res2 == '2' else 0,
        '2h_over_0_5': 1 if sg > 0 else 0,
        '2h_over_1_5': 1 if sg > 1 else 0,
        '2h_over_2_5': 1 if sg > 2 else 0,
        '2h_under_0_5': 1 if sg < 1 else 0,
        '2h_under_1_5': 1 if sg < 2 else 0,
        '2h_under_2_5': 1 if sg < 3 else 0,
        '2h_btts_si': btts2,
        '2h_btts_no': 1 - btts2,
        '2h_home_scores': 1 if sh > 0 else 0,
        '2h_away_scores': 1 if sa > 0 else 0,
    }


def _aggr(ms, keys):
    n = len(ms)
    out = {'n': n}
    if n == 0:
        for k in keys:
            out[k] = 0.0
        return out
    for k in keys:
        vals = [m[k] for m in ms if k in m]
        out[k] = sum(vals) / len(vals) if vals else 0.0
    return out


def _shrink(child_n, child_p, parent_p, keys, alpha=3.0):
    if child_n == 0:
        return {k: parent_p.get(k, 0.0) for k in keys}
    return {k: (child_p.get(k, 0.0) * child_n + alpha * parent_p.get(k, 0.0)) / (child_n + alpha) for k in keys}


def _norm_1x2(d):
    # 1X2 tempo pieno
    s = d.get('1', 0) + d.get('X', 0) + d.get('2', 0)
    if s > 0:
        for k in ('1', 'X', '2'):
            d[k] = d.get(k, 0) / s
    # 1X2 secondo tempo
    s2 = d.get('2h_1', 0) + d.get('2h_X', 0) + d.get('2h_2', 0)
    if s2 > 0:
        for k in ('2h_1', '2h_X', '2h_2'):
            d[k] = d.get(k, 0) / s2
    return d


# âââââââââââââââââââ Early-goal predictor (legacy /ml) âââââââââââââââââââ

def _build_eg_data(query_fn):
    rows = query_fn("""
        SELECT l.name AS league, m.first_goal_team AS pg,
               m.ht_home AS ht_home, m.ht_away AS ht_away,
               m.ft_home AS ft_home, m.ft_away AS ft_away,
               m.total_goals AS total_goals, m.btts AS btts, m.result AS result,
               m.st_home AS st_home, m.st_away AS st_away
        FROM matches m
        JOIN leagues l ON l.id = m.league_id
        WHERE m.ft_home IS NOT NULL AND m.ft_away IS NOT NULL
          AND m.first_goal_team IN ('home', 'away')
    """)
    per = []
    for r in rows:
        m = _ft_metrics(r)
        if m is not None:
            m2 = _2h_metrics(r)
            if m2 is not None:
                m = {**m, **m2}
            per.append((r, m))
    global_p = _norm_1x2(_aggr([m for _, m in per], _MK_ALL))
    global_p.pop('n', None)
    by_league = {}
    for r, m in per:
        by_league.setdefault(r['league'], []).append((r, m))
    leagues_out = {}
    for lg, rows_lg in by_league.items():
        ms = [m for _, m in rows_lg]
        overall = _norm_1x2(_shrink(len(ms), _aggr(ms, _MK_ALL), global_p, _MK_ALL))
        overall['n'] = len(ms)
        by_pg = {}
        for pg_db, pg_key in (('home', 'casa'), ('away', 'ospite')):
            pms = [m for r, m in rows_lg if r['pg'] == pg_db]
            if pms:
                pg_p = _norm_1x2(_shrink(len(pms), _aggr(pms, _MK_FT), overall, _MK_FT))
                pg_p['n'] = len(pms)
            else:
                pg_p = dict(overall)
                pg_p['n'] = 0
            by_pg[pg_key] = pg_p
        by_pg_ht = {'casa': {}, 'ospite': {}}
        for pg_db, pg_key in (('home', 'casa'), ('away', 'ospite')):
            parent_pg = by_pg[pg_key]
            buckets = {}
            for r, m in rows_lg:
                if r['pg'] != pg_db:
                    continue
                h = r.get('ht_home')
                a = r.get('ht_away')
                if h is None or a is None:
                    continue
                buckets.setdefault('%d-%d' % (h, a), []).append(m)
            for s, hms in buckets.items():
                if len(hms) < 3:
                    continue
                ht_p = _norm_1x2(_shrink(len(hms), _aggr(hms, _MK_FT), parent_pg, _MK_FT))
                ht_p['n'] = len(hms)
                # Correggi mercati che diventano certi dato il punteggio HT
                h_ht, a_ht = int(s.split('-')[0]), int(s.split('-')[1])
                ht_total = h_ht + a_ht
                if h_ht > 0 and a_ht > 0:
                    ht_p['btts_si'] = 1.0
                    ht_p['btts_no'] = 0.0
                if ht_total >= 2:
                    ht_p['over_1_5'] = 1.0
                    ht_p['under_1_5'] = 0.0
                if ht_total >= 3:
                    ht_p['over_2_5'] = 1.0
                    ht_p['under_2_5'] = 0.0
                if ht_total >= 4:
                    ht_p['over_3_5'] = 1.0
                    ht_p['under_3_5'] = 0.0
                by_pg_ht[pg_key][s] = ht_p
        leagues_out[lg] = {
            'n': len(rows_lg),
            'overall': overall,
            'by_primo_gol': by_pg,
            'by_primo_gol_ht': by_pg_ht,
        }
    return {
        'meta': {
            'n_matches': len(per),
            'n_leagues': len(leagues_out),
            'markets': _MK_FT,
            'generated_at': datetime.utcnow().isoformat() + 'Z',
            'alpha': 3.0,
        },
        'global': global_p,
        'leagues': leagues_out,
    }


# âââââââââââââââââââ Advanced predictor (minute + score, FT + 2H) âââââââââââââââââââ

def _build_adv_data(query_fn):
    rows = query_fn("""
        SELECT l.name AS league,
               m.ht_home, m.ht_away,
               m.st_home, m.st_away,
               m.ft_home, m.ft_away,
               m.total_goals, m.btts, m.result,
               m.goals_html
        FROM matches m
        JOIN leagues l ON l.id = m.league_id
        WHERE m.ft_home IS NOT NULL AND m.ft_away IS NOT NULL
    """)
    per_match = []
    for r in rows:
        ft = _ft_metrics(r)
        h2 = _2h_metrics(r)
        if ft is None or h2 is None:
            continue
        m = {}
        m.update(ft)
        m.update(h2)
        goals = _parse_goals(r.get('goals_html') or '')
        per_match.append({
            'league': r['league'],
            'goals': goals,
            'ht_home': r.get('ht_home') or 0,
            'ht_away': r.get('ht_away') or 0,
            'metrics': m,
        })

    # Global parent (tutti i match validi)
    global_metrics = [pm['metrics'] for pm in per_match]
    global_p = _norm_1x2(_aggr(global_metrics, _MK_ALL))
    global_p.pop('n', None)

    # Raggruppa per campionato
    by_league = {}
    for pm in per_match:
        by_league.setdefault(pm['league'], []).append(pm)

    leagues_out = {}
    for lg, matches in by_league.items():
        ms = [pm['metrics'] for pm in matches]
        # L1: league overall
        overall = _norm_1x2(_shrink(len(ms), _aggr(ms, _MK_ALL), global_p, _MK_ALL))
        overall['n'] = len(ms)

        # L2: per snapshot minuto + score
        by_minute = {}
        for M in _SNAP_MINUTES:
            score_buckets = {}
            for pm in matches:
                if M == 45:
                    h, a = pm['ht_home'], pm['ht_away']
                else:
                    h, a = _score_at(pm['goals'], M)
                key = '%d-%d' % (h, a)
                score_buckets.setdefault(key, []).append(pm['metrics'])
            bucket_out = {}
            for score_key, bms in score_buckets.items():
                if len(bms) < 3:
                    continue  # evita rumore da campione piccolo
                p = _norm_1x2(_shrink(len(bms), _aggr(bms, _MK_ALL), overall, _MK_ALL))
                p['n'] = len(bms)
                bucket_out[score_key] = p
            by_minute[str(M)] = bucket_out

        leagues_out[lg] = {
            'n': len(matches),
            'overall': overall,
            'by_minute': by_minute,
        }

    return {
        'meta': {
            'n_matches': len(per_match),
            'n_leagues': len(leagues_out),
            'markets_ft': _MK_FT,
            'markets_2h': _MK_2H,
            'snapshot_minutes': _SNAP_MINUTES,
            'alpha': 3.0,
            'generated_at': datetime.utcnow().isoformat() + 'Z',
        },
        'global': global_p,
        'leagues': leagues_out,
    }


# âââââââââââââââââââ Registrazione route âââââââââââââââââââ

def register(app, query_fn):
    """Registra le route ML sull'app Flask."""

    @app.route('/ml')
    def ml_page():
        resp = send_from_directory('templates', 'ml.html')
        resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        return resp

    @app.route('/api/ml-data')
    def api_ml_data():
        now = time.time()
        if _ML_CACHE['eg_data'] is None or (now - _ML_CACHE['eg_ts']) > _ML_TTL:
            _ML_CACHE['eg_data'] = _build_eg_data(query_fn)
            _ML_CACHE['eg_ts'] = now
        return jsonify(_ML_CACHE['eg_data'])

    @app.route('/api/ml-advanced')
    def api_ml_advanced():
        """Advanced: stats per campionato + minuto + score corrente (FT + 2H)."""
        now = time.time()
        if _ML_CACHE['adv_data'] is None or (now - _ML_CACHE['adv_ts']) > _ML_TTL:
            _ML_CACHE['adv_data'] = _build_adv_data(query_fn)
            _ML_CACHE['adv_ts'] = now
        return jsonify(_ML_CACHE['adv_data'])

    def _get_adv_data():
        """Provider usato da ml_pick: ritorna adv_data con la stessa cache di /api/ml-advanced."""
        now = time.time()
        if _ML_CACHE['adv_data'] is None or (now - _ML_CACHE['adv_ts']) > _ML_TTL:
            _ML_CACHE['adv_data'] = _build_adv_data(query_fn)
            _ML_CACHE['adv_ts'] = now
        return _ML_CACHE['adv_data']

    # Registra le route di ml_pick: /api/ml-env-check, /api/ml-live-fixtures-af,
    # /api/ml-odds-debug, /api/ml-pick
    ml_pick.register(app, _get_adv_data)
    ml_pick.register_picks_ui(app, _get_adv_data)  # /picks, /ml-accuracy, /api/ml-live-picks-all, /api/ml-accuracy-stats
    # Registra le route di odds_logger: /api/odds-logger-tick, -stats, -dump, -csv, -ddl
    odds_logger.register(app)
    ml_poisson.register(app)  # /api/ml-poisson, /api/ml-calibration-status, /api/ml-recalibrate
    predictions_settlement.register(app)  # /api/predictions-log-ddl, /api/predictions-settle, /api/predictions-log-stats
