# ml.py - ML predictor for early-goal matches (first goal within 16')
# Hierarchical Laplace smoothing with back-off: L3 -> L2 -> L1 -> L0
from flask import jsonify, send_from_directory
from datetime import datetime
import time

_ML_CACHE = {'data': None, 'ts': 0}
_ML_CACHE_TTL = 600  # 10 minuti
_ML_MK = ['1', 'X', '2',
          'over_1_5', 'over_2_5', 'over_3_5',
          'under_1_5', 'under_2_5', 'under_3_5',
          'btts_si', 'btts_no']


def _ml_metrics(r):
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
        elif th < ta:
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


def _ml_aggr(ms):
    n = len(ms)
    out = {'n': n}
    for k in _ML_MK:
        out[k] = (sum(m[k] for m in ms) / n) if n else 0.0
    return out


def _ml_shrink(child_p, n_child, parent_p, alpha=3.0):
    if n_child == 0:
        return {k: parent_p.get(k, 0.0) for k in _ML_MK}
    out = {}
    for k in _ML_MK:
        out[k] = (child_p.get(k, 0.0) * n_child + alpha * parent_p.get(k, 0.0)) / (n_child + alpha)
    return out


def _ml_norm_1x2(d):
    s = d.get('1', 0) + d.get('X', 0) + d.get('2', 0)
    if s > 0:
        for k in ('1', 'X', '2'):
            d[k] = d.get(k, 0) / s
    return d


def _build_ml_data(query_fn):
    rows = query_fn("""
        SELECT l.name AS league,
               m.first_goal_team AS pg,
               m.ht_home AS ht_home, m.ht_away AS ht_away,
               m.ft_home AS ft_home, m.ft_away AS ft_away,
               m.total_goals AS total_goals,
               m.btts AS btts,
               m.result AS result
        FROM matches m
        JOIN leagues l ON l.id = m.league_id
        WHERE m.ft_home IS NOT NULL
          AND m.ft_away IS NOT NULL
          AND m.first_goal_team IN ('home', 'away')
    """)
    per = []
    for r in rows:
        m = _ml_metrics(r)
        if m is not None:
            per.append((r, m))

    # Global (L0)
    global_p = _ml_norm_1x2(_ml_aggr([m for _, m in per]))
    global_p.pop('n', None)

    # Raggruppa per campionato
    by_league = {}
    for r, m in per:
        by_league.setdefault(r['league'], []).append((r, m))

    leagues_out = {}
    for lg, rows_lg in by_league.items():
        ms = [m for _, m in rows_lg]
        # L1: league overall (smoothed verso global)
        overall = _ml_norm_1x2(_ml_shrink(_ml_aggr(ms), len(ms), global_p))
        overall['n'] = len(ms)

        # L2: per primo_gol (smoothed verso L1)
        by_pg = {}
        for pg_db, pg_key in (('home', 'casa'), ('away', 'ospite')):
            pms = [m for r, m in rows_lg if r['pg'] == pg_db]
            if pms:
                pg_p = _ml_norm_1x2(_ml_shrink(_ml_aggr(pms), len(pms), overall))
                pg_p['n'] = len(pms)
            else:
                pg_p = dict(overall)
                pg_p['n'] = 0
            by_pg[pg_key] = pg_p

        # L3: per primo_gol + risultato 1T (smoothed verso L2)
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
                ht_p = _ml_norm_1x2(_ml_shrink(_ml_aggr(hms), len(hms), parent_pg))
                ht_p['n'] = len(hms)
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
            'markets': _ML_MK,
            'generated_at': datetime.utcnow().isoformat() + 'Z',
            'alpha': 3.0,
        },
        'global': global_p,
        'leagues': leagues_out,
    }


def register(app, query_fn):
    """Registra le route ML sull'app Flask. query_fn e' la helper DB di app.py."""

    @app.route('/ml')
    def ml_page():
        resp = send_from_directory('templates', 'ml.html')
        resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        return resp

    @app.route('/api/ml-data')
    def api_ml_data():
        now = time.time()
        if _ML_CACHE['data'] is None or (now - _ML_CACHE['ts']) > _ML_CACHE_TTL:
            _ML_CACHE['data'] = _build_ml_data(query_fn)
            _ML_CACHE['ts'] = now
        return jsonify(_ML_CACHE['data'])
