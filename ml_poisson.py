# ml_poisson.py - Motore Poisson bivariato live per generare mercati gol predittivi
#
# Stima lambda_h_residuo e lambda_a_residuo (gol attesi nei minuti rimanenti per home e away)
# da: media gol lega, minuti rimanenti, score state, forza squadre (se calibrata).
# Da quei 2 parametri deriva TUTTI i mercati gol con un'unica distribuzione joint coerente.
#
# Mercati prodotti:
#   - 1X2 FT (1 / X / 2)
#   - Over/Under 0.5, 1.5, 2.5, 3.5, 4.5
#   - BTTS yes/no
#   - HTS yes/no (Home Team Scores - segna casa)
#   - ATS yes/no (Away Team Scores - segna ospite)
#   - Double Chance (1X / X2 / 12)
#   - Draw No Bet (1 DNB / 2 DNB)
#   - Multigoal (0-1 / 2-3 / 4+)
#   - Correct Score (0:0..5:5 + "altro")
#
# Calibrazione automatica:
#   - calibrate_from_turso() ricarica i parametri da predictions_log
#   - filtra solo le partite con first_goal_minute <= CALIB_FIRST_GOAL_MAX_MIN (default 16)
#   - parte solo se count >= CALIB_MIN_SAMPLES (default 200)
#
# Esposto via:
#   - live_poisson_probs(minute, score_h, score_a, league_id, team_h_id, team_a_id)
#   - calibrate_from_turso() -> aggiornamento periodico parametri (chiamato dal cron settimanale)
#   - get_calibration_status() -> per UI di monitoraggio

import math
import time
import os
import json
import urllib.request

# ----- Config / defaults -----
DEFAULT_LEAGUE_AVG_TOTAL_PER90 = float(os.getenv('POISSON_LEAGUE_AVG_PER90', '2.7'))
DEFAULT_HOME_SHARE = float(os.getenv('POISSON_HOME_SHARE', '0.55'))
SCORE_STATE_DEF_FACTOR = float(os.getenv('POISSON_SCORE_STATE_FACTOR', '0.85'))  # leading team scores less
CALIB_FIRST_GOAL_MAX_MIN = int(os.getenv('CALIB_FIRST_GOAL_MAX_MIN', '16'))
CALIB_MIN_SAMPLES = int(os.getenv('CALIB_MIN_SAMPLES', '200'))
TURSO_URL = os.getenv('TURSO_URL', '').rstrip('/')
if TURSO_URL.startswith('libsql://'):
    TURSO_URL = 'https://' + TURSO_URL[len('libsql://'):]
TURSO_TOKEN = os.getenv('TURSO_TOKEN', '')

# ----- Parametri calibrati (aggiornati da calibrate_from_turso) -----
_CALIB_STATE = {
    'last_run_ts': 0,
    'samples_total': 0,
    'samples_filtered': 0,
    'leagues_calibrated': 0,
    'teams_calibrated': 0,
    'note': 'using defaults (no calibration yet)',
}
_LEAGUE_LAMBDA = {}    # league_id -> lambda_per90 (gol totali partita)
_LEAGUE_HOME_SHARE = {} # league_id -> quota gol home
_TEAM_ATT = {}         # team_id -> attack_strength (moltiplicativo, ~1.0)
_TEAM_DEF = {}         # team_id -> defense_strength (moltiplicativo, ~1.0; >1 = subisce di piu')


def get_calibration_status():
    """Stato corrente della calibrazione - per la UI di monitoraggio."""
    return {
        **_CALIB_STATE,
        'first_goal_max_min': CALIB_FIRST_GOAL_MAX_MIN,
        'min_samples': CALIB_MIN_SAMPLES,
        'leagues_loaded': len(_LEAGUE_LAMBDA),
        'teams_loaded': len(_TEAM_ATT),
    }


# ----- Poisson primitives -----

def _poisson_pmf(k, lam):
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def _residual_lambdas(minute, score_h, score_a, league_id=None, team_h_id=None, team_a_id=None):
    """Stima (lambda_h, lambda_a) per i minuti rimanenti."""
    league_avg = _LEAGUE_LAMBDA.get(league_id, DEFAULT_LEAGUE_AVG_TOTAL_PER90)
    home_share = _LEAGUE_HOME_SHARE.get(league_id, DEFAULT_HOME_SHARE)

    minutes_left = max(1, 90 - int(minute or 0))
    base_h = league_avg * home_share * (minutes_left / 90.0)
    base_a = league_avg * (1.0 - home_share) * (minutes_left / 90.0)

    # Score-state adjustment: chi e' in vantaggio difende di piu'
    diff = score_h - score_a
    if diff > 0:
        base_h *= SCORE_STATE_DEF_FACTOR
        base_a *= (2.0 - SCORE_STATE_DEF_FACTOR)
    elif diff < 0:
        base_a *= SCORE_STATE_DEF_FACTOR
        base_h *= (2.0 - SCORE_STATE_DEF_FACTOR)

    # Team strength multiplicativo (se calibrato)
    if team_h_id in _TEAM_ATT and team_a_id in _TEAM_DEF:
        base_h *= _TEAM_ATT[team_h_id] * _TEAM_DEF[team_a_id]
    if team_a_id in _TEAM_ATT and team_h_id in _TEAM_DEF:
        base_a *= _TEAM_ATT[team_a_id] * _TEAM_DEF[team_h_id]

    return max(0.01, base_h), max(0.01, base_a)


def _joint_dist(lam_h, lam_a, k_max=8):
    """Matrice (k_max+1)x(k_max+1): P(h_residual=i, a_residual=j) - indipendenza Poisson."""
    ph = [_poisson_pmf(i, lam_h) for i in range(k_max + 1)]
    pa = [_poisson_pmf(i, lam_a) for i in range(k_max + 1)]
    return [[ph[i] * pa[j] for j in range(k_max + 1)] for i in range(k_max + 1)]


# ----- API pubblica: distribuzione mercati live -----

def live_poisson_probs(minute, score_h, score_a, league_id=None, team_h_id=None, team_a_id=None):
    """Genera probabilita su tutti i mercati gol derivabili dal Poisson bivariato.

    Args:
        minute: int (1-90+), minuto attuale del match
        score_h, score_a: int, score corrente
        league_id, team_h_id, team_a_id: opzionali, per usare i parametri calibrati

    Returns:
        dict con chiavi: 1, X, 2, over_*_5, under_*_5, btts_si/no, hts_si/no, ats_si/no,
                         1X, X2, 12, dnb_1, dnb_2, mg_0_1, mg_2_3, mg_4_plus, cs_h_a, cs_other,
                         _lambdas (debug)
    """
    score_h = int(score_h or 0)
    score_a = int(score_a or 0)
    lam_h, lam_a = _residual_lambdas(minute, score_h, score_a, league_id, team_h_id, team_a_id)
    K = 8
    M = _joint_dist(lam_h, lam_a, K)
    out = {'_lambdas': {'h_residual': round(lam_h, 3), 'a_residual': round(lam_a, 3),
                         'minute': minute, 'score_h': score_h, 'score_a': score_a}}

    # Pre-calcola somme per riusare
    p1 = pX = p2 = 0.0
    p_btts = 0.0
    p_01 = p_23 = p_4plus = 0.0
    p_over = {0.5: 0.0, 1.5: 0.0, 2.5: 0.0, 3.5: 0.0, 4.5: 0.0}
    cs = {}

    for i in range(K + 1):
        for j in range(K + 1):
            p = M[i][j]
            ft_h = score_h + i
            ft_a = score_a + j
            tot = ft_h + ft_a

            # 1X2
            if ft_h > ft_a: p1 += p
            elif ft_h == ft_a: pX += p
            else: p2 += p

            # BTTS
            if ft_h >= 1 and ft_a >= 1:
                p_btts += p

            # Multigoal
            if tot <= 1: p_01 += p
            elif tot <= 3: p_23 += p
            else: p_4plus += p

            # OU
            for thr in p_over:
                if tot > thr:
                    p_over[thr] += p

            # Correct Score (top 0:0..5:5)
            if ft_h <= 5 and ft_a <= 5:
                key = 'cs_%d_%d' % (ft_h, ft_a)
                cs[key] = cs.get(key, 0.0) + p

    out['1'] = p1
    out['X'] = pX
    out['2'] = p2

    out['btts_si'] = p_btts
    out['btts_no'] = 1.0 - p_btts

    for thr, pv in p_over.items():
        thr_str = str(thr).replace('.', '_')
        out['over_' + thr_str] = pv
        out['under_' + thr_str] = 1.0 - pv

    # HTS / ATS (Segna casa / Segna ospite a fine match)
    if score_h >= 1:
        out['hts_si'], out['hts_no'] = 1.0, 0.0
    else:
        out['hts_no'] = math.exp(-lam_h)
        out['hts_si'] = 1.0 - out['hts_no']
    if score_a >= 1:
        out['ats_si'], out['ats_no'] = 1.0, 0.0
    else:
        out['ats_no'] = math.exp(-lam_a)
        out['ats_si'] = 1.0 - out['ats_no']

    # Doppia Chance
    out['1X'] = p1 + pX
    out['X2'] = pX + p2
    out['12'] = p1 + p2

    # Draw No Bet
    denom = p1 + p2
    if denom > 0:
        out['dnb_1'] = p1 / denom
        out['dnb_2'] = p2 / denom

    # Multigoal
    out['mg_0_1'] = p_01
    out['mg_2_3'] = p_23
    out['mg_4_plus'] = p_4plus

    # Correct Score + "altro"
    other = 1.0 - sum(cs.values())
    cs['cs_other'] = max(0.0, other)
    out.update(cs)

    return out


# ----- Calibrazione automatica da Turso (predictions_log) -----

def _turso_query(sql, args=None):
    """Esegue una query SELECT su Turso via /v2/pipeline. Ritorna lista di dict."""
    if not TURSO_URL or not TURSO_TOKEN:
        return None
    body = {
        'requests': [
            {'type': 'execute',
             'stmt': {'sql': sql, 'args': [{'type': 'text', 'value': str(a)} if isinstance(a, str)
                                            else {'type': 'integer', 'value': str(a)} if isinstance(a, int)
                                            else {'type': 'float', 'value': float(a)}
                                            for a in (args or [])]}},
            {'type': 'close'},
        ]
    }
    req = urllib.request.Request(
        TURSO_URL + '/v2/pipeline',
        data=json.dumps(body).encode('utf-8'),
        method='POST',
        headers={
            'Authorization': 'Bearer ' + TURSO_TOKEN,
            'Content-Type': 'application/json',
        }
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read().decode('utf-8'))
    res = data.get('results') or []
    if not res or res[0].get('type') != 'ok':
        return None
    rs = res[0].get('response', {}).get('result', {})
    cols = [c.get('name') for c in rs.get('cols', [])]
    rows = []
    for row in rs.get('rows', []):
        d = {}
        for k, cell in zip(cols, row):
            v = cell.get('value') if isinstance(cell, dict) else cell
            d[k] = v
        rows.append(d)
    return rows


def calibrate_from_turso():
    """Ricalcola lambda lega e team strengths usando solo partite con first_goal_minute <= CALIB_FIRST_GOAL_MAX_MIN.
    Fail-safe: se mancano sample, NON tocca i parametri esistenti.
    """
    import time
    global _LEAGUE_LAMBDA, _LEAGUE_HOME_SHARE, _TEAM_ATT, _TEAM_DEF, _CALIB_STATE
    _CALIB_STATE['last_run_ts'] = int(time.time())

    # Conta sample totali e filtrati
    try:
        tot_rows = _turso_query("SELECT COUNT(*) AS n FROM predictions_log WHERE ft_home IS NOT NULL")
        flt_rows = _turso_query(
            "SELECT COUNT(*) AS n FROM predictions_log "
            "WHERE ft_home IS NOT NULL AND first_goal_minute IS NOT NULL AND first_goal_minute <= ?",
            [CALIB_FIRST_GOAL_MAX_MIN]
        )
    except Exception as e:
        _CALIB_STATE['note'] = 'turso query failed: ' + str(e)[:200]
        return _CALIB_STATE

    samples_total = (tot_rows or [{}])[0].get('n', 0) or 0
    samples_filtered = (flt_rows or [{}])[0].get('n', 0) or 0
    _CALIB_STATE['samples_total'] = int(samples_total)
    _CALIB_STATE['samples_filtered'] = int(samples_filtered)

    if samples_filtered < CALIB_MIN_SAMPLES:
        _CALIB_STATE['note'] = 'need %d more samples (have %d, need %d)' % (
            CALIB_MIN_SAMPLES - samples_filtered, samples_filtered, CALIB_MIN_SAMPLES)
        return _CALIB_STATE

    # MLE per lega: lambda_lega = media (ft_home + ft_away)
    league_rows = _turso_query(
        "SELECT league_id, AVG(ft_home + ft_away) AS lam, "
        "AVG(CAST(ft_home AS REAL) / NULLIF(ft_home + ft_away, 0)) AS hs, "
        "COUNT(*) AS n "
        "FROM predictions_log "
        "WHERE ft_home IS NOT NULL AND first_goal_minute IS NOT NULL AND first_goal_minute <= ? "
        "GROUP BY league_id HAVING n >= 20",
        [CALIB_FIRST_GOAL_MAX_MIN]
    ) or []

    new_league_lambda = {}
    new_league_hshare = {}
    for r in league_rows:
        lid = r.get('league_id')
        lam = r.get('lam')
        hs = r.get('hs')
        if lid is not None and lam is not None and float(lam) > 0:
            new_league_lambda[int(lid)] = float(lam)
            if hs is not None:
                new_league_hshare[int(lid)] = max(0.3, min(0.7, float(hs)))

    # MLE per squadra: attack/defense strength rispetto alla media lega
    team_rows = _turso_query(
        "SELECT team_id, league_id, AVG(scored) AS s, AVG(conceded) AS c, COUNT(*) AS n FROM ("
        "  SELECT home_team_id AS team_id, league_id, ft_home AS scored, ft_away AS conceded "
        "    FROM predictions_log WHERE ft_home IS NOT NULL "
        "    AND first_goal_minute IS NOT NULL AND first_goal_minute <= ? "
        "  UNION ALL "
        "  SELECT away_team_id AS team_id, league_id, ft_away AS scored, ft_home AS conceded "
        "    FROM predictions_log WHERE ft_home IS NOT NULL "
        "    AND first_goal_minute IS NOT NULL AND first_goal_minute <= ? "
        ") GROUP BY team_id HAVING n >= 5",
        [CALIB_FIRST_GOAL_MAX_MIN, CALIB_FIRST_GOAL_MAX_MIN]
    ) or []

    new_team_att = {}
    new_team_def = {}
    for r in team_rows:
        tid = r.get('team_id')
        lid = r.get('league_id')
        s = r.get('s')
        c = r.get('c')
        if tid is None or s is None or c is None:
            continue
        league_avg = new_league_lambda.get(int(lid) if lid else -1, DEFAULT_LEAGUE_AVG_TOTAL_PER90)
        league_per_team = league_avg / 2.0  # gol medi per squadra in una partita
        if league_per_team > 0:
            new_team_att[int(tid)] = max(0.5, min(2.0, float(s) / league_per_team))
            new_team_def[int(tid)] = max(0.5, min(2.0, float(c) / league_per_team))

    _LEAGUE_LAMBDA = new_league_lambda
    _LEAGUE_HOME_SHARE = new_league_hshare
    _TEAM_ATT = new_team_att
    _TEAM_DEF = new_team_def
    _CALIB_STATE['leagues_calibrated'] = len(new_league_lambda)
    _CALIB_STATE['teams_calibrated'] = len(new_team_att)
    _CALIB_STATE['note'] = 'OK: %d leagues, %d teams calibrated from %d filtered samples' % (
        len(new_league_lambda), len(new_team_att), samples_filtered)
    return _CALIB_STATE


def maybe_recalibrate(min_interval_hours=6):
    """Auto-trigger calibrate_from_turso() solo se sono passate >= min_interval_hours dall'ultima esecuzione.
    Best-effort: cattura tutte le eccezioni, non blocca il caller (es. odds_logger tick).
    """
    try:
        last_ts = _CALIB_STATE.get('last_run_ts', 0) or 0
        elapsed = time.time() - last_ts
        if last_ts > 0 and elapsed < min_interval_hours * 3600:
            return {'skipped': True, 'reason': 'too soon', 'elapsed_h': round(elapsed/3600, 2)}
        return calibrate_from_turso()
    except Exception as e:
        return {'error': str(e)[:200]}

# ----- Flask routes (registrate da app.py) -----

def register(app):
    from flask import jsonify, request

    @app.route('/api/ml-poisson')
    def api_ml_poisson():
        """Debug: probabilita Poisson live per minuto/score arbitrari."""
        minute = request.args.get('minute', default=45, type=int)
        sh = request.args.get('score_h', default=0, type=int)
        sa = request.args.get('score_a', default=0, type=int)
        lid = request.args.get('league_id', type=int)
        thid = request.args.get('team_h_id', type=int)
        taid = request.args.get('team_a_id', type=int)
        return jsonify(live_poisson_probs(minute, sh, sa, lid, thid, taid))

    @app.route('/api/ml-calibration-status')
    def api_calib_status():
        return jsonify(get_calibration_status())

    @app.route('/api/ml-recalibrate')
    def api_recalibrate():
        # Protezione opzionale via token (riusa TICK_AUTH_TOKEN)
        tick_token = os.getenv('TICK_AUTH_TOKEN', '')
        if tick_token:
            if request.args.get('token') != tick_token:
                return jsonify({'error': 'unauthorized'}), 401
        return jsonify(calibrate_from_turso())
