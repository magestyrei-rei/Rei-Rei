# ml_pick.py - Live betting picks: combina modello ML + quote API-Football + Kelly
# Modulo complementare a ml.py. Viene importato e registrato dentro register() di ml.py.
#
# Endpoints esposti:
#   GET /api/ml-env-check            -> verifica che APISPORTS_KEY sia settata (non espone il valore)
#   GET /api/ml-live-fixtures-af     -> partite live da API-Football (con league_id, minuto, score)
#   GET /api/ml-odds-debug?fixture=X -> debug payload quote per una fixture
#   GET /api/ml-pick?fixture=X&...   -> top pick con edge positivo + Kelly stake

import os
import json
import time
import re
import urllib.request
import urllib.parse
from flask import jsonify, request

APISPORTS_BASE = 'https://v3.football.api-sports.io'
APISPORTS_KEY = os.getenv('APISPORTS_KEY', '')

# Cache
_ODDS_CACHE = {}         # {fixture_id: (ts, data)}
_ODDS_TTL = 30           # secondi - quote live si muovono veloci
_LIVE_FIX_CACHE = {'ts': 0, 'data': None}
_LIVE_FIX_TTL = 45       # secondi

# -------------------- API-Football helpers --------------------

def _apisports_get(path, params=None):
    if not APISPORTS_KEY:
        return {'error': 'APISPORTS_KEY not set'}
    url = APISPORTS_BASE + path
    if params:
        url += '?' + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={
        'x-apisports-key': APISPORTS_KEY,
        'Accept': 'application/json',
    })
    try:
        with urllib.request.urlopen(req, timeout=12) as r:
            return json.loads(r.read().decode('utf-8'))
    except Exception as e:
        return {'error': str(e), 'url': url}


def _get_live_odds(fixture_id):
    now = time.time()
    c = _ODDS_CACHE.get(fixture_id)
    if c and now - c[0] < _ODDS_TTL:
        return c[1]
    data = _apisports_get('/odds/live', {'fixture': fixture_id})
    _ODDS_CACHE[fixture_id] = (now, data)
    return data


def _get_live_fixtures_af():
    now = time.time()
    if _LIVE_FIX_CACHE['data'] is not None and (now - _LIVE_FIX_CACHE['ts']) < _LIVE_FIX_TTL:
        return _LIVE_FIX_CACHE['data']
    data = _apisports_get('/fixtures', {'live': 'all'})
    _LIVE_FIX_CACHE['ts'] = now
    _LIVE_FIX_CACHE['data'] = data
    return data


def _fixture_to_model_ctx(fixture_data):
    """Estrae (fixture_id, league_id, league_name, country, minute, score) da payload API-Football."""
    if not isinstance(fixture_data, dict):
        return None
    fx = fixture_data.get('fixture', {}) or {}
    lg = fixture_data.get('league', {}) or {}
    goals = fixture_data.get('goals', {}) or {}
    teams = fixture_data.get('teams', {}) or {}
    status = fx.get('status', {}) or {}
    elapsed = status.get('elapsed')
    if elapsed is None:
        return None
    return {
        'fixture_id': fx.get('id'),
        'league_id': lg.get('id'),
        'league_name': lg.get('name'),
        'country': lg.get('country'),
        'minute': elapsed,
        'status_short': status.get('short', ''),
        'score_home': goals.get('home') if goals.get('home') is not None else 0,
        'score_away': goals.get('away') if goals.get('away') is not None else 0,
        'home': (teams.get('home') or {}).get('name'),
        'away': (teams.get('away') or {}).get('name'),
    }

# -------------------- Market normalization --------------------

# Mappa value (lowercase) -> side per Over/Under senza linea esplicita
_OU_LINE_RE = re.compile(r'(over|under)\s*([0-9]+(?:\.[0-9]+)?)')
_LINE_IN_NAME_RE = re.compile(r'([0-9]+(?:\.[0-9]+)?)')


def _is_second_half(bet_name_lc):
    keys = ['second half', '2nd half', '2nd-half', 'seconde', 'secondo tempo', 'ht-ft']
    # NB: 'half time' può essere 1T, non 2T -> escluso
    for k in keys:
        if k in bet_name_lc:
            return True
    return False


def _is_first_half(bet_name_lc):
    # 1T markets (non ci interessano per ora, return None)
    keys = ['first half', '1st half', '1st-half', 'primo tempo', 'half time']
    # ma 'half time/full time' è un mercato diverso, esclude
    if 'half time/full time' in bet_name_lc or 'halftime/fulltime' in bet_name_lc:
        return False
    for k in keys:
        if k in bet_name_lc:
            return True
    return False


def _norm_line(x):
    """2.5 -> '2_5', 0.5 -> '0_5'"""
    s = str(x).strip()
    if '.' not in s:
        s = s + '.5' if s.isdigit() else s
    return s.replace('.', '_')


def _normalize_af_market(bet_name, bet_value, handicap=None):
    """Ritorna market_key del nostro modello, oppure None.
    Supporta sia schema /odds (pre-match) sia /odds/live (in-play, line in 'handicap')."""
    if not bet_name:
        return None
    n = bet_name.lower().strip()
    v = str(bet_value or '').lower().strip()
    h = None
    if handicap is not None and str(handicap).strip() not in ('', 'None'):
        h = str(handicap).strip()

    # Determina prefisso (1h_ / 2h_ / vuoto)
    is_1h = _is_first_half(n)
    is_2h = _is_second_half(n)
    if is_1h:
        prefix = '1h_'
    elif is_2h:
        prefix = '2h_'
    else:
        prefix = ''

    # ============ Time-windowed 1X2: "1x2 - 30 minutes" ============
    import re as _re
    tm = _re.match(r'1x2\s*-\s*(\d+)\s*minutes?', n)
    if tm:
        mins = tm.group(1)
        if v in ('home', '1'): return 't' + mins + '_1'
        if v in ('draw', 'x'): return 't' + mins + '_X'
        if v in ('away', '2'): return 't' + mins + '_2'

    # ============ 1X2 / Match Result ============
    if (('match winner' in n) or ('fulltime result' in n) or ('full time result' in n) or
            (n.startswith('1x2') and '-' not in n) or
            (n in ('result', '1x2', 'match result', 'winner'))):
        if v in ('home', '1', 'casa'): return prefix + '1'
        if v in ('draw', 'x', 'pareggio'): return prefix + 'X'
        if v in ('away', '2'): return prefix + '2'

    # ============ Second Half Winner generico ============
    if is_2h and ('winner' in n or 'result' in n):
        if v in ('home', '1'): return prefix + '1'
        if v in ('draw', 'x'): return prefix + 'X'
        if v in ('away', '2'): return prefix + '2'

    # ============ Double Chance ============
    if 'double chance' in n:
        if v in ('home or draw', '1x', '1 or x'): return prefix + 'dc_1X'
        if v in ('home or away', '12', '1 or 2'): return prefix + 'dc_12'
        if v in ('away or draw', 'x2', 'x or 2'): return prefix + 'dc_X2'

    # ============ Draw No Bet ============
    if 'draw no bet' in n:
        if v in ('home', '1'): return prefix + 'dnb_1'
        if v in ('away', '2'): return prefix + 'dnb_2'

    # ============ Goals Over/Under (con handicap field) ============
    if ('over/under' in n or 'over under' in n or 'goals over' in n or 'goals under' in n or
            'total goals' in n or 'match goals' in n or
            ('goals' in n and ('over' in v or 'under' in v))):
        line = h
        if not line:
            mv = _OU_LINE_RE.search(v)
            if mv:
                line = mv.group(2)
        if not line:
            mn = _re.search(r'(\d+(?:[.,]\d+)?)', n)
            if mn: line = mn.group(1)
        if line:
            if v == 'over' or v.startswith('over'):
                return prefix + 'over_' + _norm_line(line)
            if v == 'under' or v.startswith('under'):
                return prefix + 'under_' + _norm_line(line)

    # ============ BTTS ============
    if 'both teams' in n or 'btts' in n:
        if v in ('yes', 'si', 'si\u0301', '1'):
            return prefix + 'btts_si'
        if v in ('no', '0'):
            return prefix + 'btts_no'

    # ============ Goals Odd/Even ============
    if 'odd/even' in n and 'goals' in n or 'goals odd/even' in n:
        if v == 'odd': return prefix + 'goals_odd'
        if v == 'even': return prefix + 'goals_even'

    # ============ Correct Score / Final Score / Exact Score ============
    if 'final score' in n or 'correct score' in n or 'exact score' in n:
        cs = _re.match(r'^(\d+)\s*[-:]\s*(\d+)$', v)
        if cs:
            return prefix + 'cs_' + cs.group(1) + '_' + cs.group(2)

    # ============ How many goals will (Home/Away) Team score? ============
    if 'how many goals' in n:
        team = 'home' if 'home' in n else ('away' if 'away' in n else None)
        if team:
            if v == '1': return team + '_goals_1'
            if v == '2': return team + '_goals_2'
            if 'or more' in v or v == '3+' or v == '3 or more': return team + '_goals_3p'
            if 'no goal' in v or v == '0': return team + '_goals_0'

    # ============ Result / Both Teams To Score (combinato) ============
    if 'result / both teams' in n or 'result/both teams' in n:
        if '/' in v:
            r, b = v.split('/', 1)
            r = r.strip(); b = b.strip()
            r_key = '1' if r == 'home' else ('X' if r == 'draw' else ('2' if r == 'away' else None))
            b_key = 'si' if b in ('yes', 'si') else ('no' if b == 'no' else None)
            if r_key and b_key:
                return prefix + 'res_btts_' + r_key + '_' + b_key

    return None
def _ingest_value(out, bet_name, vd, bk_name):
    """Helper: estrae la chiave mercato e la quota da un value entry, e la inserisce in out."""
    if vd.get('suspended') is True:
        return
    val = vd.get('value', '')
    handicap = vd.get('handicap')
    mkt = _normalize_af_market(bet_name, val, handicap=handicap)
    if not mkt:
        return
    try:
        q = float(vd.get('odd', vd.get('odds', 0)) or 0)
    except (ValueError, TypeError):
        return
    if q > 1.0:
        out.setdefault(mkt, {})[bk_name] = q


def _parse_odds_payload(data):
    """Ritorna {market_key: {bookie_name_lower: quota_float}}.
    Supporta DUE schemi distinti di API-Football:
      - Pre-match (/odds):  response[i].bookmakers[j].bets[k].values[m]
      - Live in-play (/odds/live):  response[i].odds[k].values[m]   (no bookmakers, quote aggregate)
    """
    out = {}
    if not isinstance(data, dict):
        return out
    for resp in data.get('response', []) or []:
        if not isinstance(resp, dict):
            continue
        # --- Schema A: pre-match (bookmakers list) ---
        bms = resp.get('bookmakers') or []
        if bms:
            for bk in bms:
                bk_name = (bk.get('name') or '').lower().strip()
                if not bk_name:
                    continue
                for bet in bk.get('bets', []) or []:
                    bet_name = bet.get('name', '')
                    for vd in bet.get('values', []) or []:
                        _ingest_value(out, bet_name, vd, bk_name)
            continue
        # --- Schema B: live in-play (odds list direttamente) ---
        live_bets = resp.get('odds') or []
        if live_bets:
            # Usa un nome bookmaker placeholder per le quote aggregate API-Football
            bk_name = 'apifootball-live'
            # Skip se la fixture/odds sono globalmente bloccate o sospese
            st = resp.get('status') or {}
            if st.get('blocked') or st.get('stopped'):
                continue
            for bet in live_bets:
                bet_name = bet.get('name', '')
                for vd in bet.get('values', []) or []:
                    _ingest_value(out, bet_name, vd, bk_name)
    return out

# -------------------- Model lookup helpers --------------------

def _pick_snapshot(minute):
    """Snapshot disponibili: 45, 60, 70, 80."""
    if minute <= 52:
        return '45'
    if minute <= 65:
        return '60'
    if minute <= 75:
        return '70'
    return '80'


def _score_bucket(h, a, cap=3):
    h = min(int(h), cap)
    a = min(int(a), cap)
    return '%d-%d' % (h, a)


def _find_league_in_model(adv_data, league_id, league_name, country):
    """Cerca la league nel dict adv_data['leagues']. Chiavi sono nomi composti tipo 'Italy - Serie A'."""
    leagues = (adv_data or {}).get('leagues') or {}
    if not leagues:
        return None, None

    # 1. Match esatto sulla chiave composta
    if country and league_name:
        key = '%s - %s' % (country, league_name)
        if key in leagues:
            return key, leagues[key]

    # 2. Match esatto su league_name
    if league_name and league_name in leagues:
        return league_name, leagues[league_name]

    # 3. Sub-string match (case-insensitive) su league_name
    if league_name:
        ln_lc = league_name.lower()
        for k, v in leagues.items():
            if ln_lc in k.lower():
                return k, v

    # 4. Country + partial league_name
    if country and league_name:
        c_lc = country.lower()
        ln_lc = league_name.lower()
        for k, v in leagues.items():
            k_lc = k.lower()
            if c_lc in k_lc and ln_lc.split()[0] in k_lc:
                return k, v

    return None, None


def _extract_probs(league_data, minute, score_home, score_away):
    """
    Ritorna (probs_dict, source_label).
    Fallback: bucket(minute,score) -> snapshot overall -> league overall.
    """
    if not isinstance(league_data, dict):
        return {}, 'none'
    by_minute = league_data.get('by_minute') or league_data.get('snapshots') or {}
    snap = _pick_snapshot(minute)
    sk = _score_bucket(score_home, score_away)

    snap_data = by_minute.get(snap)
    if isinstance(snap_data, dict):
        # bucket score
        probs = snap_data.get(sk)
        if isinstance(probs, dict) and probs:
            return probs, 'league+min%s+score%s' % (snap, sk)
        # aggregate di questo snapshot (se presente come 'overall' o fallback)
        if 'overall' in snap_data and isinstance(snap_data['overall'], dict):
            return snap_data['overall'], 'league+min%s+overall' % snap

    overall = league_data.get('overall')
    if isinstance(overall, dict):
        return overall, 'league-overall'
    return {}, 'none'

# -------------------- Kelly + edge --------------------

def _compute_picks(model_probs, odds_by_market, bookie_pref, kelly_mult, edge_min, stake_max, capital):
    """Lista pick con edge positivo, ordinata per edge desc."""
    if not isinstance(model_probs, dict) or not isinstance(odds_by_market, dict):
        return []
    picks = []
    fallback_order = [bookie_pref, 'betfair', 'pinnacle', 'pinnacle sports', 'bet365', '1xbet', 'marathonbet', 'unibet']
    for mkt, prob in model_probs.items():
        if mkt == 'n' or not isinstance(prob, (int, float)):
            continue
        if prob <= 0.0 or prob >= 1.0:
            continue
        offers = odds_by_market.get(mkt)
        if not offers:
            continue
        chosen_bookie = None
        chosen_quota = None
        for b in fallback_order:
            if b and b in offers:
                chosen_bookie = b
                chosen_quota = offers[b]
                break
        if chosen_quota is None:
            chosen_bookie = max(offers, key=lambda k: offers[k])
            chosen_quota = offers[chosen_bookie]
        if chosen_quota is None or chosen_quota <= 1.0:
            continue

        edge = prob * chosen_quota - 1.0
        if edge < edge_min:
            continue

        b = chosen_quota - 1.0
        q = 1.0 - prob
        if b <= 0:
            continue
        f_star = (b * prob - q) / b
        if f_star <= 0:
            continue
        f_scaled = min(f_star * kelly_mult, stake_max)
        stake_eur = round(capital * f_scaled, 2)

        picks.append({
            'market': mkt,
            'model_prob': round(prob, 4),
            'bookie': chosen_bookie,
            'quota': round(chosen_quota, 3),
            'implied_prob': round(1.0 / chosen_quota, 4),
            'edge': round(edge, 4),
            'edge_pct': round(edge * 100.0, 2),
            'kelly_raw': round(f_star, 4),
            'stake_pct': round(f_scaled, 4),
            'stake_eur': stake_eur,
            'alt_bookies': {k: round(v, 3) for k, v in offers.items() if k != chosen_bookie},
        })
    picks.sort(key=lambda p: p['edge'], reverse=True)
    return picks

# -------------------- Route registration --------------------

def register(app, adv_data_provider):
    """
    Registra le route di ml_pick sull'app Flask.
    adv_data_provider: callable -> dict (ritorna _ML_CACHE['adv_data'] pronto, buildandolo se serve).
    """

    @app.route('/api/ml-env-check')
    def api_ml_env_check():
        """Verifica presenza APISPORTS_KEY senza esporre il valore."""
        return jsonify({
            'apisports_key_set': bool(APISPORTS_KEY),
            'apisports_key_len': len(APISPORTS_KEY) if APISPORTS_KEY else 0,
            'apisports_base': APISPORTS_BASE,
        })

    @app.route('/api/ml-live-fixtures-af')
    def api_ml_live_fixtures_af():
        """Lista partite live (da API-Football)."""
        data = _get_live_fixtures_af()
        if not isinstance(data, dict):
            return jsonify({'error': 'bad response'}), 502
        if 'error' in data:
            return jsonify({'error': data.get('error'), 'details': data}), 502
        out = []
        for f in data.get('response', []) or []:
            ctx = _fixture_to_model_ctx(f)
            if ctx:
                out.append(ctx)
        return jsonify({
            'fixtures': out,
            'count': len(out),
            'updated_at': int(time.time()),
            'api_errors': data.get('errors') or None,
        })

    @app.route('/api/ml-odds-debug')
    def api_ml_odds_debug():
        """Debug: quote raw + parsing per una fixture."""
        fixture = request.args.get('fixture', type=int)
        if not fixture:
            return jsonify({'error': 'missing fixture param'}), 400
        data = _get_live_odds(fixture)
        parsed = _parse_odds_payload(data) if isinstance(data, dict) else {}
        bookies = set()
        bet_names = set()
        resps = (data.get('response') or []) if isinstance(data, dict) else []
        for r in resps:
            for b in r.get('bookmakers', []) or []:
                bookies.add(b.get('name', ''))
                for bt in b.get('bets', []) or []:
                    bet_names.add(bt.get('name', ''))
        return jsonify({
            'fixture': fixture,
            'response_count': len(resps),
            'bookies_seen': sorted(list(bookies)),
            'bet_names_seen': sorted(list(bet_names)),
            'n_markets_parsed': len(parsed),
            'parsed_markets': parsed,
            'api_errors': (data.get('errors') if isinstance(data, dict) else None),
        })

    @app.route('/api/ml-pick')
    def api_ml_pick():
        """
        Calcola top pick con edge positivo + Kelly per una fixture live.
        Params:
          fixture  (int, required): id fixture API-Football
          bookie   (str, default 'betfair'): bookie preferito per edge
          kelly    (float, default 0.25): frazione Kelly
          edge_min (float, default 0.03): edge minimo 3%
          stake_max(float, default 0.05): cap stake 5%
          capital  (float, default 1000): bankroll EUR
        """
        fixture = request.args.get('fixture', type=int)
        if not fixture:
            return jsonify({'error': 'missing fixture param'}), 400
        bookie = (request.args.get('bookie') or 'betfair').lower().strip()
        kelly_mult = request.args.get('kelly', default=0.25, type=float)
        edge_min = request.args.get('edge_min', default=0.03, type=float)
        stake_max = request.args.get('stake_max', default=0.05, type=float)
        capital = request.args.get('capital', default=1000.0, type=float)

        # 1) Fixture live
        live_data = _get_live_fixtures_af()
        ctx = None
        if isinstance(live_data, dict):
            for f in (live_data.get('response') or []):
                if ((f.get('fixture') or {}).get('id')) == fixture:
                    ctx = _fixture_to_model_ctx(f)
                    break
        if not ctx:
            return jsonify({
                'error': 'fixture not live or not found',
                'fixture': fixture,
                'picks': [],
            }), 404

        # 2) Model data
        try:
            adv_data = adv_data_provider()
        except Exception as e:
            return jsonify({'error': 'model data unavailable: ' + str(e)}), 500
        lg_key, lg_data = _find_league_in_model(
            adv_data, ctx['league_id'], ctx['league_name'], ctx['country']
        )
        if not lg_data:
            return jsonify({
                'warning': 'league not covered by model',
                'league_id': ctx['league_id'],
                'league_name': ctx['league_name'],
                'country': ctx['country'],
                'ctx': ctx,
                'picks': [],
                'available_leagues_sample': list((adv_data.get('leagues') or {}).keys())[:10],
            })

        # 3) Model probs
        probs, source = _extract_probs(lg_data, ctx['minute'], ctx['score_home'], ctx['score_away'])

        # 4) Live odds
        odds_data = _get_live_odds(fixture)
        odds_by_market = _parse_odds_payload(odds_data) if isinstance(odds_data, dict) else {}

        # 5) Compute picks
        picks = _compute_picks(probs, odds_by_market, bookie, kelly_mult, edge_min, stake_max, capital)

        return jsonify({
            'fixture': fixture,
            'ctx': ctx,
            'model_league_key': lg_key,
            'model_source': source,
            'n_markets_with_odds': len(odds_by_market),
            'params': {
                'bookie_pref': bookie,
                'kelly': kelly_mult,
                'edge_min': edge_min,
                'stake_max': stake_max,
                'capital': capital,
            },
            'picks': picks,
        })
