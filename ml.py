# ml.py - ML predictor: (1) early-goal legacy + (2) minute+score live-betting
# Laplace smoothing con alpha ADATTIVO — FILTRO training: solo 1 gol <= 16'
from flask import jsonify, send_from_directory, request
from datetime import datetime
import time
import re
import ml_pick  # live betting picks: API-Football odds + Kelly
import odds_logger  # live odds snapshot logger for historical dataset
import ml_poisson  # Poisson bivariate live engine: correct score, HTS/ATS, mercati gol
import predictions_settlement  # Settlement pipeline: popola predictions_log con risultati FT

EARLY_GOAL_MAX_MIN = 16  # filtro training=deploy: 1 gol nei primi N min

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

# Snapshot minuti - granularita' fine pre-45' per gol early
_SNAP_MINUTES = [16, 25, 35, 45, 60, 70, 80]

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


def _adaptive_alpha(n, base=3.0, scale=60.0, min_alpha=0.5):
    """Alpha adattivo: piu' campioni = meno regressione. n=0->3.0, n=60->1.5, n->inf->0.5"""
    return max(min_alpha, base * scale / (n + scale))

def _shrink(child_n, child_p, parent_p, keys, alpha=None):
    if alpha is None:
        alpha = _adaptive_alpha(child_n)
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
          AND m.first_goal_minute IS NOT NULL AND m.first_goal_minute <= 16
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
        _alpha_lg = _adaptive_alpha(len(rows_lg))
        leagues_out[lg] = {
            'n': len(rows_lg),
            'confidence': round((1.0 - _alpha_lg / (len(rows_lg) + _alpha_lg)) * 100, 1),
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
          AND m.first_goal_minute IS NOT NULL AND m.first_goal_minute <= 16
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

        _alpha_lg2 = _adaptive_alpha(len(matches))
        leagues_out[lg] = {
            'n': len(matches),
            'confidence': round((1.0 - _alpha_lg2 / (len(matches) + _alpha_lg2)) * 100, 1),
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


_EARLY_GOALS_HTML = """<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<title>Early Goal History - Rei-Rei</title>
<style>
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0d1117;color:#c9d1d9;margin:0;padding:20px}
h1{color:#58a6ff;margin:0 0 16px;font-size:22px}
.nav{margin-bottom:16px;font-size:13px}
.nav a{color:#58a6ff;text-decoration:none}
.controls{display:flex;align-items:center;gap:12px;margin-bottom:16px;flex-wrap:wrap}
select{background:#161b22;color:#c9d1d9;border:1px solid #30363d;border-radius:6px;padding:6px 12px;font-size:14px}
.stats-row{display:flex;gap:12px;margin-bottom:16px;flex-wrap:wrap}
.stat-box{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:10px 16px;text-align:center;min-width:90px}
.stat-val{font-size:20px;font-weight:700;color:#3fb950}
.stat-lbl{font-size:11px;color:#8b949e;margin-top:2px}
table{width:100%;border-collapse:collapse;font-size:13px}
th{background:#161b22;color:#8b949e;padding:8px 10px;text-align:left;border-bottom:1px solid #30363d;cursor:pointer;user-select:none;white-space:nowrap}
th:hover{color:#c9d1d9}
td{padding:7px 10px;border-bottom:1px solid #21262d}
tr:hover td{background:#161b22}
.badge-home{background:#1f6feb33;color:#58a6ff;border:1px solid #1f6feb;border-radius:12px;padding:2px 8px;font-size:11px;white-space:nowrap}
.badge-away{background:#9e6a0333;color:#d29922;border:1px solid #9e6a03;border-radius:12px;padding:2px 8px;font-size:11px;white-space:nowrap}
.res-1{color:#3fb950;font-weight:700}.res-x{color:#8b949e;font-weight:700}.res-2{color:#f85149;font-weight:700}
.btts-si{color:#3fb950}.btts-no{color:#8b949e}
.refresh-info{font-size:11px;color:#8b949e;margin-left:auto}
.empty{color:#8b949e;padding:32px;text-align:center}
.spinner{display:inline-block;width:14px;height:14px;border:2px solid #30363d;border-top-color:#58a6ff;border-radius:50%;animation:spin .8s linear infinite;vertical-align:middle;margin-right:6px}
@keyframes spin{to{transform:rotate(360deg)}}
</style></head>
<body>
<div class="nav"><a href="/picks">&larr; Live Picks</a> &middot; <a href="/ml">ML Stats</a></div>
<h1>&#x23F0; Early Goal History</h1>
<div class="controls">
  <select id="lgSel"><option value="">-- Seleziona campionato --</option></select>
  <span class="refresh-info" id="rInfo">Auto-refresh 90s</span>
</div>
<div class="stats-row" id="sRow"></div>
<div id="tWrap"><div class="empty">Seleziona un campionato per vedere le ultime 30 partite con primo gol nei primi 16 minuti</div></div>
<script>
var _sc='date_utc',_sd=-1,_data=[],_timer=null,_cd=90;
async function loadLeagues(){
  var s=document.getElementById('lgSel');
  try{var r=await fetch('/api/eg-leagues'),d=await r.json();
    (d.leagues||[]).forEach(function(l){var o=new Option(l.name+(l.n?' ('+l.n+')':''),l.name);s.add(o);});
  }catch(e){}
}
async function loadMatches(lg){
  if(!lg)return;
  document.getElementById('tWrap').innerHTML='<div class="empty"><span class="spinner"></span>Caricamento...</div>';
  document.getElementById('sRow').innerHTML='';
  try{
    var r=await fetch('/api/last-eg-matches?league='+encodeURIComponent(lg)+'&limit=30');
    var d=await r.json();_data=d.matches||[];renderStats();renderTable();
  }catch(e){document.getElementById('tWrap').innerHTML='<div class="empty">Errore dati</div>';}
}
function renderStats(){
  var ms=_data,n=ms.length;if(!n){document.getElementById('sRow').innerHTML='';return;}
  var h1=0,hX=0,h2=0,ov=0,bt=0,tot=0;
  ms.forEach(function(m){
    if(m.result==='1')h1++;else if(m.result==='X')hX++;else h2++;
    var g=parseInt(m.ft_home||0)+parseInt(m.ft_away||0);
    if(g>2)ov++;if(parseInt(m.ft_home||0)>0&&parseInt(m.ft_away||0)>0)bt++;
    tot+=parseInt(m.first_goal_minute||0);
  });
  function p(v){return Math.round(v/n*100)+'%';}
  var avgm=n>0?(tot/n).toFixed(1):'-';
  var boxes=[['1/X/2',p(h1)+'/'+p(hX)+'/'+p(h2)],['Over 2.5',p(ov)],['BTTS',p(bt)],['Avg gol',avgm+"'"],['N',n]];
  document.getElementById('sRow').innerHTML=boxes.map(function(b){
    return '<div class="stat-box"><div class="stat-val">'+b[1]+'</div><div class="stat-lbl">'+b[0]+'</div></div>';
  }).join('');
}
function renderTable(){
  var ms=_data.slice().sort(function(a,b){
    var va=a[_sc],vb=b[_sc];
    if(_sc==='first_goal_minute'){va=parseInt(va||99);vb=parseInt(vb||99);}
    else if(_sc==='total'){va=parseInt(a.ft_home||0)+parseInt(a.ft_away||0);vb=parseInt(b.ft_home||0)+parseInt(b.ft_away||0);}
    return va<vb?-_sd:va>vb?_sd:0;
  });
  if(!ms.length){document.getElementById('tWrap').innerHTML='<div class="empty">Nessuna partita trovata</div>';return;}
  function th(col,lbl){return '<th data-col="'+col+'">'+lbl+(_sc===col?(_sd>0?' &#x25B2;':' &#x25BC;'):'')+'</th>';}
  var h='<table id="egTbl"><thead><tr>'+th('date_utc','Data')+th('home','Casa')+'<th>FT</th><th>HT</th>'+th('away','Ospite')+th('first_goal_minute','1gol')+th('total','Tot')+'<th>BTTS</th>'+th('result','Ris.')+'</tr></thead><tbody>';
  ms.forEach(function(m){
    var ft=m.ft_home!=null?m.ft_home+'-'+m.ft_away:'-';
    var ht=m.ht_home!=null?m.ht_home+'-'+m.ht_away:'-';
    var tot=m.ft_home!=null?parseInt(m.ft_home||0)+parseInt(m.ft_away||0):'-';
    var bt=m.ft_home!=null?(parseInt(m.ft_home)>0&&parseInt(m.ft_away)>0?'<span class="btts-si">SI</span>':'<span class="btts-no">NO</span>'):'-';
    var rc=m.result==='1'?'res-1':m.result==='X'?'res-x':'res-2';
    var fg=m.first_goal_team==='home'?'<span class="badge-home">Casa '+(m.first_goal_minute||'?')+"'</span>":m.first_goal_team==='away'?'<span class="badge-away">Osp '+(m.first_goal_minute||'?')+"'</span>":(m.first_goal_minute||'?')+"'";
    h+='<tr><td>'+(m.date_utc||'-')+'</td><td>'+(m.home||'N/D')+'</td><td><b>'+ft+'</b></td><td style="color:#8b949e">'+ht+'</td><td>'+(m.away||'N/D')+'</td><td>'+fg+'</td><td>'+tot+'</td><td>'+bt+'</td><td class="'+rc+'">'+(m.result||'-')+'</td></tr>';
  });
  document.getElementById('tWrap').innerHTML=h+'</tbody></table>';
  document.getElementById('egTbl').addEventListener('click',function(e){
    var th=e.target.closest('th[data-col]');
    if(th){if(_sc===th.dataset.col)_sd*=-1;else{_sc=th.dataset.col;_sd=-1;}renderTable();}
  });
}
function startTimer(){
  if(_timer)clearInterval(_timer);_cd=90;
  _timer=setInterval(function(){
    _cd--;document.getElementById('rInfo').textContent='Auto-refresh '+_cd+'s';
    if(_cd<=0){var lg=document.getElementById('lgSel').value;if(lg)loadMatches(lg);_cd=90;}
  },1000);
}
document.getElementById('lgSel').addEventListener('change',function(){if(this.value){loadMatches(this.value);startTimer();}});
loadLeagues();startTimer();
</script>
</body>
</html>"""

# âââââââââââââââââââ Registrazione route âââââââââââââââââââ

def register(app, query_fn):
    """Registra le route ML sull'app Flask."""


    @app.route('/early-goals')
    def early_goals_page():
        from flask import Response as _R
        return _R(_EARLY_GOALS_HTML, mimetype='text/html')

    @app.route('/api/eg-leagues')
    def api_eg_leagues():
        """Campionati con partite early-goal (primo gol <=16') — da SQLite + Turso."""
        leagues = {}
        try:
            rows = query_fn("""
                SELECT l.name AS name, COUNT(*) AS n
                FROM matches m JOIN leagues l ON l.id = m.league_id
                WHERE m.ft_home IS NOT NULL
                  AND m.first_goal_minute IS NOT NULL AND m.first_goal_minute <= 16
                GROUP BY l.name ORDER BY l.name
            """)
            for r in (rows or []):
                leagues[r['name']] = leagues.get(r['name'], 0) + int(r['n'] or 0)
        except Exception:
            pass
        try:
            import predictions_settlement as _pset
            pl = _pset._turso_select_rows(
                "SELECT DISTINCT league_name FROM predictions_log "
                "WHERE ft_home IS NOT NULL AND first_goal_minute IS NOT NULL AND first_goal_minute <= 16 "
                "ORDER BY league_name"
            ) or []
            for r in pl:
                n = (r.get('league_name') or '').strip()
                if n and n not in leagues:
                    leagues[n] = 0
        except Exception:
            pass
        out = sorted([{'name': k, 'n': v} for k, v in leagues.items()], key=lambda x: x['name'])
        return jsonify({'leagues': out})

    @app.route('/api/last-eg-matches')
    def api_last_eg_matches():
        """Ultime N partite con 1° gol <=16' per campionato. Combina Turso + SQLite."""
        league = request.args.get('league', '').strip()
        limit = min(int(request.args.get('limit', 30) or 30), 100)
        matches = []

        # 1) predictions_log (Turso) — ha team names, date, first_goal_team
        try:
            import predictions_settlement as _pset
            lnames = [league]
            if ' - ' in league:
                lnames.append(league.split(' - ', 1)[1])
            _ph = ','.join('?' for _ in lnames)
            pl_rows = _pset._turso_select_rows(
                "SELECT league_name, home_team_name, away_team_name, "
                "home_team_id, away_team_id, "
                "ft_home, ft_away, ht_home, ht_away, "
                "first_goal_minute, first_goal_team_id, date_utc, season "
                "FROM predictions_log "
                "WHERE league_name IN (%s) "
                "AND ft_home IS NOT NULL "
                "AND first_goal_minute IS NOT NULL AND first_goal_minute <= 16 "
                "ORDER BY settled_ts DESC LIMIT ?" % _ph,
                lnames + [limit]
            ) or []
            for r in pl_rows:
                fh = r.get('ft_home'); fa = r.get('ft_away')
                res = ('1' if int(fh or 0) > int(fa or 0) else ('2' if int(fa or 0) > int(fh or 0) else 'X')) if fh is not None else '?'
                _ftid = r.get('first_goal_team_id')
                _htid = r.get('home_team_id')
                _atid = r.get('away_team_id')
                _fgt = 'home' if (_ftid and _htid and int(_ftid)==int(_htid)) else ('away' if (_ftid and _atid and int(_ftid)==int(_atid)) else '')
                matches.append({
                    'home': r.get('home_team_name') or '', 'away': r.get('away_team_name') or '',
                    'ft_home': fh, 'ft_away': fa,
                    'ht_home': r.get('ht_home'), 'ht_away': r.get('ht_away'),
                    'first_goal_minute': r.get('first_goal_minute'),
                    'first_goal_team': _fgt,
                    'date_utc': (r.get('date_utc') or '')[:10],
                    'season': r.get('season'), 'result': res, 'source': 'live',
                })
        except Exception:
            pass

        # 2) SQLite storico (se servono altre partite)
        if len(matches) < limit:
            try:
                sq = query_fn("""
                    SELECT l.name AS league,
                           m.ft_home, m.ft_away, m.ht_home, m.ht_away,
                           m.first_goal_minute, m.first_goal_team,
                           m.result, m.btts, m.season
                    FROM matches m JOIN leagues l ON l.id = m.league_id
                    WHERE l.name = ?
                      AND m.ft_home IS NOT NULL
                      AND m.first_goal_minute IS NOT NULL AND m.first_goal_minute <= 16
                    ORDER BY m.id DESC LIMIT ?
                """, (league, limit - len(matches)))
                for r in (sq or []):
                    fh = r.get('ft_home'); fa = r.get('ft_away')
                    res = r.get('result') or ('1' if int(fh or 0) > int(fa or 0) else ('2' if int(fa or 0) > int(fh or 0) else 'X'))
                    matches.append({
                        'home': '', 'away': '',
                        'ft_home': fh, 'ft_away': fa,
                        'ht_home': r.get('ht_home'), 'ht_away': r.get('ht_away'),
                        'first_goal_minute': r.get('first_goal_minute'),
                        'first_goal_team': r.get('first_goal_team') or '',
                        'date_utc': '', 'season': r.get('season'), 'result': res, 'source': 'history',
                    })
            except Exception:
                pass

        return jsonify({'league': league, 'count': len(matches), 'matches': matches[:limit]})

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


    @app.route('/api/ml-trend')
    def api_ml_trend():
        """Trend training: n_matches e confidence del modello per stagione."""
        league_name = request.args.get('league', '').strip()
        alpha = 3.0
        if league_name:
            rows = query_fn("""
                SELECT COALESCE(m.season, 'N/D') as season, COUNT(*) as n
                FROM matches m JOIN leagues l ON l.id = m.league_id
                WHERE l.name = ? GROUP BY COALESCE(m.season,'N/D')
                ORDER BY COALESCE(m.season,'N/D')
            """, (league_name,))
        else:
            rows = query_fn("""
                SELECT COALESCE(season,'N/D') as season, COUNT(*) as n FROM matches
                GROUP BY COALESCE(season,'N/D') ORDER BY COALESCE(season,'N/D')
            """)
        cumulative = 0
        seasons_out = []
        for row in rows:
            cumulative += row['n']
            confidence = round((1.0 - alpha / (cumulative + alpha)) * 100, 1)
            seasons_out.append({'season': row['season'], 'n_season': row['n'],
                                 'n_cumulative': cumulative, 'confidence': confidence})
        return jsonify({'league': league_name, 'alpha': alpha, 'seasons': seasons_out})

    @app.route('/api/ml-picks-stats')
    def api_ml_picks_stats():
        """Statistiche predizioni per campionato da predictions_log (471+ partite settled)."""
        league_name = request.args.get('league', '').strip()
        try:
            import predictions_settlement as pset
            if league_name:
                # ML usa "Country - League", predictions_log salva solo "League" (API-Football)
                _lnames = [league_name]
                if ' - ' in league_name:
                    _lnames.append(league_name.split(' - ', 1)[1])
                _ph = ','.join('?' for _ in _lnames)
                rows = pset._turso_select_rows(
                    "SELECT season, date_utc, ft_home, ft_away, ht_home, ht_away, first_goal_minute "
                    "FROM predictions_log WHERE league_name IN (%s) AND ft_home IS NOT NULL AND ft_away IS NOT NULL "
                    "ORDER BY date_utc" % _ph,
                    _lnames
                )
            else:
                rows = pset._turso_select_rows(
                    "SELECT season, date_utc, league_name, ft_home, ft_away, ht_home, ht_away, first_goal_minute "
                    "FROM predictions_log WHERE ft_home IS NOT NULL AND ft_away IS NOT NULL ORDER BY date_utc"
                )
        except Exception as e:
            return jsonify({'error': str(e), 'league': league_name, 'seasons': [], 'total': 0})

        if not rows:
            return jsonify({'league': league_name, 'seasons': [], 'rolling': [], 'total': 0})

        def o25(m): ft = (m.get('ft_home') or 0) + (m.get('ft_away') or 0); return None if m.get('ft_home') is None else (1 if ft > 2 else 0)
        def o15(m): ft = (m.get('ft_home') or 0) + (m.get('ft_away') or 0); return None if m.get('ft_home') is None else (1 if ft > 1 else 0)
        def btts(m): return None if m.get('ft_home') is None else (1 if (m.get('ft_home') or 0) > 0 and (m.get('ft_away') or 0) > 0 else 0)
        def eg(m): fgm = m.get('first_goal_minute'); return None if fgm is None else (1 if fgm <= 16 else 0)
        def _rate(ms, fn): vals = [fn(x) for x in ms if fn(x) is not None]; return round(sum(vals)/len(vals)*100,1) if vals else None

        from collections import defaultdict
        by_season = defaultdict(list)
        for r in rows:
            by_season[str(r.get('season') or 'N/D')].append(r)

        seasons_out = []
        cum = 0
        for s in sorted(by_season.keys()):
            ms = by_season[s]; cum += len(ms)
            seasons_out.append({'season': s, 'n': len(ms), 'n_cum': cum,
                'over_1_5': _rate(ms, o15), 'over_2_5': _rate(ms, o25),
                'btts': _rate(ms, btts), 'early_goal': _rate(ms, eg)})

        win = 10
        rolling = []
        for i in range(win - 1, len(rows)):
            batch = rows[max(0, i - win + 1): i + 1]
            rolling.append({'n': i + 1, 'date': (rows[i].get('date_utc') or '')[:10],
                'over_2_5': _rate(batch, o25), 'btts': _rate(batch, btts), 'early_goal': _rate(batch, eg)})

        # Aggiungi ml_acc (win rate da ml_picks_log) per ogni rolling entry
        try:
            ml_rows = pset._turso_select_rows(
                "SELECT date(settled_at,'unixepoch') as d, result FROM ml_picks_log "
                "WHERE league_name IN (%s) AND result IS NOT NULL ORDER BY settled_at" % _ph,
                _lnames
            )
            _daily = {}
            for r in ml_rows:
                d = (r.get('d') or '')[:10]
                if d:
                    if d not in _daily: _daily[d] = [0, 0]
                    _daily[d][1] += 1
                    if r.get('result') == 'WIN': _daily[d][0] += 1
            _sorted_d = sorted(_daily.keys())
            _cum_w, _cum_n = 0, 0
            _acc_by_date = {}
            for d in _sorted_d:
                _cum_w += _daily[d][0]; _cum_n += _daily[d][1]
                _acc_by_date[d] = round(_cum_w / _cum_n * 100, 1) if _cum_n > 0 else None
            for entry in rolling:
                rd = entry.get('date', '')
                matching = [d for d in _sorted_d if d <= rd]
                entry['ml_acc'] = _acc_by_date.get(max(matching)) if matching else None
        except Exception:
            for entry in rolling:
                entry['ml_acc'] = None
        
        return jsonify({'league': league_name, 'total': len(rows),
                        'seasons': seasons_out, 'rolling': rolling[-50:]})

    @app.route('/api/ml-picks-leagues')
    def api_ml_picks_leagues():
        """Campionati in predictions_log (nomi reali da API-Football)."""
        try:
            import predictions_settlement as pset
            rows = pset._turso_select_rows(
                "SELECT league_name, COUNT(*) as n FROM predictions_log "
                "WHERE ft_home IS NOT NULL AND league_name IS NOT NULL "
                "GROUP BY league_name ORDER BY n DESC"
            )
            return jsonify({'leagues': [{'name': r['league_name'], 'n': r['n']} for r in rows if r.get('league_name')]})
        except Exception as e:
            return jsonify({'error': str(e), 'leagues': []})

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
