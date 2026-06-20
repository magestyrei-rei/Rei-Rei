#!/usr/bin/env python3
"""Backfill storico di UN campionato chiamando l'endpoint /api/backfill-league su
Render (che usa la chiave api-sports gia' presente li'), stagione per stagione, e
scrive data/<id>_gol-16min.html nel formato standard.

Uso:  python scripts/backfill_league.py <league_id> <season_start> <season_end>
Env:  TICK_AUTH_TOKEN (obbligatorio), BACKFILL_BASE (default https://rei-rei.onrender.com)
"""
import os, sys, json, time, urllib.request, urllib.parse

BASE = os.environ.get("BACKFILL_BASE", "https://rei-rei.onrender.com").rstrip("/")
TOKEN = os.environ.get("TICK_AUTH_TOKEN", "")

STYLE = ('body{font-family:Arial,sans-serif;margin:24px;background:#f5f7fb;color:#172033}'
         'table{width:100%;border-collapse:collapse;background:#fff}'
         'th,td{border:1px solid #d9e0ee;padding:8px 10px;text-align:left;vertical-align:top;font-size:13px}'
         'th{background:#1f3b73;color:#fff}.away-goal{color:#d11a2a;font-weight:700}'
         'tr:nth-child(even) td{background:#f8faff}')
THEAD = ('<tr><th>Stagione</th><th>Data</th><th>Ora</th><th>Casa</th><th>Ospite</th>'
         '<th>Campionato</th><th>1T</th><th>2T</th><th>Finale</th><th>Minuti gol</th>'
         '<th>Quota 1</th><th>Quota X</th><th>Quota 2</th></tr>')


def fetch_season(lid, season):
    url = BASE + "/api/backfill-league?" + urllib.parse.urlencode(
        {"league": lid, "season": season, "token": TOKEN})
    with urllib.request.urlopen(url, timeout=180) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    if len(sys.argv) < 4:
        print("uso: backfill_league.py <league_id> <season_start> <season_end>")
        sys.exit(2)
    if not TOKEN:
        print("ERRORE: TICK_AUTH_TOKEN non impostato")
        sys.exit(2)
    lid = int(sys.argv[1]); s0 = int(sys.argv[2]); s1 = int(sys.argv[3])
    rows = {}            # (data, casa, ospite) -> riga html (dedup)
    league_name = None
    for season in range(s0, s1 + 1):
        try:
            d = fetch_season(lid, season)
        except Exception as e:
            print("stagione %s: ERRORE %s (riprovo una volta)" % (season, e))
            time.sleep(5)
            try:
                d = fetch_season(lid, season)
            except Exception as e2:
                print("stagione %s: fallita %s" % (season, e2)); continue
        if d.get("error"):
            print("stagione %s: %s" % (season, d["error"])); continue
        if d.get("league_name"):
            league_name = d["league_name"]
        for it in d.get("rows", []):
            rows[(it["date"], it["home"], it["away"])] = it["row"]
        print("stagione %s -> %s righe (totale %s)" % (season, d.get("count", 0), len(rows)))
        time.sleep(1)
    if not rows:
        print("NESSUNA riga trovata: non scrivo nulla.")
        sys.exit(1)
    league_name = league_name or ("Lega %d" % lid)
    body = "\n".join(rows[k] for k in sorted(rows))
    html = (
        '<!DOCTYPE html>\n<html lang="it">\n<head>\n<meta charset="utf-8">\n'
        '<title>%s - Primo gol entro 16 minuti</title>\n<style>%s</style>\n</head>\n<body>\n'
        '<h1>%s - Primo gol entro il 16 minuto</h1>\n<table>\n<thead>\n%s\n</thead>\n<tbody>\n%s\n</tbody>\n</table>\n</body>\n</html>\n'
    ) % (league_name, STYLE, league_name, THEAD, body)
    out = "data/%d_gol-16min.html" % lid
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print("scritto %s : %d partite, campionato '%s'" % (out, len(rows), league_name))


if __name__ == "__main__":
    main()
