#!/usr/bin/env python3
"""archive_write.py - prende il JSON di /api/earlygoal-sync (argv[1]) e scrive le
righe gol-16min nei file data/<league_id>_gol-16min.html, con dedup. Usato dal
workflow archive-tick.yml dopo la chiamata all'endpoint."""
import json, re, sys
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

def keys_in(c):
    body = re.sub(r'<thead>.*?</thead>', '', c, flags=re.DOTALL)
    k = set()
    for r in re.findall(r'<tr>\s*(.*?)\s*</tr>', body, re.DOTALL):
        cl = re.findall(r'<td[^>]*>(.*?)</td>', r, re.DOTALL)
        if len(cl) >= 5:
            c2 = [re.sub(r'<[^>]+>', '', x).strip() for x in cl]
            k.add((c2[0], c2[1], c2[3], c2[4]))
    return k

def main():
    if len(sys.argv) < 2:
        print("uso: archive_write.py <sync.json>"); return
    try:
        payload = json.load(open(sys.argv[1], encoding="utf-8"))
    except Exception as e:
        print("JSON non leggibile:", e); return
    matches = payload.get("matches", []) if isinstance(payload, dict) else []
    if not matches:
        print("nessuna nuova partita da scrivere"); return
    by = {}
    for m in matches:
        by.setdefault(m["league_id"], []).append(m["row"])
    total = 0
    for lid, rows in by.items():
        f = DATA / ("%d_gol-16min.html" % lid)
        if not f.exists():
            print("  file mancante per lega %s - salto" % lid); continue
        c = f.read_text(encoding="utf-8", errors="replace")
        exist = keys_in(c)
        add = []
        for r in rows:
            cl = re.findall(r'<td[^>]*>(.*?)</td>', r, re.DOTALL)
            c2 = [re.sub(r'<[^>]+>', '', x).strip() for x in cl]
            key = (c2[0], c2[1], c2[3], c2[4]) if len(c2) >= 5 else None
            if key is None or key in exist:
                continue
            exist.add(key); add.append(r)
        if add:
            block = "\n" + "\n".join(add) + "\n"
            c = c.replace("</table>", block + "</table>", 1)
            f.write_text(c, encoding="utf-8")
            total += len(add)
            print("  lega %s: +%d righe" % (lid, len(add)))
    print("TOTALE righe scritte su data/:", total)

if __name__ == "__main__":
    main()
