#!/usr/bin/env python3
"""
rebuild_from_html.py - Rigenera football.db.gz dai file HTML in data/.

Pipeline:
  1. Decomprime football.db.gz esistente (se c'e') -> football.db,
     cosi' eventuali tabelle NON derivate dagli HTML (es. ML/odds runtime)
     vengono preservate.
  2. Ricostruisce le tabelle leagues + matches dai file data/*_gol-16min.html
     tramite migrate.run() (che fa DROP+CREATE solo di quelle due tabelle).
  3. Ricomprime football.db -> football.db.gz.

Usato da .github/workflows/rebuild-db.yml e lanciabile in locale:
    python3 scripts/rebuild_from_html.py
"""
import gzip
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))   # per importare migrate.py dalla root del repo

# Su Windows lo stdout e' cp1252 e i print Unicode di migrate.py crashano.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8')
    except Exception:
        pass

import migrate  # noqa: E402

DB = ROOT / 'football.db'
GZ = ROOT / 'football.db.gz'
DATA = ROOT / 'data'


def main():
    if GZ.exists() and not DB.exists():
        print('Decompressione %s -> %s ...' % (GZ.name, DB.name))
        with gzip.open(GZ, 'rb') as f_in, open(DB, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    migrate.DB_PATH = DB
    migrate.DATA_DIR = DATA
    migrate.run()

    print('Compressione %s -> %s ...' % (DB.name, GZ.name))
    with open(DB, 'rb') as f_in, gzip.open(GZ, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    print('[OK] %s aggiornato (%d KB)' % (GZ.name, GZ.stat().st_size // 1024))


if __name__ == '__main__':
    main()
