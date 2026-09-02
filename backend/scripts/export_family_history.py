# -*- coding: utf-8 -*-
"""
export_family_history.py - xuat lai frontend/public/sale_dataset_ver_2.csv
(file bi thieu trong repo -> cac phan lich su cua ProductAnalysis khong co du lieu)
Dong: date, family, unit_sales, perishable  (tong toan chuoi, tu historical_sales)
Chay: python backend/scripts/export_family_history.py
"""
import sys
import csv
import time
import sqlite3
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DB_PATH = Path(__file__).resolve().parent.parent / "src" / "database" / "retail.db"
OUT = Path(__file__).resolve().parent.parent.parent / "frontend" / "public" / "sale_dataset_ver_2.csv"

t0 = time.time()
conn = sqlite3.connect(str(DB_PATH))
conn.execute("PRAGMA cache_size=-256000")
OUT.parent.mkdir(parents=True, exist_ok=True)

cur = conn.execute("""
    SELECT h.date,
           i.family,
           ROUND(SUM(h.unit_sales), 1) AS unit_sales,
           MAX(i.perishable)           AS perishable
    FROM historical_sales h
    JOIN items i ON i.item_nbr = h.item_nbr
    GROUP BY h.date, i.family
    ORDER BY h.date, i.family
""")

n = 0
with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["date", "family", "unit_sales", "perishable"])
    while True:
        rows = cur.fetchmany(5000)
        if not rows:
            break
        w.writerows(rows)
        n += len(rows)
conn.close()
print(f"Xong: {n:,} dong -> {OUT} ({time.time()-t0:.0f}s)")
