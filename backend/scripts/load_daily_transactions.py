# -*- coding: utf-8 -*-
"""
load_daily_transactions.py - nap so giao dich (≈ so hoa don) theo ngay x cua hang
tu file raw Kaggle transactions.csv vao bang daily_transactions cua retail.db.
Chay: python backend/scripts/load_daily_transactions.py
"""
import sys
import csv
import sqlite3
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DB_PATH = Path(__file__).resolve().parent.parent / "src" / "database" / "retail.db"
SRC = Path(__file__).resolve().parent.parent.parent / "ml_training" / "data" / "raw" / "transactions.csv"

if not SRC.exists():
    raise SystemExit(f"Khong tim thay nguon: {SRC}")

conn = sqlite3.connect(str(DB_PATH))
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("""CREATE TABLE IF NOT EXISTS daily_transactions (
                    date TEXT NOT NULL,
                    store_nbr INTEGER NOT NULL,
                    n_invoices INTEGER NOT NULL,
                    PRIMARY KEY (date, store_nbr))""")

n = 0
with open(SRC, newline="", encoding="utf-8") as f:
    rd = csv.DictReader(f)
    batch = []
    for row in rd:
        d = (row["date"].strip(), int(row["store_nbr"]), int(row["transactions"]))
        batch.append(d)
        if len(batch) >= 5000:
            conn.executemany("INSERT OR REPLACE INTO daily_transactions VALUES (?,?,?)", batch)
            n += len(batch); batch = []
    if batch:
        conn.executemany("INSERT OR REPLACE INTO daily_transactions VALUES (?,?,?)", batch)
        n += len(batch)
conn.commit()

lo, hi, tot = conn.execute(
    "SELECT MIN(date), MAX(date), SUM(n_invoices) FROM daily_transactions").fetchone()
conn.close()
print(f"Xong: {n:,} dong ({lo} -> {hi}), tong {tot:,} giao dich")
