# -*- coding: utf-8 -*-
"""
build_business_cache.py (chạy 1 lần sau khi có historical_sales)
Dựng bảng agg_daily_business(date, store_nbr, revenue, returns, cogs) - USD
từ historical_sales × items × family_prices (GIÁ THAM CHIẾU, không phải giá thật).
Đồng bộ bộ số với FAMILY_ECONOMICS phía frontend.

Chạy: python backend/scripts/build_business_cache.py
"""
import sys
import time
import sqlite3
import logging
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_business_cache")

ENV_DB = os.environ.get("DB_PATH") if (os := __import__("os")) else None
DB_PATH = Path(ENV_DB) if ENV_DB else (
    Path(__file__).resolve().parent.parent / "src" / "database" / "retail.db")

# family, unit_price, cost_ratio, return_rate  -- khớp FAMILY_ECONOMICS (App.tsx)
FAMILY_PRICES = [
    ("GROCERY I", 1.4, 0.80, 0.015), ("GROCERY II", 1.6, 0.78, 0.02),
    ("BEVERAGES", 1.2, 0.74, 0.01), ("CLEANING", 2.3, 0.72, 0.015),
    ("PRODUCE", 1.0, 0.82, 0.04), ("DAIRY", 1.5, 0.80, 0.03),
    ("BREAD/BAKERY", 0.9, 0.70, 0.03), ("POULTRY", 3.4, 0.84, 0.02),
    ("MEATS", 4.2, 0.84, 0.02), ("DELI", 3.6, 0.80, 0.02),
    ("EGGS", 2.0, 0.82, 0.03), ("FROZEN FOODS", 2.6, 0.76, 0.02),
    ("PREPARED FOODS", 2.3, 0.72, 0.03), ("SEAFOOD", 5.0, 0.85, 0.04),
    ("LIQUOR,WINE,BEER", 4.5, 0.72, 0.005), ("PERSONAL CARE", 2.6, 0.68, 0.02),
    ("HOME CARE", 3.1, 0.70, 0.015), ("BEAUTY", 3.4, 0.65, 0.03),
    ("HOME AND KITCHEN I", 6.5, 0.68, 0.03), ("HOME AND KITCHEN II", 5.5, 0.68, 0.03),
    ("HOME APPLIANCES", 28, 0.80, 0.04), ("PLAYERS AND ELECTRONICS", 35, 0.84, 0.05),
    ("HARDWARE", 4.8, 0.72, 0.02), ("PET SUPPLIES", 3.2, 0.72, 0.02),
    ("SCHOOL AND OFFICE SUPPLIES", 2.4, 0.66, 0.02), ("BABY CARE", 3.3, 0.72, 0.03),
    ("CELEBRATION", 2.8, 0.62, 0.04), ("LADIESWEAR", 7.5, 0.60, 0.06),
    ("LINGERIE", 5.5, 0.58, 0.06), ("LAWN AND GARDEN", 6.0, 0.66, 0.03),
    ("BOOKS", 4.2, 0.62, 0.02), ("MAGAZINES", 1.8, 0.55, 0.02),
    ("ABRASIVES", 2.1, 0.68, 0.02),
]
DEF_PRICE, DEF_COST, DEF_RET = 1.8, 0.78, 0.02


def main():
    t0 = time.time()
    if not DB_PATH.exists():
        raise SystemExit(f"Khong tim thay DB: {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-256000")  # 256MB cho pass quet lon

    logger.info(">>> Ghi bang gia tham chieu family_prices...")
    conn.execute("""CREATE TABLE IF NOT EXISTS family_prices (
                        family TEXT PRIMARY KEY,
                        unit_price REAL NOT NULL,
                        cost_ratio REAL NOT NULL,
                        return_rate REAL NOT NULL)""")
    conn.executemany(
        "INSERT OR REPLACE INTO family_prices VALUES (?,?,?,?)",
        [(f, p, c, r) for f, p, c, r in FAMILY_PRICES])
    conn.commit()

    logger.info(">>> Dung lai agg_daily_business (dung bang tam _new roi swap nguyen tu)...")
    # Dung bang _new truoc roi moi DROP + RENAME de API khong bao gio thay
    # bang trang hoac thieu bang trong luc quet.
    conn.execute("DROP TABLE IF EXISTS agg_daily_business_new")
    conn.execute("""CREATE TABLE agg_daily_business_new (
                        date TEXT NOT NULL,
                        store_nbr INTEGER NOT NULL,
                        revenue REAL NOT NULL,
                        returns REAL NOT NULL,
                        cogs REAL NOT NULL,
                        PRIMARY KEY (date, store_nbr))""")
    # Gia von tinh tren so ban rong (sau tra hang) - giong cong thuc frontend
    conn.execute(f"""
        INSERT INTO agg_daily_business_new
        SELECT h.date,
               h.store_nbr,
               SUM(h.unit_sales * COALESCE(fp.unit_price, {DEF_PRICE})),
               SUM(h.unit_sales * COALESCE(fp.unit_price, {DEF_PRICE})
                              * COALESCE(fp.return_rate, {DEF_RET})),
               SUM(h.unit_sales * (1 - COALESCE(fp.return_rate, {DEF_RET}))
                              * COALESCE(fp.unit_price, {DEF_PRICE})
                              * COALESCE(fp.cost_ratio, {DEF_COST}))
        FROM historical_sales h
        LEFT JOIN items i ON i.item_nbr = h.item_nbr
        LEFT JOIN family_prices fp ON UPPER(fp.family) = UPPER(COALESCE(i.family, ''))
        GROUP BY h.date, h.store_nbr
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_biz_new_store_date ON agg_daily_business_new(store_nbr, date)")
    conn.commit()
    conn.execute("DROP TABLE IF EXISTS agg_daily_business")
    conn.execute("ALTER TABLE agg_daily_business_new RENAME TO agg_daily_business")
    conn.commit()
    # (Index idx_biz_new_store_date duoc RENAME keo theo bang, khong can tao lai)
    conn.commit()

    n, lo, hi = conn.execute(
        "SELECT COUNT(*), MIN(date), MAX(date) FROM agg_daily_business").fetchone()
    sample = conn.execute(
        "SELECT date, store_nbr, ROUND(revenue,2), ROUND(returns,2), ROUND(cogs,2) "
        "FROM agg_daily_business ORDER BY date LIMIT 3").fetchall()
    conn.close()

    logger.info("agg_daily_business: %s dong (%s -> %s)", f"{n:,}", lo, hi)
    for s in sample:
        logger.info("  mau: %s", s)
    logger.info("HOAN TAT trong %.0fs. API /api/metrics/history san sang.", time.time() - t0)


if __name__ == "__main__":
    main()
