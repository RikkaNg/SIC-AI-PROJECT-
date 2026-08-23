"""
build_sales_cache.py
Dựng bảng tổng hợp doanh số lịch sử theo (store_nbr, item_nbr) từ historical_sales.

Bảng `agg_item_store_sales` giúp các API sản phẩm (/api/top-products, /api/products,
/api/family-mix) chạy tức thì thay vì GROUP BY trực tiếp trên ~59 triệu dòng
historical_sales (quá chậm cho request thời gian thực).

- Dùng index idx_hist_store_date: quét riêng từng cửa hàng, commit sau mỗi store
  để có tiến độ và có thể chạy lại an toàn (idempotent).
- Chạy lại bất cứ lúc nào để làm mới:  python backend/scripts/build_sales_cache.py
"""
import logging
import sqlite3
import sys
import time
from pathlib import Path

# Cho phép chạy trực tiếp: python backend/scripts/build_sales_cache.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_sales_cache")

DB_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "database" / "retail.db"
)


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS agg_item_store_sales (
            store_nbr INTEGER NOT NULL,
            item_nbr  INTEGER NOT NULL,
            unit_sales REAL NOT NULL DEFAULT 0,
            PRIMARY KEY (store_nbr, item_nbr)
        ) WITHOUT ROWID;
        CREATE INDEX IF NOT EXISTS idx_agg_item ON agg_item_store_sales(item_nbr);

        CREATE TABLE IF NOT EXISTS agg_forecast_date_family (
            store_nbr INTEGER NOT NULL,
            date      TEXT NOT NULL,
            family    TEXT NOT NULL,
            predicted_sales REAL NOT NULL DEFAULT 0,
            PRIMARY KEY (store_nbr, date, family)
        ) WITHOUT ROWID;
    """)
    conn.commit()


def build(conn: sqlite3.Connection) -> None:
    stores = [int(r[0]) for r in conn.execute(
        "SELECT DISTINCT store_nbr FROM historical_sales ORDER BY store_nbr"
    ).fetchall()]
    logger.info(f"Sẽ tổng hợp doanh số 2016 cho {len(stores)} cửa hàng ...")

    started = time.perf_counter()
    total_rows = 0
    for i, store in enumerate(stores, 1):
        t0 = time.perf_counter()
        conn.execute("DELETE FROM agg_item_store_sales WHERE store_nbr = ?", (store,))
        cur = conn.execute("""
            INSERT INTO agg_item_store_sales (store_nbr, item_nbr, unit_sales)
            SELECT ?, item_nbr, SUM(unit_sales)
            FROM historical_sales
            WHERE store_nbr = ?
            GROUP BY item_nbr
        """, (store, store))
        conn.commit()
        total_rows += cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
        logger.info(f"[{i:>2}/{len(stores)}] Store {store:>2}: {cur.rowcount:>5} SKU "
                    f"({time.perf_counter() - t0:5.1f}s)")

    logger.info(f"HOÀN THÀNH: {total_rows:,} dòng (store × item) "
                f"trong {time.perf_counter() - started:.0f}s")


def build_forecast_aggregate(conn: sqlite3.Connection) -> None:
    """
    Tổng hợp dự báo theo (store_nbr, date, family) từ bảng forecasts JOIN items.
    Kết quả chỉ ~28k dòng nhưng giúp /api/family-trend phản hồi tức thì
    thay vì quét 3.37 triệu dòng forecasts mỗi request.
    """
    t0 = time.perf_counter()
    conn.execute("DELETE FROM agg_forecast_date_family")
    cur = conn.execute("""
        INSERT INTO agg_forecast_date_family (store_nbr, date, family, predicted_sales)
        SELECT f.store_nbr, f.date, i.family, SUM(f.predicted_sales)
        FROM forecasts f
        JOIN items i ON i.item_nbr = f.item_nbr
        GROUP BY f.store_nbr, f.date, i.family
    """)
    conn.commit()
    logger.info(f"HOÀN THÀNH agg_forecast_date_family: {cur.rowcount:,} dòng "
                f"({time.perf_counter() - t0:.0f}s)")


def main() -> None:
    if not DB_PATH.exists():
        logger.error(f"Không tìm thấy {DB_PATH}. Chạy ml_training/src/init_database.py trước.")
        sys.exit(1)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    try:
        ensure_schema(conn)
        build(conn)
        build_forecast_aggregate(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
