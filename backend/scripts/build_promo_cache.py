# backend/scripts/build_promo_cache.py
"""
Dựng bảng tổng hợp hiệu quả khuyến mãi agg_promo_family_stats từ historical_sales
(59 triệu dòng) để tool evaluate_promotion_impact của LLM Agent trả lời tức thì
thay vì quét bảng lớn mỗi lần chat. Chạy MỘT LẦN (hoặc sau khi nạp dữ liệu mới):

    python backend/scripts/build_promo_cache.py
"""
import logging
import sqlite3
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent / "src" / "database" / "retail.db"


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS agg_promo_family_stats (
            store_nbr   INTEGER NOT NULL,
            family      TEXT NOT NULL,
            onpromotion INTEGER NOT NULL,
            item_day_rows  INTEGER NOT NULL,
            total_units    REAL NOT NULL,
            promo_days     INTEGER NOT NULL,
            PRIMARY KEY (store_nbr, family, onpromotion)
        ) WITHOUT ROWID;
    """)
    conn.commit()


def build(conn: sqlite3.Connection) -> None:
    """
    Một câu lệnh duy nhất quét tuần tự historical_sales (nhanh hơn nhiều so với
    54 lệnh per-store: mỗi lệnh phải scan index range + sort riêng, tổng ~2 giờ).
    """
    started = time.perf_counter()
    logger.info("Quét historical_sales và GROUP BY (store, family, onpromotion) ...")
    conn.execute("DELETE FROM agg_promo_family_stats")
    conn.execute("""
        INSERT INTO agg_promo_family_stats
        SELECT h.store_nbr, it.family, h.onpromotion,
               COUNT(*), SUM(h.unit_sales), COUNT(DISTINCT h.date)
        FROM historical_sales h
        JOIN items it ON h.item_nbr = it.item_nbr
        GROUP BY h.store_nbr, it.family, h.onpromotion
    """)
    conn.commit()
    n = conn.execute("SELECT COUNT(*) FROM agg_promo_family_stats").fetchone()[0]
    logger.info(f"Hoàn tất {n} dòng trong {time.perf_counter() - started:.1f}s")


def main() -> None:
    if not DB_PATH.exists():
        raise SystemExit(f"Khong tim thay DB: {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-256000")  # 256MB cho pass quet lon
    conn.execute("PRAGMA temp_store=MEMORY")
    try:
        ensure_schema(conn)
        build(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
