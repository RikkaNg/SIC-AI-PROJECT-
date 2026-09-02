# -*- coding: utf-8 -*-
"""
load_forecasts_inventory.py
Nạp RIÊNG forecasts + inventory vào backend/src/database/retail.db
KHÔNG đụng historical_sales / agg_item_store_sales (59 triệu dòng).

Tái sử dụng nguyên logic của init_database.py:
  1. ingest_forecasts     - nạp submission_*.csv (merge test.csv để có store/item/date)
  2. generate_inventory   - sinh tồn kho velocity-based từ forecasts
  3. build_sku_stats      - dựng lại ABC + fc_total_16d + dải tin cậy (fc trước đây = 0)

Chạy: python ml_training/src/load_forecasts_inventory.py
"""

import sys
import logging
import sqlite3
from pathlib import Path

# Console Windows mặc định cp1252 - ép UTF-8 để log tiếng Việt không crash
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from init_database import (
    DB_PATH, optimize_sqlite, ingest_forecasts, generate_inventory,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    if not DB_PATH.exists():
        raise SystemExit(f"Không tìm thấy {DB_PATH} - chạy init_database.py trước.")

    conn = sqlite3.connect(DB_PATH)
    try:
        optimize_sqlite(conn)

        before = conn.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0]
        logger.info("Trước khi nạp: forecasts=%s dòng", f"{before:,}")

        ingest_forecasts(conn)          # DELETE rồi nạp lại từ submission_*.csv
        after = conn.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0]
        if after == 0:
            raise SystemExit(
                "forecasts vẫn rỗng! Kiểm tra ml_training/data/processed/ đã có "
                "submission_local.csv / submission_blend_hybrid.csv / submission_ensemble.csv chưa."
            )
        logger.info("Sau khi nạp: forecasts=%s dòng", f"{after:,}")

        generate_inventory(conn)        # sinh tồn kho từ forecasts vừa nạp

        # Dựng lại sku_stats (ABC, sold_2016, fc_total_16d) - bước này bắt buộc chạy lại
        # vì lần trước forecasts rỗng nên fc_total_16d toàn 0.
        try:
            from build_sku_stats import build_sku_stats
            logger.info(">>> Rebuilding sku_stats cache...")
            build_sku_stats(conn)
        except Exception as e:
            logger.warning("Bỏ qua rebuild sku_stats do lỗi: %s", e)

        rng = conn.execute(
            "SELECT MIN(date), MAX(date), COUNT(DISTINCT store_nbr) FROM forecasts"
        ).fetchone()
        inv = conn.execute("SELECT COUNT(*) FROM inventory").fetchone()[0]
        logger.info("=" * 55)
        logger.info("HOÀN TẤT: forecasts %s dòng (%s → %s, %s cửa hàng), inventory %s dòng",
                    f"{after:,}", rng[0], rng[1], rng[2], f"{inv:,}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
