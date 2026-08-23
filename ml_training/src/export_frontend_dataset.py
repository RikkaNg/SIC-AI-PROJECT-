"""
export_frontend_dataset.py
Xuất dữ liệu bán hàng lịch sử (aggregate theo ngày x ngành hàng) từ retail.db
sang frontend/public/sale_dataset_ver_2.csv để dashboard React đọc bằng Papa.parse.

Cột đầu ra khớp interface RawSaleData trong App.tsx: date, family, perishable, unit_sales.

Chạy: python ml_training/src/export_frontend_database.py
(không cần pandas - chỉ dùng sqlite3 + csv của thư viện chuẩn)
"""

import csv
import logging
import sqlite3
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SRC_DIR = Path(__file__).resolve().parent               # ml_training/src/
PROJECT_ROOT = SRC_DIR.parent.parent                    # SIC-AI-PROJECT-/

DB_PATH = PROJECT_ROOT / "backend" / "src" / "database" / "retail.db"
OUTPUT_PATH = PROJECT_ROOT / "frontend" / "public" / "sale_dataset_ver_2.csv"


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy database: {DB_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        # Tối ưu cho aggregate trên DB lớn (~59 triệu dòng)
        conn.execute("PRAGMA cache_size=-262144")   # 256MB page cache
        conn.execute("PRAGMA mmap_size=2147483648") # memory-mapped I/O 2GB
        conn.execute("PRAGMA temp_store=MEMORY")

        total_rows = conn.execute("SELECT COUNT(*) FROM historical_sales").fetchone()[0]
        logger.info(f"historical_sales có {total_rows:,} dòng. Bắt đầu aggregate...")

        query = """
            SELECT h.date,
                   i.family,
                   i.perishable,
                   ROUND(SUM(h.unit_sales), 2) AS unit_sales
            FROM historical_sales h
            JOIN items i ON h.item_nbr = i.item_nbr
            GROUP BY h.date, i.family, i.perishable
            ORDER BY h.date, unit_sales DESC
        """
        cursor = conn.execute(query)

        written = 0
        with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "family", "perishable", "unit_sales"])
            while True:
                batch = cursor.fetchmany(50_000)
                if not batch:
                    break
                writer.writerows(batch)
                written += len(batch)

        logger.info(f"Đã ghi {written:,} dòng vào {OUTPUT_PATH}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
