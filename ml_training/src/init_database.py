"""
init_database.py
Khởi tạo SQLite DB (retail.db) cho Backend Gateway:
- Khắc phục triệt để lỗi SQLite 'too many SQL variables' bằng executemany().
- Đồng bộ đường dẫn xuất thẳng sang backend/src/database/retail.db.
- Streaming nạp dữ liệu theo Chunk và Batch Insert kiểm soát RAM.
- Tối ưu PRAGMA (WAL mode, async I/O, 64MB Cache) và Composite Indexing.
"""

import sys
from pathlib import Path
import sqlite3
import logging
import numpy as np
import pandas as pd

# ====================== 1. CẤU HÌNH ĐƯỜNG DẪN ======================
SRC_DIR = Path(__file__).resolve().parent               # ml_training/src/
BASE_DIR = SRC_DIR.parent                              # ml_training/
PROJECT_ROOT = BASE_DIR.parent                         # SIC-AI-PROJECT-/

# Dữ liệu nguồn
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Thư mục Database Backend
BACKEND_DB_DIR = PROJECT_ROOT / "backend" / "src" / "database"
BACKEND_DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = BACKEND_DB_DIR / "retail.db"

# ====================== 2. LOGGING & SEED ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Cố định random seed để số liệu tồn kho giả lập nhất quán giữa các lần chạy
np.random.seed(42)


# ====================== 3. CÁC HÀM XỬ LÝ DATABASE ======================
def optimize_sqlite(conn: sqlite3.Connection):
    """Bật các cờ tối ưu hóa tốc độ ghi và truy vấn trên SQLite."""
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")  # Cấp phát 64MB RAM Cache
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.commit()
    logger.info("    SQLite optimized (WAL mode, NORMAL sync, 64MB Cache)")


def init_schema(conn: sqlite3.Connection):
    """Khởi tạo Schema bảng và hệ thống Index tăng tốc truy vấn API."""
    logger.info(">>> Initializing database schema...")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS stores (
            store_nbr INTEGER PRIMARY KEY,
            city TEXT,
            state TEXT,
            type TEXT,
            cluster INTEGER
        );

        CREATE TABLE IF NOT EXISTS items (
            item_nbr INTEGER PRIMARY KEY,
            family TEXT,
            class INTEGER,
            perishable INTEGER
        );

        CREATE TABLE IF NOT EXISTS forecasts (
            id INTEGER PRIMARY KEY,
            store_nbr INTEGER,
            item_nbr INTEGER,
            date TEXT,
            predicted_sales REAL
        );

        CREATE TABLE IF NOT EXISTS inventory (
            store_nbr INTEGER,
            item_nbr INTEGER,
            current_stock REAL,
            lead_time_days INTEGER,
            PRIMARY KEY (store_nbr, item_nbr)
        );

        -- Index phục vụ lọc nhanh theo thời gian và cửa hàng
        CREATE INDEX IF NOT EXISTS idx_forecasts_store_date ON forecasts(store_nbr, date);
        CREATE INDEX IF NOT EXISTS idx_forecasts_item ON forecasts(item_nbr);
        CREATE INDEX IF NOT EXISTS idx_forecasts_date ON forecasts(date);
        CREATE INDEX IF NOT EXISTS idx_inventory_store ON inventory(store_nbr);
    """)
    conn.commit()


def ingest_stores_items(conn: sqlite3.Connection):
    """Nạp danh mục stores và items (Giữ nguyên cấu trúc Schema và Primary Key)."""
    logger.info(">>> Ingesting stores & items...")

    conn.execute("DELETE FROM stores")
    conn.execute("DELETE FROM items")
    conn.commit()

    stores_df = pd.read_csv(RAW_DIR / "stores.csv")
    stores_df.to_sql("stores", conn, if_exists="append", index=False)

    items_df = pd.read_csv(RAW_DIR / "items.csv")
    items_df.to_sql("items", conn, if_exists="append", index=False)

    logger.info(f"    Loaded {len(stores_df):,} stores and {len(items_df):,} items.")


def ingest_forecasts(conn: sqlite3.Connection):
    """
    Nạp kết quả dự báo từ submission_ensemble.csv kết hợp test.csv.
    Sử dụng executemany() để khắc phục triệt để giới hạn biến của SQLite.
    """
    sub_file = PROCESSED_DIR / "submission_ensemble.csv"
    test_file = RAW_DIR / "test.csv"

    if not sub_file.exists():
        logger.warning(f"    File '{sub_file.name}' not found. Please run predict.py first!")
        return

    logger.info(f">>> Ingesting forecasts from {sub_file.name}...")

    # Đọc test.csv để lấy mapping id -> (store_nbr, item_nbr, date)
    test_df = pd.read_csv(
        test_file,
        usecols=["id", "store_nbr", "item_nbr", "date"],
        dtype={"id": "int32", "store_nbr": "int16", "item_nbr": "int32", "date": "str"}
    )

    conn.execute("DELETE FROM forecasts")
    conn.commit()

    insert_sql = """
        INSERT INTO forecasts (id, store_nbr, item_nbr, date, predicted_sales)
        VALUES (?, ?, ?, ?, ?)
    """

    chunk_size = 200_000
    total_rows = 0

    for chunk in pd.read_csv(sub_file, chunksize=chunk_size, dtype={"id": "int32", "unit_sales": "float32"}):
        merged = chunk.merge(test_df, on="id", how="left")
        merged = merged[["id", "store_nbr", "item_nbr", "date", "unit_sales"]]

        # executemany + itertuples không bị dính giới hạn SQL variable limit
        conn.executemany(insert_sql, merged.itertuples(index=False, name=None))
        conn.commit()

        total_rows += len(merged)
        logger.info(f"    Inserted {total_rows:,} forecast records...")

    logger.info(f"    Total forecasts completed: {total_rows:,}")


def generate_inventory(conn: sqlite3.Connection):
    """Khởi tạo bảng tồn kho ngẫu nhiên (Vectorized bằng NumPy để đạt tốc độ cao)."""
    logger.info(">>> Generating inventory...")
    conn.execute("DELETE FROM inventory")

    cursor = conn.execute("SELECT DISTINCT store_nbr, item_nbr FROM forecasts")
    pairs = cursor.fetchall()

    if not pairs:
        logger.warning("    No forecasts found. Skipping inventory generation.")
        return

    n_pairs = len(pairs)
    stocks = np.random.randint(10, 200, size=n_pairs)
    leads = np.random.randint(1, 8, size=n_pairs)

    batch = [
        (pairs[i][0], pairs[i][1], int(stocks[i]), int(leads[i]))
        for i in range(n_pairs)
    ]

    conn.executemany(
        "INSERT INTO inventory (store_nbr, item_nbr, current_stock, lead_time_days) VALUES (?, ?, ?, ?)",
        batch
    )
    conn.commit()
    logger.info(f"    Generated inventory for {n_pairs:,} store-item pairs.")


# ====================== 4. ENTRYPOINT ======================
def main(reset: bool = False):
    """
    Args:
        reset: True = Xóa file DB vật lý và tạo lại hoàn toàn.
               False = Giữ nguyên file DB, chỉ cập nhật/làm mới dữ liệu các bảng.
    """
    if reset and DB_PATH.exists():
        DB_PATH.unlink()
        logger.info(">>> Removed existing database file.")

    logger.info(f">>> Target Database Location: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)

    try:
        optimize_sqlite(conn)
        init_schema(conn)
        ingest_stores_items(conn)
        ingest_forecasts(conn)
        generate_inventory(conn)

        cursor = conn.execute("SELECT COUNT(*) FROM forecasts")
        count = cursor.fetchone()[0]
        logger.info("\n" + "=" * 55)
        logger.info("DATABASE INITIALIZATION SUCCESSFUL")
        logger.info(f"Database File : {DB_PATH}")
        logger.info(f"Total Forecasts: {count:,}")
        logger.info("=" * 55)
    finally:
        conn.close()


if __name__ == "__main__":
    main(reset=False)