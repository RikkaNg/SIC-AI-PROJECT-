"""
init_database.py (v2.5 - Production Ready)
Khởi tạo SQLite DB (retail.db) cho Backend Gateway:
- Tự động nạp file dự báo tối ưu nhất (submission_local.csv / submission_blend_hybrid.csv / submission_ensemble.csv).
- Thay thế toàn bộ to_sql bằng executemany() để tối đa tốc độ ghi và kiểm soát RAM.
- Đồng bộ đường dẫn xuất thẳng sang backend/src/database/retail.db.
- Tối ưu PRAGMA (WAL mode, async I/O, 64MB Cache) và Composite Indexing phục vụ AI Agent & API.
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
    """Khởi tạo Schema bảng và hệ thống Index tăng tốc truy vấn API / AI Agent."""
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
            perishable INTEGER,
            name TEXT
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
            last_updated TIMESTAMP,
            PRIMARY KEY (store_nbr, item_nbr)
        );

        CREATE TABLE IF NOT EXISTS historical_sales (
            date TEXT,
            store_nbr INTEGER,
            item_nbr INTEGER,
            unit_sales REAL,
            onpromotion INTEGER
        );

        -- Index tổ hợp tăng tốc truy vấn API và AI Agent
        CREATE INDEX IF NOT EXISTS idx_forecasts_store_date ON forecasts(store_nbr, date);
        CREATE INDEX IF NOT EXISTS idx_forecasts_item ON forecasts(item_nbr);
        CREATE INDEX IF NOT EXISTS idx_forecasts_date ON forecasts(date);
        CREATE INDEX IF NOT EXISTS idx_inventory_store ON inventory(store_nbr);
        CREATE INDEX IF NOT EXISTS idx_hist_store_date ON historical_sales(store_nbr, date);
        CREATE INDEX IF NOT EXISTS idx_hist_item ON historical_sales(item_nbr);
    """)
    conn.commit()


def ingest_stores_items(conn: sqlite3.Connection):
    """Nạp danh mục stores và items qua executemany."""
    logger.info(">>> Ingesting stores & items...")

    conn.execute("DELETE FROM stores")
    conn.execute("DELETE FROM items")
    conn.commit()

    stores_df = pd.read_csv(RAW_DIR / "stores.csv")
    conn.executemany(
        "INSERT INTO stores (store_nbr, city, state, type, cluster) VALUES (?, ?, ?, ?, ?)",
        stores_df[["store_nbr", "city", "state", "type", "cluster"]].itertuples(index=False, name=None)
    )

    items_df = pd.read_csv(RAW_DIR / "items.csv")
    # Tên sản phẩm sinh bởi generate_product_names.py (tùy chọn - thiếu thì name NULL)
    names_path = RAW_DIR / "product_names.csv"
    if names_path.exists():
        names_df = pd.read_csv(names_path, dtype={"item_nbr": "int64", "name": "str"})
        items_df = items_df.merge(names_df, on="item_nbr", how="left")
    else:
        items_df["name"] = None
    conn.executemany(
        "INSERT INTO items (item_nbr, family, class, perishable, name) VALUES (?, ?, ?, ?, ?)",
        items_df[["item_nbr", "family", "class", "perishable", "name"]].itertuples(index=False, name=None)
    )
    conn.commit()

    logger.info(f"    Loaded {len(stores_df):,} stores and {len(items_df):,} items "
                f"(named: {int(items_df['name'].notna().sum()):,}).")


def resolve_submission_file() -> Path:
    """Tự động ưu tiên chọn file dự báo có độ chính xác cao nhất."""
    candidates = [
        PROCESSED_DIR / "submission_blend_hybrid.csv",
        PROCESSED_DIR / "submission_local.csv",
        PROCESSED_DIR / "submission_ensemble.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def ingest_forecasts(conn: sqlite3.Connection):
    """Nạp kết quả dự báo SKU kết hợp metadata từ test.csv."""
    sub_file = resolve_submission_file()
    test_file = RAW_DIR / "test.csv"

    if sub_file is None:
        logger.warning("    No submission file found in processed directory. Please run predict_local.py first!")
        return

    logger.info(f">>> Ingesting forecasts from '{sub_file.name}'...")

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

    chunk_size = 250_000
    total_rows = 0

    for chunk in pd.read_csv(sub_file, chunksize=chunk_size, dtype={"id": "int32", "unit_sales": "float32"}):
        merged = chunk.merge(test_df, on="id", how="left")
        merged = merged[["id", "store_nbr", "item_nbr", "date", "unit_sales"]]

        conn.executemany(insert_sql, merged.itertuples(index=False, name=None))
        conn.commit()

        total_rows += len(merged)
        logger.info(f"    Inserted {total_rows:,} forecast records...")

    logger.info(f"    Total forecasts completed: {total_rows:,}")


def generate_inventory(conn: sqlite3.Connection):
    """Khởi tạo bảng tồn kho dựa trên Sales Velocity và Lead Time."""
    logger.info(">>> Generating velocity-based inventory...")
    conn.execute("DELETE FROM inventory")

    query = """
        SELECT store_nbr, item_nbr, AVG(predicted_sales) as avg_daily_sales
        FROM forecasts
        GROUP BY store_nbr, item_nbr
    """
    df_pairs = pd.read_sql_query(query, conn)

    if df_pairs.empty:
        logger.warning("    No forecasts found. Skipping inventory generation.")
        return

    n_pairs = len(df_pairs)
    df_pairs["lead_time_days"] = np.random.randint(1, 8, size=n_pairs)

    # 60% Bình thường, 20% Sắp hết (Understock), 20% Đọng vốn (Overstock)
    states = np.random.choice([0, 1, 2], size=n_pairs, p=[0.6, 0.2, 0.2])

    coverage_days = np.where(
        states == 0, df_pairs["lead_time_days"] + 4,
        np.where(states == 1, np.random.randint(1, 3, size=n_pairs),
                 np.random.randint(30, 45, size=n_pairs))
    )

    df_pairs["current_stock"] = np.nan_to_num(df_pairs["avg_daily_sales"] * coverage_days)
    df_pairs["current_stock"] = df_pairs["current_stock"].round(0).astype(int).clip(lower=0)

    batch = [
        (int(row["store_nbr"]), int(row["item_nbr"]), int(row["current_stock"]), int(row["lead_time_days"]))
        for _, row in df_pairs.iterrows()
    ]

    conn.executemany(
        "INSERT INTO inventory (store_nbr, item_nbr, current_stock, lead_time_days) VALUES (?, ?, ?, ?)",
        batch
    )
    conn.commit()

    understock_count = int((states == 1).sum())
    overstock_count = int((states == 2).sum())
    logger.info(f"    Inventory generated for {n_pairs:,} store-item pairs.")
    logger.info(f"    Simulated states: {n_pairs - understock_count - overstock_count:,} Normal, {understock_count:,} Understock, {overstock_count:,} Overstock.")


def ingest_historical_sales(conn: sqlite3.Connection):
    """Nạp dữ liệu bán hàng lịch sử từ 2016 qua streaming executemany."""
    logger.info(">>> Ingesting historical sales (from 2016)...")

    conn.execute("DELETE FROM historical_sales")
    conn.commit()

    start_date_str = "2016-01-01"
    dtypes = {
        "date": "str",
        "store_nbr": "int16",
        "item_nbr": "int32",
        "unit_sales": "float32",
        "onpromotion": "float32"
    }

    insert_sql = """
        INSERT INTO historical_sales (date, store_nbr, item_nbr, unit_sales, onpromotion)
        VALUES (?, ?, ?, ?, ?)
    """

    chunk_size = 2_000_000
    total_rows = 0

    logger.info(">>> Streaming train.csv in chunks (filtering >= 2016-01-01)...")
    for chunk in pd.read_csv(
        RAW_DIR / "train.csv",
        usecols=["date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"],
        dtype=dtypes,
        chunksize=chunk_size
    ):
        c_hist = chunk[chunk["date"] >= start_date_str].copy()

        if not c_hist.empty:
            c_hist["onpromotion"] = c_hist["onpromotion"].fillna(0).astype(int)
            c_hist["unit_sales"] = c_hist["unit_sales"].clip(lower=0)

            conn.executemany(
                insert_sql,
                c_hist[["date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"]].itertuples(index=False, name=None)
            )
            conn.commit()
            total_rows += len(c_hist)
            logger.info(f"    Inserted {total_rows:,} historical rows...")

    logger.info(f"    Historical sales ingestion completed: {total_rows:,} rows.")


def build_sales_aggregate(conn: sqlite3.Connection):
    """
    Dựng bảng tổng hợp doanh số 2016 theo (store_nbr, item_nbr) cho API sản phẩm.
    Bảng agg_item_store_sales giúp /api/top-products, /api/products, /api/family-mix
    chạy tức thì thay vì GROUP BY trực tiếp trên ~59 triệu dòng historical_sales.
    (Đồng bộ logic với backend/scripts/build_sales_cache.py)
    """
    logger.info(">>> Building sales aggregate (agg_item_store_sales)...")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS agg_item_store_sales (
            store_nbr INTEGER NOT NULL,
            item_nbr  INTEGER NOT NULL,
            unit_sales REAL NOT NULL DEFAULT 0,
            PRIMARY KEY (store_nbr, item_nbr)
        ) WITHOUT ROWID;
        CREATE INDEX IF NOT EXISTS idx_agg_item ON agg_item_store_sales(item_nbr);
    """)
    conn.commit()

    stores = [int(r[0]) for r in conn.execute(
        "SELECT DISTINCT store_nbr FROM historical_sales ORDER BY store_nbr"
    ).fetchall()]
    total_rows = 0
    for i, store in enumerate(stores, 1):
        cur = conn.execute("""
            DELETE FROM agg_item_store_sales WHERE store_nbr = ?;
        """, (store,))
        cur = conn.execute("""
            INSERT INTO agg_item_store_sales (store_nbr, item_nbr, unit_sales)
            SELECT ?, item_nbr, SUM(unit_sales)
            FROM historical_sales
            WHERE store_nbr = ?
            GROUP BY item_nbr
        """, (store, store))
        conn.commit()
        total_rows += cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
        if i % 10 == 0 or i == len(stores):
            logger.info(f"    Aggregate progress: {i}/{len(stores)} stores, {total_rows:,} rows")

    logger.info(f"    Sales aggregate built: {total_rows:,} store-item rows.")


# ====================== 4. ENTRYPOINT ======================
def main(reset: bool = False):
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
        ingest_historical_sales(conn)
        build_sales_aggregate(conn)

        # Bổ sung: dựng bảng sku_stats + model_meta cho API phân tích SKU
        # (ABC, sold_2016, dự báo 16 ngày ± dải tin cậy, validation proxy).
        # Không bắt buộc - lỗi ở đây không làm hỏng database đã nạp xong.
        try:
            from build_sku_stats import build_sku_stats  # cùng thư mục src/
            logger.info(">>> Building sku_stats cache (ABC / sold_2016 / validation)...")
            build_sku_stats(conn)
        except Exception as e:
            logger.warning(f"    Bỏ qua build_sku_stats do lỗi: {e}")

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