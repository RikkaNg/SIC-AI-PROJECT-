# backend/src/database/connection.py
import sqlite3
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Ưu tiên biến môi trường DB_PATH (đồng bộ với llm_agent/tools.py),
# fallback về file retail.db nằm cạnh module này.
DB_PATH = Path(os.environ.get("DB_PATH") or (Path(__file__).resolve().parent / "retail.db"))

_last_updated_checked = False


def get_db_connection():
    """Tạo connection đến SQLite và bật WAL mode"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def _ensure_last_updated(conn: sqlite3.Connection):
    """
    Migration idempotent: thêm cột last_updated cho bảng inventory nếu thiếu.
    ALTER TABLE ADD COLUMN trên SQLite chỉ sửa metadata nên chạy gần tức thì,
    an toàn với DB dung lượng lớn.
    """
    global _last_updated_checked
    if _last_updated_checked:
        return
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(inventory)")}
    if "last_updated" not in columns:
        conn.execute("ALTER TABLE inventory ADD COLUMN last_updated TIMESTAMP")
        conn.commit()
        logger.info("Đã thêm cột last_updated vào bảng inventory.")
    _last_updated_checked = True


# Hàm dùng cho inventory_routes.py
def get_inventory_db(store_nbr: int, item_nbr: int):
    conn = get_db_connection()
    try:
        _ensure_last_updated(conn)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT store_nbr, item_nbr, current_stock, last_updated FROM inventory WHERE store_nbr = ? AND item_nbr = ?",
            (store_nbr, item_nbr)
        )
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()

def update_stock_db(store_nbr: int, item_nbr: int, quantity: int):
    conn = get_db_connection()
    try:
        _ensure_last_updated(conn)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE inventory SET current_stock = current_stock + ?, last_updated = CURRENT_TIMESTAMP WHERE store_nbr = ? AND item_nbr = ?",
            (quantity, store_nbr, item_nbr)
        )
        conn.commit()
        if cursor.rowcount == 0:
            raise Exception("Không tìm thấy sản phẩm để cập nhật")
    finally:
        conn.close()
