# backend/src/database/connection.py
import sqlite3
import os
import logging

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "retail.db")

def get_db_connection():
    """Tạo connection đến SQLite và bật WAL mode"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

# Hàm dùng cho inventory_routes.py
def get_inventory_db(store_nbr: int, item_nbr: int):
    conn = get_db_connection()
    try:
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