"""
backend/scripts/init_auth.py (chạy 1 lần)
Khởi tạo auth.db cho Row-Level Isolation:
- Tạo bảng users / user_stores.
- Seed tài khoản mẫu (idempotent - không ghi đè user đã tồn tại):
    admin    / admin123   -> admin, toàn bộ 54 cửa hàng
    manager1 / manager123 -> cửa hàng 1-10
    manager2 / manager123 -> cửa hàng 11-20
CẢNH BÁO: đổi mật khẩu trước khi triển khai production!

Chạy:  python backend/scripts/init_auth.py
"""

import logging
import sqlite3
import sys
from pathlib import Path

# Cho phép chạy trực tiếp file này: thêm backend/src vào sys.path để import security
BACKEND_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(BACKEND_SRC))

from security import AUTH_DB_PATH, hash_password  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("init_auth")

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    username      TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,
    salt          TEXT NOT NULL,
    role          TEXT NOT NULL CHECK (role IN ('admin', 'manager')),
    display_name  TEXT NOT NULL,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS user_stores (
    username  TEXT NOT NULL REFERENCES users(username),
    store_nbr INTEGER NOT NULL,
    PRIMARY KEY (username, store_nbr)
);
"""


def upsert_user(conn: sqlite3.Connection, username: str, password: str,
                role: str, display_name: str, stores):
    exists = conn.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone()
    salt_hex, hash_hex = hash_password(password)
    if exists is None:
        conn.execute(
            "INSERT INTO users (username, password_hash, salt, role, display_name) VALUES (?, ?, ?, ?, ?)",
            (username, hash_hex, salt_hex, role, display_name),
        )
        logger.info(f"  + Đã tạo user '{username}' ({role}, {display_name})")
    else:
        logger.info(f"  = User '{username}' đã tồn tại, bỏ qua.")

    if stores is not None:  # None = admin (không cần gán store)
        added = 0
        for s in stores:
            cur = conn.execute(
                "INSERT OR IGNORE INTO user_stores (username, store_nbr) VALUES (?, ?)",
                (username, int(s)),
            )
            added += cur.rowcount
        if added:
            logger.info(f"    Gán {added} cửa hàng cho '{username}'.")


def main():
    AUTH_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(AUTH_DB_PATH))
    try:
        conn.executescript(SCHEMA)
        conn.commit()

        logger.info(">>> Seeding tài khoản mẫu...")
        upsert_user(conn, "admin", "admin123", "admin", "Quản trị hệ thống", None)
        upsert_user(conn, "manager1", "manager123", "manager", "Quản lý cửa hàng 1-10",
                    range(1, 11))
        upsert_user(conn, "manager2", "manager123", "manager", "Quản lý cửa hàng 11-20",
                    range(11, 21))
        conn.commit()

        n_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        n_links = conn.execute("SELECT COUNT(*) FROM user_stores").fetchone()[0]
        logger.info(f"Hoàn tất. users={n_users}, user_stores={n_links}, DB={AUTH_DB_PATH}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
