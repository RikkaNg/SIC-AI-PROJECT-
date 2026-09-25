"""
backend/scripts/init_auth.py (chạy 1 lần, có thể chạy lại để đồng bộ)
Khởi tạo auth.db cho Row-Level Isolation:
- Tạo bảng users / user_stores.
- Seed tài khoản mẫu (idempotent - chạy lại nhiều lần cho cùng kết quả):
    admin     / admin123   -> admin, toàn bộ 54 cửa hàng
    manager1  / manager123 -> cửa hàng 1
    manager2  / manager123 -> cửa hàng 2
    ...
    manager54 / manager123 -> cửa hàng 54
  Mỗi managerN chỉ quản đúng cửa hàng N. Phạm vi cửa hàng của các user
  trong danh sách seed được ĐỒNG BỘ theo khai báo dưới đây mỗi lần chạy:
  gán cửa hàng thủ công cho manager1..manager54 ngoài script sẽ bị ghi đè.
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

N_STORES = 54


def upsert_user(conn: sqlite3.Connection, username: str, password: str,
                role: str, display_name: str, stores):
    exists = conn.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone()
    if exists is None:
        salt_hex, hash_hex = hash_password(password)
        conn.execute(
            "INSERT INTO users (username, password_hash, salt, role, display_name) VALUES (?, ?, ?, ?, ?)",
            (username, hash_hex, salt_hex, role, display_name),
        )
        logger.info(f"  + Đã tạo user '{username}' ({role}, {display_name})")
    else:
        logger.info(f"  = User '{username}' đã tồn tại, giữ nguyên mật khẩu.")

    if stores is not None:  # None = admin (không gán store, xem toàn hệ thống)
        # Đồng bộ khai báo: xóa gán cũ rồi ghi đúng phạm vi hiện hành
        stores = [int(s) for s in stores]
        conn.execute("DELETE FROM user_stores WHERE username = ?", (username,))
        conn.executemany(
            "INSERT INTO user_stores (username, store_nbr) VALUES (?, ?)",
            [(username, s) for s in stores],
        )
        logger.info(f"    Gán {len(stores)} cửa hàng cho '{username}': {stores if len(stores) <= 3 else f'{stores[0]}..{stores[-1]}'}")


def main():
    AUTH_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(AUTH_DB_PATH))
    try:
        conn.executescript(SCHEMA)
        conn.commit()

        logger.info(">>> Seeding tài khoản mẫu...")
        upsert_user(conn, "admin", "admin123", "admin", "Quản trị hệ thống", None)
        for store_nbr in range(1, N_STORES + 1):
            upsert_user(conn, f"manager{store_nbr}", "manager123", "manager",
                        f"Quản lý cửa hàng {store_nbr}", [store_nbr])
        conn.commit()

        n_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        n_links = conn.execute("SELECT COUNT(*) FROM user_stores").fetchone()[0]
        logger.info(f"Hoàn tất. users={n_users}, user_stores={n_links}, DB={AUTH_DB_PATH}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
