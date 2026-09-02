# -*- coding: utf-8 -*-
"""
apply_product_names.py (v1.0)
Migration một lần: thêm cột `name` vào bảng items của retail.db và nạp tên từ
product_names.csv (sinh bởi generate_product_names.py).

- Không rebuild DB: chỉ ALTER TABLE (nếu thiếu cột) + UPDATE theo item_nbr.
- Idempotent: chạy lại nhiều lần an toàn.

Chạy: python ml_training/src/apply_product_names.py
"""

import sys
import sqlite3
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent
PROJECT_ROOT = BASE_DIR.parent

RAW_DIR = BASE_DIR / "data" / "raw"
NAMES_CSV = RAW_DIR / "product_names.csv"
DB_PATH = PROJECT_ROOT / "backend" / "src" / "database" / "retail.db"


def main() -> int:
    if not DB_PATH.exists():
        raise SystemExit(f"Không tìm thấy {DB_PATH}")
    if not NAMES_CSV.exists():
        raise SystemExit(f"Không tìm thấy {NAMES_CSV} - chạy generate_product_names.py trước.")

    names_df = pd.read_csv(NAMES_CSV, dtype={"item_nbr": "int64", "name": "str"})
    if names_df["name"].isna().any() or not names_df["name"].is_unique:
        raise SystemExit("product_names.csv có tên rỗng/trùng - regenerate lại.")

    conn = sqlite3.connect(DB_PATH)
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(items)")}
        if "name" not in cols:
            conn.execute("ALTER TABLE items ADD COLUMN name TEXT")
            conn.commit()
            print("[OK] Đã thêm cột items.name")
        else:
            print("[OK] Cột items.name đã tồn tại - chỉ UPDATE dữ liệu")

        # SQL cần (name, item_nbr) - CSV có (item_nbr, name) nên phải đảo
        rows = [(name, item_nbr) for item_nbr, name in names_df.itertuples(index=False, name=None)]
        conn.executemany("UPDATE items SET name = ? WHERE item_nbr = ?", rows)
        conn.commit()

        total, named = conn.execute(
            "SELECT COUNT(*), COUNT(name) FROM items"
        ).fetchone()
        print(f"[OK] Đã nạp {len(rows):,} tên - coverage {named:,}/{total:,} items")

        if named != total:
            missing = conn.execute(
                "SELECT item_nbr FROM items WHERE name IS NULL LIMIT 10"
            ).fetchall()
            raise SystemExit(f"Còn {total - named} items thiếu tên, VD: {missing}")

        print("\n=== Mẫu sau khi nạp ===")
        for item_nbr, family, name in conn.execute(
            "SELECT item_nbr, family, name FROM items WHERE family IN ('BEVERAGES', 'PRODUCE') LIMIT 6"
        ):
            print(f"  {item_nbr:<10} {family:<12} {name}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
