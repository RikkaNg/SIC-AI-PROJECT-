"""
gen_jwt_secret.py - Sinh JWT_SECRET ngẫu nhiên và ghi vào .env

Chỉ thay thế khi JWT_SECRET bị thiếu, rỗng hoặc còn là giá trị dev mặc định
('dev-secret-change-me'). Secret người dùng tự đặt không bị đụng đến.

Chạy:  python backend/scripts/gen_jwt_secret.py
(run_local.ps1 tự gọi script này trước khi khởi động backend)
"""

import re
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = ROOT / ".env"
EXAMPLE_PATH = ROOT / ".envexamble"
DEV_DEFAULT = "dev-secret-change-me"


def main() -> int:
    if not ENV_PATH.exists():
        if not EXAMPLE_PATH.exists():
            print(f"LOI: khong tim thay {ENV_PATH} va {EXAMPLE_PATH} de khoi tao.")
            return 1
        ENV_PATH.write_text(EXAMPLE_PATH.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Da tao .env tu {EXAMPLE_PATH.name}.")

    lines = ENV_PATH.read_text(encoding="utf-8").splitlines()
    pattern = re.compile(r"^JWT_SECRET=(.*)$")
    idx = next((i for i, line in enumerate(lines) if pattern.match(line)), None)

    if idx is not None:
        current = pattern.match(lines[idx]).group(1).strip()
        if current and current != DEV_DEFAULT:
            print("JWT_SECRET da duoc cau hinh rieng - giu nguyen.")
            return 0
        lines[idx] = f"JWT_SECRET={secrets.token_urlsafe(48)}"
    else:
        lines.append(f"JWT_SECRET={secrets.token_urlsafe(48)}")

    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Da sinh JWT_SECRET moi ({len(secrets.token_urlsafe(48))} ky tu) vao .env.")
    print("Luu y: cac token JWT cu het hieu luc - restart backend de nap secret moi.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
