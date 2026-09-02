# backend/src/security.py
"""
Row-Level Isolation & Authentication cho Retail API Gateway.

- Mật khẩu: PBKDF2-HMAC-SHA256 (thuần stdlib).
- Token: JWT HS256 (PyJWT) cho web UI + X-API-Key tĩnh cho ERP/POS tích hợp.
- Identity.allowed_stores = None  -> được xem TOÀN HỆ THỐNG (admin / ERP).
- Identity.allowed_stores = set() -> chưa được gán cửa hàng nào (chặn mọi query dữ liệu).
"""
import hashlib
import hmac
import logging
import os
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

# ====================== CẤU HÌNH ======================
JWT_SECRET = os.environ.get("JWT_SECRET", "dev-secret-change-me")
JWT_ALGORITHM = "HS256"
if JWT_SECRET == "dev-secret-change-me":
    logger.warning("JWT_SECRET đang dùng giá trị mặc định 'dev-secret-change-me' - CHỈ dùng cho môi trường dev!")

TOKEN_EXPIRE_MINUTES = int(os.environ.get("TOKEN_EXPIRE_MINUTES", "480"))
ERP_API_KEY = os.environ.get("ERP_API_KEY") or None

AUTH_DB_PATH = Path(
    os.environ.get("AUTH_DB_PATH")
    or (Path(__file__).resolve().parent / "database" / "auth.db")
)

# retail.db - dùng để map cluster -> stores khi validate tool compare_cluster_trends
_RETAIL_DB_PATH = Path(
    os.environ.get("DB_PATH")
    or (Path(__file__).resolve().parent / "database" / "retail.db")
)


# ====================== IDENTITY ======================
@dataclass(frozen=True)
class Identity:
    username: str
    role: str
    display_name: str
    allowed_stores: Optional[Set[int]]  # None = tất cả cửa hàng

    @property
    def is_admin(self) -> bool:
        return self.allowed_stores is None


# ====================== PASSWORD HASHING ======================
def _pbkdf2(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000).hex()


def hash_password(password: str, salt_hex: Optional[str] = None):
    """Trả về (salt_hex, hash_hex). Dùng khi tạo user mới."""
    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
    return salt.hex(), _pbkdf2(password, salt)


def verify_password(password: str, salt_hex: str, expected_hash_hex: str) -> bool:
    return hmac.compare_digest(_pbkdf2(password, bytes.fromhex(salt_hex)), expected_hash_hex)


# ====================== JWT ======================
def create_access_token(username: str, role: str, display_name: str,
                        allowed_stores: Optional[Set[int]],
                        expires_minutes: Optional[int] = None) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": username,
        "role": role,
        "name": display_name,
        # stores=None nghĩa là admin/ERP: không giới hạn cửa hàng
        "stores": sorted(allowed_stores) if allowed_stores is not None else None,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=expires_minutes or TOKEN_EXPIRE_MINUTES)).timestamp()),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> Dict:
    """Raise jwt.ExpiredSignatureError / jwt.InvalidTokenError nếu lỗi."""
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])


# ====================== AUTH DB HELPERS ======================
def get_auth_connection() -> sqlite3.Connection:
    if not AUTH_DB_PATH.exists():
        raise RuntimeError(
            f"Không tìm thấy {AUTH_DB_PATH}. Vui lòng chạy: python backend/scripts/init_auth.py"
        )
    conn = sqlite3.connect(str(AUTH_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def authenticate(username: str, password: str) -> Optional[Identity]:
    """Đăng nhập bằng username/password. Trả về None nếu sai thông tin."""
    conn = get_auth_connection()
    try:
        row = conn.execute(
            "SELECT username, password_hash, salt, role, display_name FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        if row is None or not verify_password(password, row["salt"], row["password_hash"]):
            return None
        if row["role"] == "admin":
            return Identity(row["username"], row["role"], row["display_name"], None)
        stores = {
            int(r["store_nbr"])
            for r in conn.execute(
                "SELECT store_nbr FROM user_stores WHERE username = ?", (username,)
            )
        }
        return Identity(row["username"], row["role"], row["display_name"], stores)
    finally:
        conn.close()


# ====================== FASTAPI DEPENDENCY ======================
_bearer_scheme = HTTPBearer(auto_error=False)
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
    api_key: str = Depends(_api_key_header),
) -> Identity:
    """Nhận diện từ Bearer JWT hoặc X-API-Key (ERP/POS). Raise 401 nếu không hợp lệ."""
    # Ưu tiên 1: API key tĩnh cho hệ thống ngoài
    if api_key and ERP_API_KEY and hmac.compare_digest(api_key, ERP_API_KEY):
        return Identity(username="erp-integration", role="admin", display_name="ERP Integration", allowed_stores=None)

    # Ưu tiên 2: JWT
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Thiếu thông tin xác thực (cần Bearer token hoặc X-API-Key).",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = decode_token(credentials.credentials)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token đã hết hạn. Vui lòng đăng nhập lại.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Token không hợp lệ.")

    raw_stores = payload.get("stores")
    allowed = None if raw_stores is None else {int(s) for s in raw_stores}
    return Identity(
        username=str(payload.get("sub", "")),
        role=str(payload.get("role", "manager")),
        display_name=str(payload.get("name") or payload.get("sub", "")),
        allowed_stores=allowed,
    )


# ====================== RLS HELPERS ======================
def ensure_store_access(identity: Identity, store_nbr: int) -> None:
    """Raise 403 nếu identity không được phép xem cửa hàng này."""
    if identity.allowed_stores is not None and int(store_nbr) not in identity.allowed_stores:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Bạn không có quyền truy cập dữ liệu cửa hàng {store_nbr}. "
                   f"Phạm vi của bạn: {sorted(identity.allowed_stores)}.",
        )


def identity_scope_sql(identity: Identity, requested_store: Optional[int],
                       alias: str = "") -> Tuple[str, list]:
    """
    Hàm Row-Level Isolation DUY NHẤT cho mọi route đọc dữ liệu bán lẻ.
    Trả về (where_sql, params) cho cột `[alias.]store_nbr` theo identity.

    - requested_store chỉ định -> phải thuộc phạm vi (else 403), filter theo nó.
    - Không chỉ định:
        + admin/ERP (allowed_stores=None) -> không filter (toàn hệ thống)
        + manager có scope rỗng -> 403 (không được gán cửa hàng nào)
        + manager bình thường -> WHERE ... IN (?,...)
    """
    col = f"{alias}.store_nbr" if alias else "store_nbr"
    if requested_store is not None:
        ensure_store_access(identity, requested_store)
        return f" WHERE {col} = ?", [int(requested_store)]
    if identity.allowed_stores is None:
        return "", []
    if not identity.allowed_stores:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="Tài khoản chưa được gán cửa hàng nào. Liên hệ quản trị viên.")
    marks = ",".join("?" * len(identity.allowed_stores))
    return f" WHERE {col} IN ({marks})", sorted(identity.allowed_stores)


def store_filter(identity: Identity, requested_store: Optional[int]):
    """Wrapper tương thích ngược: scope trên cột `store_nbr` không tiền tố."""
    return identity_scope_sql(identity, requested_store, alias="")


# ====================== TOOL ACCESS VALIDATION (LLM Agent) ======================
_FORBIDDEN_TEMPLATE = (
    "forbidden: Bạn chỉ được truy cập các cửa hàng {scope}. "
    "Vui lòng chỉ phân tích trong phạm vi này."
)


def _stores_in_cluster(cluster_nbr: int) -> Set[int]:
    if not _RETAIL_DB_PATH.exists():
        return set()
    conn = sqlite3.connect(f"file:{_RETAIL_DB_PATH}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT store_nbr FROM stores WHERE cluster = ?", (cluster_nbr,)
        ).fetchall()
        return {int(r[0]) for r in rows}
    finally:
        conn.close()


def validate_tool_access(tool_name: str, tool_args: Dict, identity_allowed: Optional[Set[int]]) -> Optional[Dict]:
    """
    Kiểm tra quyền trước khi LLM Agent thực thi tool.
    Trả về None nếu OK; trả về dict lỗi {"status": "forbidden", ...} nếu vi phạm
    (dict này sẽ được đưa lại cho LLM để diễn giải lịch sự, không lộ dữ liệu).
    """
    if identity_allowed is None:  # admin / ERP: toàn quyền
        return None

    # Scope rỗng = chưa được gán cửa hàng nào: chặn mọi tool đọc dữ liệu,
    # khớp hành vi 403 của identity_scope_sql ở các route thường.
    if not identity_allowed:
        return {"status": "forbidden",
                "message": _FORBIDDEN_TEMPLATE.format(scope="(không có)")}

    scope_text = ", ".join(map(str, sorted(identity_allowed)))

    # Tool có tham số store_nbr bắt buộc hoặc tùy chọn
    store_nbr = tool_args.get("store_nbr")
    if store_nbr is not None:
        if int(store_nbr) not in identity_allowed:
            return {"status": "forbidden",
                    "message": _FORBIDDEN_TEMPLATE.format(scope=scope_text)}
        return None

    # Tool so sánh 2 cửa hàng trực tiếp (store_1/store_2, VD: compare_stores_revenue):
    # cả hai đều phải nằm trong phạm vi - không được phụ thuộc cờ "store_nbr" duy nhất.
    for _key in ("store_1", "store_2"):
        _s = tool_args.get(_key)
        if _s is not None and int(_s) not in identity_allowed:
            return {"status": "forbidden",
                    "message": _FORBIDDEN_TEMPLATE.format(scope=scope_text)}

    # So sánh cluster: mọi cửa hàng thuộc 2 cụm đều phải nằm trong phạm vi
    if tool_name == "compare_cluster_trends":
        involved: Set[int] = set()
        for c in ("cluster_1", "cluster_2"):
            if tool_args.get(c) is not None:
                involved |= _stores_in_cluster(int(tool_args[c]))
        if not involved <= identity_allowed:
            return {"status": "forbidden",
                    "message": _FORBIDDEN_TEMPLATE.format(scope=scope_text)}
        return None

    # get_sales_summary không có store_nbr -> agent sẽ chèn _allowed_stores (filter IN)
    return None
