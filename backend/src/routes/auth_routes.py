# backend/src/routes/auth_routes.py
"""API xác thực: POST /api/auth/login, GET /api/auth/me."""
import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.src.security import Identity, authenticate, get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()


class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/login")
def login(req: LoginRequest):
    identity = authenticate(req.username.strip(), req.password)
    if identity is None:
        raise HTTPException(status_code=401, detail="Sai tên đăng nhập hoặc mật khẩu.")

    from backend.src.security import create_access_token
    token = create_access_token(
        username=identity.username,
        role=identity.role,
        display_name=identity.display_name,
        allowed_stores=identity.allowed_stores,
    )
    logger.info(f"Login OK: {identity.username} (role={identity.role})")
    return {
        "access_token": token,
        "token_type": "bearer",
        "username": identity.username,
        "role": identity.role,
        "display_name": identity.display_name,
        # stores = [] + role=admin nghĩa là được xem toàn hệ thống
        "stores": sorted(identity.allowed_stores) if identity.allowed_stores is not None else [],
    }


@router.get("/me")
def me(user: Identity = Depends(get_current_user)):
    return {
        "username": user.username,
        "role": user.role,
        "display_name": user.display_name,
        "stores": sorted(user.allowed_stores) if user.allowed_stores is not None else [],
        "all_stores": user.is_admin,
    }
