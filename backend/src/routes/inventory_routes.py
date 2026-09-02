# backend/src/routes/inventory_routes.py
"""API quản lý tồn kho - CÓ Row-Level Isolation (đọc và ghi đều kiểm tra phạm vi)."""
import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.src.database.connection import get_inventory_db, update_stock_db
from backend.src.security import Identity, ensure_store_access, get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()

# Thông điệp lỗi chung - chi tiết exception chỉ log phía server, không lộ ra client.
_GENERIC_500 = "Lỗi hệ thống khi cập nhật tồn kho. Vui lòng thử lại sau."

class RestockRequest(BaseModel):
    store_nbr: int
    item_nbr: int
    quantity: int

@router.get("/inventory/{store_nbr}/{item_nbr}")
def get_inventory_endpoint(store_nbr: int, item_nbr: int,
                           user: Identity = Depends(get_current_user)):
    ensure_store_access(user, store_nbr)
    try:
        data = get_inventory_db(store_nbr, item_nbr)
        if not data:
            raise HTTPException(status_code=404, detail="Không tìm thấy sản phẩm")
        return {"status": "success", "data": data}
    except HTTPException:
        raise
    except Exception:
        logger.exception("Lỗi đọc tồn kho store=%s item=%s", store_nbr, item_nbr)
        raise HTTPException(status_code=500, detail=_GENERIC_500)

@router.post("/inventory/restock")
def restock_endpoint(req: RestockRequest,
                     user: Identity = Depends(get_current_user)):
    ensure_store_access(user, req.store_nbr)
    try:
        update_stock_db(req.store_nbr, req.item_nbr, req.quantity)
        return {"status": "success", "message": f"Đã thêm {req.quantity} cho SP {req.item_nbr}"}
    except Exception:
        logger.exception("Lỗi restock store=%s item=%s qty=%s", req.store_nbr, req.item_nbr, req.quantity)
        raise HTTPException(status_code=500, detail=_GENERIC_500)
