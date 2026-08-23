# backend/src/routes/inventory_routes.py
"""API quản lý tồn kho - CÓ Row-Level Isolation (đọc và ghi đều kiểm tra phạm vi)."""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.src.database.connection import get_inventory_db, update_stock_db
from backend.src.security import Identity, ensure_store_access, get_current_user

router = APIRouter()

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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/inventory/restock")
def restock_endpoint(req: RestockRequest,
                     user: Identity = Depends(get_current_user)):
    ensure_store_access(user, req.store_nbr)
    try:
        update_stock_db(req.store_nbr, req.item_nbr, req.quantity)
        return {"status": "success", "message": f"Đã thêm {req.quantity} cho SP {req.item_nbr}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
