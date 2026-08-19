# backend/src/routes/inventory_routes.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.src.database.connection import get_inventory_db, update_stock_db

router = APIRouter()

class RestockRequest(BaseModel):
    store_nbr: int
    item_nbr: int
    quantity: int

@router.get("/inventory/{store_nbr}/{item_nbr}")
def get_inventory_endpoint(store_nbr: int, item_nbr: int):
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
def restock_endpoint(req: RestockRequest):
    try:
        update_stock_db(req.store_nbr, req.item_nbr, req.quantity)
        return {"status": "success", "message": f"Đã thêm {req.quantity} cho SP {req.item_nbr}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))