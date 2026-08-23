# backend/src/routes/forecast_routes.py
"""Forward dự báo sang ML Service - CÓ Row-Level Isolation."""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from backend.src.security import Identity, ensure_store_access, get_current_user
from backend.src.services.ml_client import get_ml_forecast

router = APIRouter()

class ForecastRequest(BaseModel):
    store_nbr: int
    family: str
    features: Dict[str, Any]  # Chứa các feature như lag_1, rolling_7, day_of_week...

@router.post("/forecast")
async def forecast_endpoint(req: ForecastRequest,
                            user: Identity = Depends(get_current_user)):
    """
    Endpoint nhận request dự báo từ Web UI và forward sang ML Service (:8001).
    Chỉ cho phép dự báo cửa hàng trong phạm vi của user.
    """
    ensure_store_access(user, req.store_nbr)
    try:
        forecast_data = await get_ml_forecast(
            store_nbr=req.store_nbr,
            family=req.family,
            features=req.features
        )
        return {"status": "success", "data": forecast_data}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Forecast Service Unavailable: {str(e)}")
