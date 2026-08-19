# backend/src/routes/forecast_routes.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from backend.src.services.ml_client import get_ml_forecast

router = APIRouter()

class ForecastRequest(BaseModel):
    store_nbr: int
    family: str
    features: Dict[str, Any]  # Chứa các feature như lag_1, rolling_7, day_of_week...

@router.post("/forecast")
async def forecast_endpoint(req: ForecastRequest):
    """
    Endpoint nhận request dự báo từ Web UI và forward sang ML Service (:8001)
    """
    try:
        # Gọi sang ml_client.py
        forecast_data = await get_ml_forecast(
            store_nbr=req.store_nbr,
            family=req.family,
            features=req.features
        )
        return {"status": "success", "data": forecast_data}
    except Exception as e:
        # Bắt lỗi nếu ML Service bị sập hoặc không phản hồi
        raise HTTPException(status_code=503, detail=f"Forecast Service Unavailable: {str(e)}")