# backend/src/services/ml_client.py
import httpx
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# URL của ml_service (chạy ở port 8001)
ML_SERVICE_URL = os.environ.get("ML_SERVICE_URL", "http://localhost:8001")

async def get_ml_forecast(store_nbr: int, family: str, features: Dict[str, Any]):
    """
    Gọi sang ML Service để lấy dự báo doanh số cho 1 sản phẩm (family) tại 1 cửa hàng.
    Khớp với SinglePredictRequest của ML Service.
    """
    # Cấu trúc payload gửi đi phải khớp 100% với main.py bên ml_service
    payload = {
        "store_nbr": store_nbr,
        "family": family,
        "features": features
    }
    
    async with httpx.AsyncClient() as client:
        try:
            # Gọi POST /predict ở ml_service
            response = await client.post(f"{ML_SERVICE_URL}/predict", json=payload, timeout=15.0)
            response.raise_for_status()  # Báo lỗi nếu status code là 4xx hoặc 5xx
            return response.json()
            
        except httpx.RequestError as e:
            logger.error(f"Lỗi kết nối tới ML Service: {e}")
            raise Exception("Không thể kết nối tới ML Service. Có thể service chưa chạy hoặc bị sập.")
        except httpx.HTTPStatusError as e:
            logger.error(f"Lỗi từ ML Service: {e.response.text}")
            raise Exception(f"ML Service trả về lỗi: {e.response.status_code} - {e.response.text}")