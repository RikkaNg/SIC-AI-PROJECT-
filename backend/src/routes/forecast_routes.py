# backend/src/routes/forecast_routes.py
"""Forward dự báo sang ML Service - CÓ Row-Level Isolation.

Graceful degradation (§4.3): khi ML Service không phản hồi, fallback về bảng
`forecasts` precomputed trong SQLite (kết quả tổng theo ngày của family) thay vì
trả 503 - hệ thống vẫn có dữ liệu dự báo hợp lệ để hiển thị.
"""
import logging
import os
import sqlite3
from datetime import date
from pathlib import Path
from typing import Dict, List, Union

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.src.security import Identity, ensure_store_access, get_current_user
from backend.src.services.ml_client import get_ml_forecast, MLServiceError

logger = logging.getLogger(__name__)
router = APIRouter()

# Resolve DB_PATH giống dashboard_routes.py: env trước, file cạnh module sau
_ENV_DB_PATH = os.getenv("DB_PATH")
if _ENV_DB_PATH:
    DB_PATH = Path(_ENV_DB_PATH)
else:
    _CURRENT_DIR = Path(__file__).resolve().parent             # backend/src/routes
    DB_PATH = _CURRENT_DIR.parent / "database" / "retail.db"   # backend/src/database/retail.db

# Số ngày dự báo tối đa trả về từ bảng precomputed khi fallback
FALLBACK_MAX_DAYS = 16

# Feature đầu vào của preprocessor (lag_1, rolling_7, day_of_week...) là số hoặc
# nhãn phân loại - giới hạn kiểu scalar để không forward payload tùy ý sang ML service.
FeatureValue = Union[int, float, str]


class ForecastRequest(BaseModel):
    store_nbr: int
    family: str
    features: Dict[str, FeatureValue]


MAX_FEATURES = 64


def _precomputed_family_forecast(store_nbr: int, family: str) -> List[dict]:
    """Tổng dự báo theo ngày cho 1 family tại 1 cửa hàng từ bảng forecasts
    precomputed (item-level, group lại theo ngày). Không raise - trả list rỗng
    nếu DB thiếu/không có dữ liệu để caller xử lý thống nhất."""
    if not DB_PATH.exists():
        logger.error(f"Fallback thất bại: không tìm thấy DB tại {DB_PATH}")
        return []
    conn = None
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT f.date AS date,
                   ROUND(SUM(f.predicted_sales), 4) AS predicted_sales
            FROM forecasts f
            JOIN items it ON f.item_nbr = it.item_nbr
            WHERE f.store_nbr = ? AND it.family = ?
            GROUP BY f.date
            ORDER BY f.date
            LIMIT ?
            """,
            (int(store_nbr), family, FALLBACK_MAX_DAYS),
        ).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.Error as e:
        logger.error(f"Lỗi SQLite khi fallback bảng forecasts: {e}")
        return []
    finally:
        if conn:
            conn.close()


@router.post("/forecast")
async def forecast_endpoint(req: ForecastRequest,
                            user: Identity = Depends(get_current_user)):
    """
    Endpoint nhận request dự báo và forward sang ML Service (:8001).
    Chỉ cho phép dự báo cửa hàng trong phạm vi của user.

    ML Service sập -> trả source=precomputed_fallback với dữ liệu từ bảng
    forecasts thay vì 503; chỉ 503 khi cả hai nguồn đều không khả dụng.
    """
    ensure_store_access(user, req.store_nbr)
    if not req.features or len(req.features) > MAX_FEATURES:
        raise HTTPException(
            status_code=422,
            detail=f"features phải có từ 1 đến {MAX_FEATURES} trường.",
        )
    try:
        forecast_data = await get_ml_forecast(
            store_nbr=req.store_nbr,
            family=req.family,
            features=req.features
        )
        return {"status": "success", "source": "live_ml_service", "data": forecast_data}
    except HTTPException:
        raise
    except MLServiceError:
        logger.exception("ML Service không khả dụng - fallback về bảng forecasts precomputed")
        fallback_rows = _precomputed_family_forecast(req.store_nbr, req.family)
        if fallback_rows:
            dates = [row["date"] for row in fallback_rows]
            return {
                "status": "fallback",
                "source": "precomputed_fallback",
                "data": {
                    "store_nbr": req.store_nbr,
                    "family": req.family,
                    "series": fallback_rows,
                    "date_from": dates[0],
                    "date_to": dates[-1],
                    "as_of": date.today().isoformat(),
                },
            }
        raise HTTPException(status_code=503,
                            detail="Dịch vụ dự báo tạm thời không khả dụng. Vui lòng thử lại sau.")
    except Exception:
        logger.exception("Forward /forecast sang ML service thất bại")
        raise HTTPException(status_code=503,
                            detail="Dịch vụ dự báo tạm thời không khả dụng. Vui lòng thử lại sau.")
