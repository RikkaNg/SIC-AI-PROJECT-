# backend/src/routes/scenario_routes.py
"""
Scenario Lab API - chạy kịch bản what-if: sửa số liệu -> dự báo lại bằng
mô hình thật (ml_service) -> phân tích deterministic -> trả kết quả cho
UI "Kịch bản" và LLM tool run_scenario_analysis.

Mọi endpoint áp Row-Level Isolation theo cửa hàng của user (giống các route khác).
"""
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.src.security import Identity, ensure_store_access, get_current_user
from backend.src.services import scenario_service
from backend.src.services.scenario_service import ScenarioError

logger = logging.getLogger(__name__)
router = APIRouter()

# Thông điệp lỗi dữ liệu (khác lỗi ML service) -> map sang 404 thay vì 502
_DATA_ERROR_MARKERS = ("Không có", "Không tìm thấy", "trống")


class ScenarioRunRequest(BaseModel):
    store_nbr: int = Field(..., ge=1, le=999)
    family: str = Field(..., min_length=1, max_length=64)
    horizon_days: int = Field(16, ge=7, le=16)
    demand_multiplier: float = Field(1.0, ge=0.0, le=10.0,
                                     description="Hệ số nhu cầu: 1.5 = +50%.")
    promo_days: Optional[int] = Field(None, ge=0, le=16,
                                      description="Số ngày khuyến mãi trong kỳ (None = theo lịch thật).")
    oil_price: Optional[float] = Field(None, ge=20.0, le=200.0,
                                       description="Giá dầu USD (None = giá thật từ oil.csv).")
    traffic_change_pct: Optional[float] = Field(None, ge=-90.0, le=200.0,
                                                description="% thay đổi lưu lượng khách.")
    event_type: str = Field("none", pattern="^(none|holiday|earthquake)$",
                            description="Sự kiện bất ngờ đè lên lịch.")
    event_days: int = Field(0, ge=0, le=16)
    stock_override: Optional[float] = Field(None, ge=0.0,
                                            description="Tồn kho muốn giả lập (None = tồn thật).")
    lead_time_override: Optional[float] = Field(None, ge=0.0, le=60.0)


def _map_scenario_error(e: ScenarioError) -> HTTPException:
    msg = str(e)
    if any(marker in msg for marker in _DATA_ERROR_MARKERS):
        return HTTPException(status_code=404, detail=msg)
    return HTTPException(status_code=502, detail=msg)


@router.get("/scenario/meta")
def scenario_meta(store_nbr: int, family: str,
                  user: Identity = Depends(get_current_user)):
    """Giá trị mặc định để form kịch bản prefill (tồn, lead time, giá, promo baseline)."""
    ensure_store_access(user, store_nbr)
    try:
        return scenario_service.get_scenario_meta(store_nbr, family)
    except ScenarioError as e:
        raise _map_scenario_error(e)
    except Exception:
        logger.exception("Lỗi không xử lý được ở GET /api/scenario/meta")
        raise HTTPException(status_code=500, detail="Lỗi nội bộ khi đọc thông tin kịch bản.")


@router.post("/scenario/run")
def scenario_run(req: ScenarioRunRequest,
                 user: Identity = Depends(get_current_user)):
    """Chạy 1 kịch bản what-if trọn vẹn và trả series + KPI + phân tích + đề xuất."""
    ensure_store_access(user, req.store_nbr)
    try:
        return scenario_service.run_scenario(
            store_nbr=req.store_nbr,
            family=req.family,
            horizon_days=req.horizon_days,
            demand_multiplier=req.demand_multiplier,
            promo_days=req.promo_days,
            oil_price=req.oil_price,
            traffic_change_pct=req.traffic_change_pct,
            event_type=req.event_type,
            event_days=req.event_days,
            stock_override=req.stock_override,
            lead_time_override=req.lead_time_override,
        )
    except ScenarioError as e:
        raise _map_scenario_error(e)
    except Exception:
        logger.exception("Lỗi không xử lý được ở POST /api/scenario/run")
        raise HTTPException(status_code=500, detail="Lỗi nội bộ khi chạy kịch bản.")
