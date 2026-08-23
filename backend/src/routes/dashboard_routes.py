# backend/src/routes/dashboard_routes.py
"""
API cho Dashboard React (App.tsx) - CÓ Row-Level Isolation:
- GET /api/stores                 -> [1, 2, ...]  chỉ các cửa hàng trong phạm vi user
- GET /api/predictions            -> [{date, predicted_sales}, ...] (tổng theo ngày)
       Tham số tùy chọn: store_nbr, family (tên nhóm hàng từ bảng items),
       date_from / date_to (YYYY-MM-DD) để cắt khoảng thời gian.
- GET /api/kpi                    -> {total_predicted_sales, avg_per_day, forecast_days}
       Cùng bộ tham số tùy chọn như /predictions.
- GET /api/forecast-meta          -> {date_from, date_to} biên thời gian của dữ liệu dự báo
       (theo phạm vi user + family nếu truyền) để frontend giới hạn ô chọn ngày.

Mọi query đều filter theo Identity.allowed_stores (admin/ERP = toàn hệ thống).
Đọc trực tiếp SQLite retail.db ở chế độ read-only, không cần pandas.
"""
import os
import sqlite3
import logging
from pathlib import Path
from typing import List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.src.security import Identity, get_current_user, store_filter

logger = logging.getLogger(__name__)
router = APIRouter()

# Resolve DB_PATH giống hệt llm_agent/tools.py: env trước, file cạnh module sau
_ENV_DB_PATH = os.getenv("DB_PATH")
if _ENV_DB_PATH:
    DB_PATH = Path(_ENV_DB_PATH)
else:
    _CURRENT_DIR = Path(__file__).resolve().parent          # backend/src/routes
    DB_PATH = _CURRENT_DIR.parent / "database" / "retail.db"  # backend/src/database/retail.db


def _get_readonly_connection() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Database not found at {DB_PATH}. Vui lòng chạy init_database.py trước.",
        )
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _append_condition(where: str, params: list, cond_sql: str, value) -> Tuple[str, list]:
    """Nối thêm điều kiện AND vào mệnh đề WHERE đang dựng dần."""
    where = f"{where} AND {cond_sql}" if where else f" WHERE {cond_sql}"
    return where, params + [value]


def _build_forecast_filters(
    user: Identity,
    store_nbr: Optional[int],
    family: Optional[str],
    date_from: Optional[str],
    date_to: Optional[str],
) -> Tuple[str, str, list]:
    """
    Dựng (join_sql, where, params) chung cho các endpoint đọc bảng forecasts.
    - store_filter luôn được áp dụng TRƯỚC (Row-Level Isolation không thể bị bỏ qua).
    - family -> JOIN items; date_from/date_to so sánh chuỗi ISO YYYY-MM-DD.
    """
    where, params = store_filter(user, store_nbr)
    join_sql = ""
    if family:
        join_sql = " JOIN items ON items.item_nbr = forecasts.item_nbr"
        where, params = _append_condition(where, params, "items.family = ?", family)
    if date_from:
        where, params = _append_condition(where, params, "forecasts.date >= ?", date_from)
    if date_to:
        where, params = _append_condition(where, params, "forecasts.date <= ?", date_to)
    return join_sql, where, params


@router.get("/stores")
def list_stores(user: Identity = Depends(get_current_user)) -> List[int]:
    """Danh sách mã cửa hàng trong PHẠM VI của user (frontend dùng để render tab chi nhánh)."""
    where, params = store_filter(user, None)
    conn = None
    try:
        conn = _get_readonly_connection()
        rows = conn.execute(
            f"SELECT DISTINCT store_nbr FROM forecasts{where} ORDER BY store_nbr", params
        ).fetchall()
        return [int(r["store_nbr"]) for r in rows]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in list_stores: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()


@router.get("/predictions")
def get_predictions(store_nbr: Optional[int] = Query(default=None),
                    family: Optional[str] = Query(default=None),
                    date_from: Optional[str] = Query(default=None),
                    date_to: Optional[str] = Query(default=None),
                    user: Identity = Depends(get_current_user)) -> List[dict]:
    """Tổng doanh số dự báo theo từng ngày (lọc cửa hàng trong phạm vi user,
    tùy chọn lọc ngành hàng và khoảng ngày)."""
    join_sql, where, params = _build_forecast_filters(user, store_nbr, family, date_from, date_to)
    conn = None
    try:
        conn = _get_readonly_connection()
        query = f"""
            SELECT forecasts.date              AS date,
                   ROUND(SUM(forecasts.predicted_sales), 2) AS predicted_sales
            FROM forecasts{join_sql}{where}
            GROUP BY forecasts.date ORDER BY forecasts.date
        """
        rows = conn.execute(query, params).fetchall()
        if not rows:
            raise HTTPException(status_code=404,
                                detail="Không có dữ liệu dự báo cho bộ lọc này (thử bỏ lọc ngành hàng hoặc mở rộng khoảng thời gian).")
        return [{"date": r["date"], "predicted_sales": float(r["predicted_sales"])} for r in rows]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()


@router.get("/kpi")
def get_kpi(store_nbr: Optional[int] = Query(default=None),
            family: Optional[str] = Query(default=None),
            date_from: Optional[str] = Query(default=None),
            date_to: Optional[str] = Query(default=None),
            user: Identity = Depends(get_current_user)) -> dict:
    """KPI tổng quan (tổng dự báo, trung bình mỗi ngày) theo cùng bộ lọc của /predictions."""
    join_sql, where, params = _build_forecast_filters(user, store_nbr, family, date_from, date_to)
    conn = None
    try:
        conn = _get_readonly_connection()
        query = f"""
            SELECT ROUND(SUM(forecasts.predicted_sales), 2) AS total,
                   COUNT(DISTINCT forecasts.date)           AS n_days
            FROM forecasts{join_sql}{where}
        """
        row = conn.execute(query, params).fetchone()
        if row is None or row["total"] is None or not row["n_days"]:
            raise HTTPException(status_code=404,
                                detail="Không có dữ liệu dự báo cho bộ lọc này (thử bỏ lọc ngành hàng hoặc mở rộng khoảng thời gian).")

        total = float(row["total"])
        n_days = int(row["n_days"])
        return {
            "total_predicted_sales": round(total, 2),
            "avg_per_day": round(total / n_days, 2),
            "forecast_days": n_days,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_kpi: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()


@router.get("/forecast-meta")
def get_forecast_meta(store_nbr: Optional[int] = Query(default=None),
                      family: Optional[str] = Query(default=None),
                      user: Identity = Depends(get_current_user)) -> dict:
    """Biên thời gian (MIN/MAX date) của dữ liệu dự báo trong phạm vi user,
    dùng để giới hạn ô chọn 'Từ ngày/Đến ngày' phía frontend."""
    join_sql, where, params = _build_forecast_filters(user, store_nbr, family, None, None)
    conn = None
    try:
        conn = _get_readonly_connection()
        query = f"""
            SELECT MIN(forecasts.date) AS date_from,
                   MAX(forecasts.date) AS date_to
            FROM forecasts{join_sql}{where}
        """
        row = conn.execute(query, params).fetchone()
        if row is None or not row["date_from"]:
            raise HTTPException(status_code=404, detail="Không có dữ liệu dự báo.")
        return {"date_from": row["date_from"], "date_to": row["date_to"]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_forecast_meta: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()
