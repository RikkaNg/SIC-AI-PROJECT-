# backend/src/routes/dashboard_routes.py
"""
API cho Dashboard React (App.tsx) - CÓ Row-Level Isolation:
- GET /api/stores                 -> [1, 2, ...]  chỉ các cửa hàng trong phạm vi user
- GET /api/predictions?store_nbr= -> [{date, predicted_sales}, ...] (tổng theo ngày)
- GET /api/kpi?store_nbr=         -> {total_predicted_sales, avg_per_day}

Mọi query đều filter theo Identity.allowed_stores (admin/ERP = toàn hệ thống).
Đọc trực tiếp SQLite retail.db ở chế độ read-only, không cần pandas.
"""
import os
import sqlite3
import logging
from pathlib import Path
from typing import List, Optional

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
                    user: Identity = Depends(get_current_user)) -> List[dict]:
    """Tổng doanh số dự báo theo từng ngày (chỉ trong phạm vi cửa hàng của user)."""
    where, params = store_filter(user, store_nbr)
    conn = None
    try:
        conn = _get_readonly_connection()
        query = f"""
            SELECT date, ROUND(SUM(predicted_sales), 2) AS predicted_sales
            FROM forecasts{where}
            GROUP BY date ORDER BY date
        """
        rows = conn.execute(query, params).fetchall()
        if not rows:
            raise HTTPException(status_code=404, detail="Không có dữ liệu dự báo.")
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
            user: Identity = Depends(get_current_user)) -> dict:
    """KPI tổng quan trong phạm vi của user: tổng dự báo toàn kỳ và trung bình mỗi ngày."""
    where, params = store_filter(user, store_nbr)
    conn = None
    try:
        conn = _get_readonly_connection()
        query = f"""
            SELECT ROUND(SUM(predicted_sales), 2) AS total,
                   COUNT(DISTINCT date)           AS n_days
            FROM forecasts{where}
        """
        row = conn.execute(query, params).fetchone()
        if row is None or row["total"] is None or not row["n_days"]:
            raise HTTPException(status_code=404, detail="Không có dữ liệu dự báo.")

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
