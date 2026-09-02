# backend/src/routes/product_routes.py
"""
API danh mục SẢN PHẨM CỤ THỂ (SKU-level) cho Dashboard - CÓ Row-Level Isolation.

Nguồn dữ liệu thật trong retail.db:
- items                  : danh mục 4.100 sản phẩm (item_nbr, family, class, perishable)
- inventory              : tồn kho theo cửa hàng (sinh từ sales velocity khi init DB)
- agg_item_store_sales   : tổng doanh số lịch sử 2016 theo cửa hàng × sản phẩm
                           (dựng bởi backend/scripts/build_sales_cache.py)
- forecasts              : dự báo theo ngày của model LightGBM

Endpoints:
- GET /api/products        -> trang danh sách sản phẩm (tìm kiếm/lọc/sắp xếp phân trang server-side)
- GET /api/top-products    -> top sản phẩm bán chạy theo doanh số 2016 (theo SKU cụ thể)
- GET /api/family-mix      -> thị phần theo nhóm hàng (pie chart)
- GET /api/family-trend    -> xu hướng dự báo theo ngày × nhóm hàng (line/area chart)
- GET /api/product-families-> danh sách nhóm hàng cho dropdown lọc
"""
import json
import logging
import math
import os
import sqlite3
import time
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.src.security import Identity, get_current_user, identity_scope_sql

logger = logging.getLogger(__name__)
router = APIRouter()

_ENV_DB_PATH = os.getenv("DB_PATH")
if _ENV_DB_PATH:
    DB_PATH = Path(_ENV_DB_PATH)
else:
    _CURRENT_DIR = Path(__file__).resolve().parent              # backend/src/routes
    DB_PATH = _CURRENT_DIR.parent / "database" / "retail.db"    # backend/src/database/retail.db


def _get_readonly_connection() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Database not found at {DB_PATH}. Vui lòng chạy init_database.py trước.",
        )
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


# ----------------------------------------------------------------------
# Cache TTL nhỏ cho các query nặng (family-trend quét bảng forecasts)
# ----------------------------------------------------------------------
_TTL_CACHE: dict = {}
_TTL_SECONDS = 300


def _ttl_get(key):
    hit = _TTL_CACHE.get(key)
    if hit and time.monotonic() - hit[0] < _TTL_SECONDS:
        return hit[1]
    return None


def _ttl_put(key, value):
    _TTL_CACHE[key] = (time.monotonic(), value)


# RMSLE toàn cục mặc định - CHỈ dùng khi bảng model_meta chưa có giá trị
# (chưa chạy build_sku_stats.py). Giá trị thật luôn được đọc từ DB.
_DEFAULT_GLOBAL_RMSLE = 0.357

# Ngưỡng tồn kho thấp CHO MỖI CỬA HÀNG. Khi xem nhiều cửa hàng (scope gộp),
# ngưỡng tổng = 30 × số cửa hàng trong phạm vi -> nhất quán với tô màu frontend.
LOW_STOCK_PER_STORE = 30


def _global_rmsle(conn: sqlite3.Connection) -> float:
    """Đọc global_rmsle từ bảng model_meta (build_sku_stats.py dựng), có cache TTL."""
    cached = _ttl_get(("global_rmsle",))
    if cached is not None:
        return float(cached)
    value = _DEFAULT_GLOBAL_RMSLE
    try:
        if _table_exists(conn, "model_meta"):
            row = conn.execute(
                "SELECT value FROM model_meta WHERE key = 'global_rmsle'"
            ).fetchone()
            if row and row["value"]:
                value = float(row["value"])
    except (sqlite3.Error, TypeError, ValueError):
        pass
    _ttl_put(("global_rmsle",), value)
    return value


# ======================================================================
# 1. DANH SÁCH SẢN PHẨM (Quản lý sản phẩm)
# ======================================================================

@router.get("/products")
def list_products(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    search: Optional[str] = Query(default=None, description="Tìm theo mã sản phẩm / nhóm hàng"),
    family: Optional[str] = Query(default=None, description="Lọc chính xác theo nhóm hàng"),
    status: Optional[str] = Query(default=None, pattern="^(active|outofstock)$"),
    sort: str = Query(default="sold", pattern="^(sold|stock|name|item)$"),
    order: str = Query(default="desc", pattern="^(asc|desc)$"),
    store_nbr: Optional[int] = Query(default=None),
    user: Identity = Depends(get_current_user),
) -> dict:
    """Danh mục sản phẩm thật từ retail.db: tồn kho gộp theo phạm vi + tổng bán 2016."""
    inv_scope_sql, inv_params = identity_scope_sql(user, store_nbr, alias="inventory")
    agg_scope_sql, agg_params = identity_scope_sql(user, store_nbr, alias="agg")

    # Bổ sung (sku_stats - ABC / dự báo 16 ngày / dải tin cậy): chỉ JOIN khi bảng đã dựng,
    # để API vẫn hoạt động trên database cũ chưa chạy build_sku_stats.py.
    sku_stats_ready = False
    global_rmsle = _DEFAULT_GLOBAL_RMSLE
    conn_pre = None
    try:
        conn_pre = _get_readonly_connection()
        sku_stats_ready = _table_exists(conn_pre, "sku_stats")
        if sku_stats_ready:
            global_rmsle = _global_rmsle(conn_pre)
    except HTTPException:
        raise
    finally:
        if conn_pre:
            conn_pre.close()

    core_params: list = []
    core_sql = f"""
        SELECT
            i.item_nbr                       AS item_nbr,
            i.name                           AS name,
            i.family                         AS family,
            i.class                          AS class_code,
            i.perishable                     AS perishable,
            CAST(COALESCE(v.stock, 0) AS INTEGER) AS stock,
            ROUND(COALESCE(s.unit_sales, 0), 1)   AS sold_2016,
            CASE WHEN COALESCE(v.stock, 0) <= 0 THEN 'outofstock' ELSE 'active' END AS status
    """
    if sku_stats_ready:
        core_sql += """,
            ss.abc_class                     AS abc_class,
            ROUND(COALESCE(ss.fc_total_16d, 0), 1) AS fc_total_16d,
            COALESCE(ss.family_rmsle, {_g})       AS family_rmsle
        """.replace("{_g}", repr(global_rmsle))
    core_sql += f"""
        FROM items i
        LEFT JOIN (
            SELECT item_nbr, SUM(current_stock) AS stock
            FROM inventory{inv_scope_sql}
            GROUP BY item_nbr
        ) v ON v.item_nbr = i.item_nbr
        LEFT JOIN (
            SELECT item_nbr, SUM(unit_sales) AS unit_sales
            FROM agg_item_store_sales agg{agg_scope_sql}
            GROUP BY item_nbr
        ) s ON s.item_nbr = i.item_nbr
    """
    if sku_stats_ready:
        core_sql += "\n        LEFT JOIN sku_stats ss ON ss.item_nbr = i.item_nbr\n"
    core_sql += "        WHERE 1=1\n"
    if search:
        core_sql += (" AND (CAST(i.item_nbr AS TEXT) LIKE ? OR UPPER(i.family) LIKE UPPER(?)"
                     " OR UPPER(i.name) LIKE UPPER(?))")
        like = f"%{search.strip()}%"
        core_params += [like, like, like]
    if family:
        core_sql += " AND i.family = ?"
        core_params.append(family.strip())
    if status:
        core_sql += " AND status = ?"
        core_params.append(status)

    sort_col = {
        "sold": "sold_2016", "stock": "stock",
        "name": "COALESCE(name, '#' || item_nbr)", "item": "item_nbr",
    }[sort]
    direction = "ASC" if order == "asc" else "DESC"
    outer_sql = f"""
        SELECT * FROM ({core_sql})
        ORDER BY {sort_col} {direction}, item_nbr ASC
        LIMIT ? OFFSET ?
    """

    conn = None
    try:
        conn = _get_readonly_connection()
        cache_ready = _table_exists(conn, "agg_item_store_sales")

        count_row = conn.execute(f"SELECT COUNT(*) AS c FROM ({core_sql})",
                                 [*inv_params, *agg_params, *core_params]).fetchone()
        total = int(count_row["c"])

        rows = conn.execute(
            outer_sql,
            [*inv_params, *agg_params, *core_params, page_size, (page - 1) * page_size],
        ).fetchall()
        def _sku_extra(r: sqlite3.Row) -> dict:
            """Dải tin cậy dự báo 16 ngày: fc × exp(±rmsle_family) (1-sigma log-space)."""
            fc = float(r["fc_total_16d"] or 0.0)
            sig = float(r["family_rmsle"]) if r["family_rmsle"] is not None else global_rmsle
            return {
                "abc_class": r["abc_class"],
                "fc_total_16d": fc,
                "fc_low": round(fc * math.exp(-sig), 1),
                "fc_high": round(fc * math.exp(sig), 1),
            }

        items = []
        for r in rows:
            item = {
                "item_nbr": int(r["item_nbr"]),
                "name": r["name"],
                "family": r["family"],
                "class_code": int(r["class_code"]) if r["class_code"] is not None else None,
                "perishable": int(r["perishable"] or 0),
                "stock": int(r["stock"]),
                "sold_2016": float(r["sold_2016"] or 0),
                "status": r["status"],
            }
            if sku_stats_ready:
                item.update(_sku_extra(r))
            else:
                item.update({"abc_class": None, "fc_total_16d": None, "fc_low": None, "fc_high": None})
            items.append(item)
        # Số cửa hàng trong phạm vi đang xem (1 nếu chọn store cụ thể) -> ngưỡng
        # tồn kho thấp tổng = LOW_STOCK_PER_STORE × n_stores, khớp cách gộp stock.
        stores_row = conn.execute(
            f"SELECT COUNT(DISTINCT store_nbr) AS c FROM inventory{inv_scope_sql}",
            inv_params,
        ).fetchone()
        n_stores = max(1, int(stores_row["c"] or 0))
        low_stock_threshold = LOW_STOCK_PER_STORE * n_stores

        low_stock = conn.execute("""
            SELECT COUNT(*) AS c FROM (
                SELECT COALESCE(SUM(v.stock), 0) AS stock
                FROM items i
                LEFT JOIN (SELECT item_nbr, SUM(current_stock) AS stock
                           FROM inventory{isql} GROUP BY item_nbr) v
                  ON v.item_nbr = i.item_nbr
                GROUP BY i.item_nbr HAVING stock > 0 AND stock < ?
            )
        """.replace("{isql}", inv_scope_sql), [*inv_params, low_stock_threshold]).fetchone()

        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": max(1, -(-total // page_size)),
            "low_stock_count": int(low_stock["c"]) if low_stock else 0,
            "low_stock_threshold": low_stock_threshold,
            "sales_cache_ready": cache_ready,
            "items": items,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in list_products")
        raise HTTPException(status_code=500,
                            detail="Lỗi hệ thống khi truy vấn dữ liệu. Vui lòng thử lại sau.")
    finally:
        if conn:
            conn.close()


# ======================================================================
# 2. TOP SẢN PHẨM BÁN CHẠY (SKU cụ thể, doanh số 2016)
# ======================================================================

@router.get("/top-products")
def get_top_products(
    limit: int = Query(default=8, ge=1, le=25),
    store_nbr: Optional[int] = Query(default=None),
    user: Identity = Depends(get_current_user),
) -> dict:
    """Top sản phẩm cụ thể (item_nbr) bán chạy nhất năm 2016 trong phạm vi user."""
    scope_sql, params = identity_scope_sql(user, store_nbr, alias="c")

    conn = None
    try:
        conn = _get_readonly_connection()
        if not _table_exists(conn, "agg_item_store_sales"):
            raise HTTPException(
                status_code=503,
                detail=("Bảng tổng hợp agg_item_store_sales chưa được dựng. "
                        "Chạy: python backend/scripts/build_sales_cache.py"),
            )

        # abc_class từ sku_stats (tùy chọn - DB cũ chưa dựng bảng vẫn chạy được)
        sku_stats_ready = _table_exists(conn, "sku_stats")
        ss_join = "\n            LEFT JOIN sku_stats ss ON ss.item_nbr = c.item_nbr" if sku_stats_ready else ""
        ss_col = ",\n                   ss.abc_class               AS abc_class" if sku_stats_ready else ""
        ss_group = ", ss.abc_class" if sku_stats_ready else ""

        rows = conn.execute(f"""
            SELECT c.item_nbr,
                   i.name,
                   i.family,
                   i.class                    AS class_code,
                   i.perishable{ss_col},
                   ROUND(SUM(c.unit_sales), 1) AS unit_sales
            FROM agg_item_store_sales c
            JOIN items i ON i.item_nbr = c.item_nbr{ss_join}
            {scope_sql}
            GROUP BY c.item_nbr, i.name, i.family, i.class, i.perishable{ss_group}
            ORDER BY unit_sales DESC
            LIMIT ?
        """, [*params, limit]).fetchall()

        total_row = conn.execute(f"""
            SELECT ROUND(SUM(c.unit_sales), 1) AS t
            FROM agg_item_store_sales c
            {scope_sql}
        """, params).fetchone()
        grand_total = float(total_row["t"] or 0) if total_row else 0.0

        products = []
        for idx, r in enumerate(rows, start=1):
            units = float(r["unit_sales"] or 0)
            product = {
                "rank": idx,
                "item_nbr": int(r["item_nbr"]),
                "name": r["name"],
                "family": r["family"],
                "class_code": int(r["class_code"]) if r["class_code"] is not None else None,
                "perishable": int(r["perishable"] or 0),
                "unit_sales": units,
                "share_pct": round(units / grand_total * 100, 1) if grand_total > 0 else 0.0,
            }
            if sku_stats_ready:
                product["abc_class"] = r["abc_class"]
            products.append(product)
        return {"period": "2016", "scope": store_nbr or "all", "items": products}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in get_top_products")
        raise HTTPException(status_code=500,
                            detail="Lỗi hệ thống khi truy vấn dữ liệu. Vui lòng thử lại sau.")
    finally:
        if conn:
            conn.close()


# ======================================================================
# 3. THỊ PHẦN NHÓM HÀNG (Pie chart)
# ======================================================================

@router.get("/family-mix")
def get_family_mix(
    top: int = Query(default=4, ge=1, le=12),
    store_nbr: Optional[int] = Query(default=None),
    user: Identity = Depends(get_current_user),
) -> dict:
    """Thị phần doanh số 2016 theo nhóm hàng (Top N + 'Khác') trong phạm vi user."""
    scope_sql, params = identity_scope_sql(user, store_nbr, alias="c")

    conn = None
    try:
        conn = _get_readonly_connection()
        if not _table_exists(conn, "agg_item_store_sales"):
            raise HTTPException(
                status_code=503,
                detail=("Bảng tổng hợp agg_item_store_sales chưa được dựng. "
                        "Chạy: python backend/scripts/build_sales_cache.py"),
            )

        rows = conn.execute(f"""
            SELECT i.family,
                   ROUND(SUM(c.unit_sales), 1) AS unit_sales
            FROM agg_item_store_sales c
            JOIN items i ON i.item_nbr = c.item_nbr
            {scope_sql}
            GROUP BY i.family
            ORDER BY unit_sales DESC
        """, params).fetchall()

        data = [{"name": r["family"], "value": round(float(r["unit_sales"] or 0))}
                for r in rows]
        if len(data) > top:
            head, tail = data[:top], data[top:]
            others = sum(d["value"] for d in tail)
            data = head + ([{"name": "Khác", "value": others}] if others > 0 else [])
        return {"period": "2016", "items": data}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in get_family_mix")
        raise HTTPException(status_code=500,
                            detail="Lỗi hệ thống khi truy vấn dữ liệu. Vui lòng thử lại sau.")
    finally:
        if conn:
            conn.close()


# ======================================================================
# 4. XU HƯỚNG DỰ BÁO THEO NHÓM HÀNG (Area chart, dữ liệu model thật)
# ======================================================================

@router.get("/family-trend")
def get_family_trend(
    days: int = Query(default=16, ge=1, le=31),
    top_families: int = Query(default=6, ge=1, le=10),
    store_nbr: Optional[int] = Query(default=None),
    user: Identity = Depends(get_current_user),
) -> dict:
    """
    Chuỗi thời gian dự báo (model LightGBM) theo ngày × nhóm hàng trong kỳ dự báo.
    Ưu tiên bảng tổng hợp agg_forecast_date_family (nhanh, tức thì);
    fallback quét trực tiếp forecasts nếu bảng chưa được dựng.
    """
    scope_sql, params = identity_scope_sql(user, store_nbr, alias="f")
    cache_key = ("family_trend", tuple(params), scope_sql, days, top_families)
    cached = _ttl_get(cache_key)
    if cached is not None:
        return cached

    conn = None
    try:
        conn = _get_readonly_connection()
        bounds = conn.execute(
            "SELECT MIN(date) AS dmin, MAX(date) AS dmax FROM forecasts"
        ).fetchone()
        if not bounds or not bounds["dmin"]:
            raise HTTPException(status_code=404, detail="Không có dữ liệu dự báo.")

        dmin, dmax = bounds["dmin"], bounds["dmax"]
        has_agg = _table_exists(conn, "agg_forecast_date_family")

        if has_agg:
            # Bản tổng hợp: scope theo cột store_nbr của bảng agg.
            # Lưu ý: sau "WHERE 1=1" phải dùng dạng "AND ..." chứ không phải "WHERE ..." thứ hai.
            agg_scope_sql, agg_params = identity_scope_sql(user, store_nbr, alias="a")
            agg_scope_and_sql = agg_scope_sql.replace(" WHERE ", " AND ", 1)
            window_agg_sql = """
                AND a.date >= (SELECT MIN(date) FROM agg_forecast_date_family)
                AND a.date <= date((SELECT MIN(date) FROM agg_forecast_date_family), '+' || ? || ' day')
            """

            top_rows = conn.execute(f"""
                SELECT a.family, ROUND(SUM(a.predicted_sales), 1) AS total
                FROM agg_forecast_date_family a
                WHERE 1=1 {window_agg_sql} {agg_scope_and_sql}
                GROUP BY a.family ORDER BY total DESC LIMIT ?
            """, [days - 1, *agg_params, top_families]).fetchall()
            families = [r["family"] for r in top_rows]
            if not families:
                raise HTTPException(status_code=404,
                                    detail="Không có dữ liệu dự báo cho phạm vi này.")

            fam_marks = ",".join("?" * len(families))
            series_rows = conn.execute(f"""
                SELECT a.date,
                       a.family,
                       ROUND(SUM(a.predicted_sales), 1) AS predicted
                FROM agg_forecast_date_family a
                WHERE 1=1 {window_agg_sql} {agg_scope_and_sql}
                  AND a.family IN ({fam_marks})
                GROUP BY a.date, a.family
                ORDER BY a.date
            """, [days - 1, *agg_params, *families]).fetchall()

            by_date: dict = {}
            for r in series_rows:
                by_date.setdefault(r["date"], {})[r["family"]] = round(float(r["predicted"] or 0))

            result = {
                "horizon_days": days,
                "date_from": dmin,
                "date_to": dmax,
                "families": families,
                "series": [{"date": d, **by_date[d]} for d in sorted(by_date)],
            }
            _ttl_put(cache_key, result)
            return result

        # ---- Fallback: quét trực tiếp forecasts (chậm hơn, chỉ dùng khi thiếu cache) ----
        scope_and_sql = scope_sql.replace(" WHERE ", " AND ", 1)
        window_sql = """
            AND f.date >= (SELECT MIN(date) FROM forecasts)
            AND f.date <= date((SELECT MIN(date) FROM forecasts), '+' || ? || ' day')
        """

        top_rows = conn.execute(f"""
            SELECT i.family, ROUND(SUM(f.predicted_sales), 1) AS total
            FROM forecasts f
            JOIN items i ON i.item_nbr = f.item_nbr
            WHERE 1=1 {window_sql} {scope_and_sql}
            GROUP BY i.family ORDER BY total DESC LIMIT ?
        """, [days - 1, *params, top_families]).fetchall()
        families = [r["family"] for r in top_rows]

        if not families:
            raise HTTPException(status_code=404, detail="Không có dữ liệu dự báo cho phạm vi này.")

        fam_marks = ",".join("?" * len(families))
        series_rows = conn.execute(f"""
            SELECT f.date,
                   i.family,
                   ROUND(SUM(f.predicted_sales), 1) AS predicted
            FROM forecasts f
            JOIN items i ON i.item_nbr = f.item_nbr
            WHERE 1=1 {window_sql} {scope_and_sql}
              AND i.family IN ({fam_marks})
            GROUP BY f.date, i.family
            ORDER BY f.date
        """, [days - 1, *params, *families]).fetchall()

        by_date: dict = {}
        for r in series_rows:
            row = by_date.setdefault(r["date"], {})
            row[r["family"]] = round(float(r["predicted"] or 0))

        result = {
            "horizon_days": days,
            "date_from": dmin,
            "date_to": dmax,
            "families": families,
            "series": [
                {"date": d, **by_date[d]}
                for d in sorted(by_date)
            ],
        }
        _ttl_put(cache_key, result)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in get_family_trend")
        raise HTTPException(status_code=500,
                            detail="Lỗi hệ thống khi truy vấn dữ liệu. Vui lòng thử lại sau.")
    finally:
        if conn:
            conn.close()


# ======================================================================
# 5. DANH SÁCH NHÓM HÀNG (dropdown lọc)
# ======================================================================

# ======================================================================
# CHỈ SỐ KINH DOANH LỊCH SỬ (/api/metrics/history)
# ======================================================================
@router.get("/metrics/history")
def get_business_metrics(
    date_from: Optional[str] = Query(default=None),
    date_to: Optional[str] = Query(default=None),
    store_nbr: Optional[int] = Query(default=None),
    user: Identity = Depends(get_current_user),
) -> dict:
    """
    Chuỗi chỉ số kinh doanh theo NGÀY (USD, giá THAM CHIẾU - dataset gốc không có giá):
      revenue      = SUM(số_lượng × giá_bán_tham_chiếu theo nhóm hàng)
      returns      = revenue × tỷ_lệ_trả_hàng(theo nhóm)
      gross_profit = revenue - returns - cogs (giá vốn tính trên số bán ròng)
    Nguồn: bảng tổng hợp agg_daily_business dựng một lần bởi
            backend/scripts/build_business_cache.py.
    Áp Row-Level Isolation như mọi endpoint khác.
    """
    scope_sql, params = identity_scope_sql(user, store_nbr, alias="b")
    scope_and_sql = scope_sql.replace(" WHERE ", " AND ", 1)
    cache_key = ("biz_history", tuple(params), scope_sql, date_from, date_to)
    cached = _ttl_get(cache_key)
    if cached is not None:
        return cached

    conn = None
    try:
        conn = _get_readonly_connection()
        if not _table_exists(conn, "agg_daily_business"):
            raise HTTPException(
                status_code=503,
                detail=("Chưa có bảng agg_daily_business trong DB. "
                        "Chạy một lần: python backend/scripts/build_business_cache.py"))

        rows = conn.execute(f"""
            SELECT b.date AS date,
                   ROUND(SUM(b.revenue), 2) AS revenue,
                   ROUND(SUM(b.returns), 2) AS returns,
                   ROUND(SUM(b.cogs), 2) AS cogs,
                   COALESCE(SUM(t.n_invoices), 0) AS invoices
            FROM agg_daily_business b
            LEFT JOIN daily_transactions t
                   ON t.date = b.date AND t.store_nbr = b.store_nbr
            WHERE 1=1
              {scope_and_sql}
              {"AND b.date >= ?" if date_from else ""}
              {"AND b.date <= ?" if date_to else ""}
            GROUP BY b.date ORDER BY b.date
        """, [*params, *([date_from] if date_from else []),
              *([date_to] if date_to else [])]).fetchall()

        items = []
        for r in rows:
            rev = float(r["revenue"] or 0)
            ret = float(r["returns"] or 0)
            cogs = float(r["cogs"] or 0)
            items.append({
                "date": r["date"],
                "revenue": rev,
                "returns": ret,
                "gross_profit": round(rev - ret - cogs, 2),
                "cogs": cogs,
                "invoices": int(r["invoices"] or 0),
            })
        result = {"currency": "USD", "prices": "reference", "items": items}
        _ttl_put(cache_key, result)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in get_business_metrics")
        raise HTTPException(
            status_code=500,
            detail="Lỗi hệ thống khi truy vấn chỉ số kinh doanh.")
    finally:
        if conn:
            conn.close()


@router.get("/product-families")
def list_families(user: Identity = Depends(get_current_user)) -> List[str]:
    """Danh sách nhóm hàng (family) có trong danh mục sản phẩm."""
    conn = None
    try:
        conn = _get_readonly_connection()
        rows = conn.execute(
            "SELECT DISTINCT family FROM items ORDER BY family"
        ).fetchall()
        return [r["family"] for r in rows]
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in list_families")
        raise HTTPException(status_code=500,
                            detail="Lỗi hệ thống khi truy vấn dữ liệu. Vui lòng thử lại sau.")
    finally:
        if conn:
            conn.close()


# ======================================================================
# 6. VALIDATION CHẤT LƯỢNG SKU-LEVEL (proxy theo lớp ABC)
# ======================================================================

@router.get("/model/validation")
def get_model_validation(user: Identity = Depends(get_current_user)) -> dict:
    """
    Chất lượng dự báo SKU-level (validation proxy) do build_sku_stats.py tính:
    độ lệch giữa 2 cửa sổ 45 ngày liên tiếp, nhóm theo lớp ABC.
    Trả về rỗng nếu database chưa dựng model_meta (chưa chạy build_sku_stats.py).
    """
    conn = None
    try:
        conn = _get_readonly_connection()
        if not _table_exists(conn, "model_meta"):
            return {"ready": False,
                    "detail": "Chưa có validation. Chạy: python ml_training/src/build_sku_stats.py"}
        meta = {r["key"]: r["value"] for r in
                conn.execute("SELECT key, value FROM model_meta").fetchall()}
        try:
            by_class = json.loads(meta.get("sku_validation", "{}"))
        except json.JSONDecodeError:
            by_class = {}
        return {
            "ready": True,
            "global_rmsle": float(meta.get("global_rmsle", _DEFAULT_GLOBAL_RMSLE)),
            "built_at": meta.get("built_at"),
            "method": ("Độ lệch % giữa cửa sổ 45 ngày cuối so với 45 ngày liền trước "
                       "(w1 vs w0), tổng hợp theo từng SKU rồi nhóm theo lớp ABC"),
            "by_abc_class": by_class,
        }
    except Exception as e:
        logger.exception("Error in get_model_validation")
        raise HTTPException(status_code=500,
                            detail="Lỗi hệ thống khi truy vấn dữ liệu. Vui lòng thử lại sau.")
    finally:
        if conn:
            conn.close()
