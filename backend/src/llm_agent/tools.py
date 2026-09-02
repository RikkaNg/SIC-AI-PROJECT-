"""
tools.py
Bộ công cụ Function Calling & Phân tích Tồn kho cho LLM Agent.
"""

import os
import sqlite3
import logging
import math
import threading
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

# Scenario Lab (dự báo lại với số liệu chỉnh tay) - module này không phụ thuộc
# ngược lại llm_agent nên import thẳng được.
from backend.src.services.scenario_service import (
    ScenarioError,
    run_scenario as _run_scenario_impl,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ======================================================================
# CẤU HÌNH ĐƯỜNG DẪN DATABASE LINH HOẠT
# ======================================================================
ENV_DB_PATH = os.getenv("DB_PATH")
if ENV_DB_PATH:
    DB_PATH = Path(ENV_DB_PATH)
else:
    # Tự động dò tìm file retail.db
    CURRENT_DIR = Path(__file__).resolve().parent
    DB_PATH = CURRENT_DIR.parent / "database" / "retail.db"
    if not DB_PATH.exists():
        DB_PATH = CURRENT_DIR.parent.parent / "src" / "database" / "retail.db"


class ReusableReadOnlyConnection(sqlite3.Connection):
    """
    Kết nối SQLite read-only tái sử dụng (§4.4).

    Toàn bộ tool call kết thúc bằng conn.close() - với kết nối thread-local dùng
    chung, close() phải là no-op để call site hiện tại không cần sửa. Dọn dẹp
    thật khi process thoát qua close_all_db_connections().
    """

    def close(self) -> None:  # noqa: D102 - no-op có chủ đích, xem docstring lớp
        pass

    def really_close(self) -> None:
        super().close()


_db_thread_local = threading.local()


def get_db_connection() -> sqlite3.Connection:
    """Kết nối SQLite read-only/WAL tái sử dụng theo thread (FastAPI threadpool).

    Mỗi thread worker giữ đúng 1 kết nối; gọi lại chỉ chạy SELECT 1 để kiểm tra
    kết nối còn sống, rẻ hơn nhiều so với mở connection mới mỗi tool call.
    """
    conn: Optional[ReusableReadOnlyConnection] = getattr(_db_thread_local, "conn", None)
    if conn is not None:
        try:
            conn.execute("SELECT 1")
            return conn
        except sqlite3.Error:
            _drop_thread_connection(conn)

    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at {DB_PATH}. Vui lòng chạy init_database.py trước.")
    conn = sqlite3.connect(
        f"file:{DB_PATH}?mode=ro", uri=True,
        factory=ReusableReadOnlyConnection, check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    _db_thread_local.conn = conn
    return conn


def _drop_thread_connection(conn: ReusableReadOnlyConnection) -> None:
    """Đóng hẳn kết nối hỏng và gỡ khỏi thread-local."""
    try:
        conn.really_close()
    except sqlite3.Error:
        pass
    _db_thread_local.conn = None


def close_all_db_connections() -> None:
    """Đóng kết nối của thread hiện tại (gọi lúc shutdown nếu cần)."""
    conn: Optional[ReusableReadOnlyConnection] = getattr(_db_thread_local, "conn", None)
    if conn is not None:
        _drop_thread_connection(conn)


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """Kiểm tra bảng tồn tại trong DB (mở read-only nên chỉ tra sqlite_master)."""
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table_name,)
    ).fetchone()
    return row is not None


def _forbidden_empty_scope() -> Dict[str, str]:
    """Kết quả chuẩn khi user chưa được gán cửa hàng nào (scope RLS rỗng)."""
    return {"status": "forbidden",
            "message": "forbidden: Bạn chưa được gán cửa hàng nào. Liên hệ quản trị viên."}


def _scope_filter_sql(store_nbr: Optional[int], _allowed_stores: Optional[frozenset],
                      column: str = "store_nbr") -> tuple:
    """
    Sinh mảnh WHERE cho Row-Level Isolation của tool có store_nbr tùy chọn.
    Trả về (where_sql, params, error_dict) - error_dict khác None nghĩa là bị chặn.
    """
    if store_nbr is not None:
        return f" AND {column} = ?", [int(store_nbr)], None
    if _allowed_stores is not None:
        if not _allowed_stores:
            return "", [], _forbidden_empty_scope()
        marks = ",".join("?" * len(_allowed_stores))
        return f" AND {column} IN ({marks})", sorted(int(s) for s in _allowed_stores), None
    return "", [], None


# ======================================================================
# 1. NHÓM THỐNG KÊ & DỰ BÁO DOANH SỐ
# ======================================================================

def get_sales_summary(store_nbr: Optional[int] = None, days: int = 7,
                      _allowed_stores: Optional[frozenset] = None) -> Dict[str, Any]:
    """
    Tính tổng doanh số dự báo trong N ngày đầu tiên của chu kỳ dự báo.

    Row-Level Isolation: khi không chỉ định store_nbr và có @_allowed_stores
    (do agent chèn theo phạm vi user), chỉ tổng hợp các cửa hàng được phép.
    """
    conn = None
    try:
        conn = get_db_connection()
        # Sử dụng date() chuẩn SQLite để tính khoảng ngày
        query = """
            SELECT f.item_nbr, it.name, it.family, SUM(f.predicted_sales) as total_sales
            FROM forecasts f
            LEFT JOIN items it ON f.item_nbr = it.item_nbr
            WHERE f.date >= (SELECT MIN(date) FROM forecasts)
              AND f.date < date((SELECT MIN(date) FROM forecasts), '+' || ? || ' day')
        """
        params = [days]

        if store_nbr is not None:
            query += " AND f.store_nbr = ?"
            params.append(store_nbr)
        elif _allowed_stores is not None:
            # RLS: giới hạn tổng hợp trong phạm vi cửa hàng của user.
            # LƯU Ý: phải so sánh `is not None` thay vì truthiness - scope RỖNG
            # (user chưa được gán cửa hàng nào) phải trả kết quả chặn, KHÔNG được
            # rơi xuống truy vấn không filter (rò rỉ dữ liệu toàn hệ thống).
            if not _allowed_stores:
                return {"status": "forbidden",
                        "message": "forbidden: Bạn chưa được gán cửa hàng nào. Liên hệ quản trị viên."}
            marks = ",".join("?" * len(_allowed_stores))
            query += f" AND f.store_nbr IN ({marks})"
            params.extend(sorted(int(s) for s in _allowed_stores))

        query += " GROUP BY f.item_nbr, it.name, it.family ORDER BY total_sales DESC"

        df = pd.read_sql_query(query, conn, params=params)

        if df.empty:
            return {"status": "error", "message": "Không có dữ liệu dự báo cho khoảng thời gian này."}

        total_sales = df['total_sales'].sum()
        top_items = df.head(3).to_dict(orient='records')

        return {
            "store_nbr": store_nbr if store_nbr else "Toàn hệ thống",
            "forecast_period_days": days,
            "total_forecast_sales": round(float(total_sales), 2),
            "total_distinct_items": int(len(df)),
            "top_3_selling_items": top_items
        }
    except Exception as e:
        logger.error(f"Error in get_sales_summary: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        if conn:
            conn.close()


# ======================================================================
# 2. NHÓM QUẢN TRỊ TỒN KHO & CUNG ỨNG
# ======================================================================

def check_stockout_risk(store_nbr: int) -> Dict[str, Any]:
    """
    Kiểm tra danh sách các mặt hàng có nguy cơ hết hàng (Stockout) trong 16 ngày tới.
    """
    conn = None
    try:
        conn = get_db_connection()
        query = """
            SELECT 
                f.item_nbr,
                it.name,
                it.family,
                ROUND(SUM(f.predicted_sales), 2) as forecast_demand,
                i.current_stock,
                i.lead_time_days,
                ROUND(SUM(f.predicted_sales) - i.current_stock, 2) as deficit_quantity
            FROM forecasts f
            JOIN inventory i ON f.store_nbr = i.store_nbr AND f.item_nbr = i.item_nbr
            LEFT JOIN items it ON f.item_nbr = it.item_nbr
            WHERE f.store_nbr = ?
            GROUP BY f.item_nbr, it.name, it.family, i.current_stock, i.lead_time_days
            HAVING forecast_demand > i.current_stock
            ORDER BY deficit_quantity DESC
            LIMIT 10
        """
        df = pd.read_sql_query(query, conn, params=[store_nbr])

        if df.empty:
            return {"status": "success", "message": f"Tất cả mặt hàng tại Store {store_nbr} đều đủ mức tồn kho an toàn."}

        return {
            "store_nbr": store_nbr,
            "risk_items_count": len(df),
            "items_at_risk": df.to_dict(orient='records')
        }
    except Exception as e:
        logger.error(f"Error in check_stockout_risk: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        if conn:
            conn.close()


def calculate_reorder_point(store_nbr: int, item_nbr: int) -> Dict[str, Any]:
    """
    Tính Điểm đặt hàng lại (ROP) và Tồn kho an toàn (Safety Stock).
    """
    conn = None
    try:
        conn = get_db_connection()
        query = """
            SELECT 
                i.current_stock,
                i.lead_time_days,
                AVG(f.predicted_sales) as avg_daily_demand,
                f.item_nbr,
                it.name,
                it.family
            FROM inventory i
            JOIN forecasts f ON i.store_nbr = f.store_nbr AND i.item_nbr = f.item_nbr
            LEFT JOIN items it ON i.item_nbr = it.item_nbr
            WHERE i.store_nbr = ? AND i.item_nbr = ?
            GROUP BY i.current_stock, i.lead_time_days, f.item_nbr, it.name, it.family
        """
        df = pd.read_sql_query(query, conn, params=[store_nbr, item_nbr])

        if df.empty:
            return {"status": "error", "message": f"Không tìm thấy dữ liệu cho Store {store_nbr} - Item {item_nbr}."}

        row = df.iloc[0]
        lead_time = float(row['lead_time_days'] if row['lead_time_days'] else 3)
        avg_demand = float(row['avg_daily_demand'])

        # Giả định CV (Coefficient of Variation) = 0.2, Z = 1.65 (95% Service Level)
        std_demand = 0.2 * avg_demand
        safety_stock = 1.65 * std_demand * np.sqrt(lead_time)
        rop = (avg_demand * lead_time) + safety_stock
        current_stock = float(row['current_stock'])

        return {
            "store_nbr": store_nbr,
            "item_nbr": int(item_nbr),
            "name": row['name'],
            "family": row['family'],
            "current_stock": current_stock,
            "avg_daily_demand": round(avg_demand, 2),
            "lead_time_days": int(lead_time),
            "safety_stock": round(safety_stock, 2),
            "reorder_point": round(rop, 2),
            "need_reorder": bool(current_stock <= rop)
        }
    except Exception as e:
        logger.error(f"Error in calculate_reorder_point: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        if conn:
            conn.close()


def calculate_purchase_target(store_nbr: int, item_nbr: int) -> Dict[str, Any]:
    """
    Tính số lượng đề xuất đặt hàng mới (Suggested Purchase Quantity).
    """
    conn = None
    try:
        conn = get_db_connection()
        query = """
            SELECT 
                SUM(f.predicted_sales) as total_forecast, 
                i.current_stock,
                i.lead_time_days,
                it.name
            FROM forecasts f
            JOIN inventory i ON f.store_nbr = i.store_nbr AND f.item_nbr = i.item_nbr
            LEFT JOIN items it ON f.item_nbr = it.item_nbr
            WHERE f.store_nbr = ? AND f.item_nbr = ?
            GROUP BY i.current_stock, i.lead_time_days, it.name
        """
        df = pd.read_sql_query(query, conn, params=[store_nbr, item_nbr])

        if df.empty or df.iloc[0]['total_forecast'] is None:
            return {"status": "error", "message": f"Không có dữ liệu cho Store {store_nbr} - Item {item_nbr}."}

        row = df.iloc[0]
        forecast = float(row['total_forecast'])
        stock = float(row['current_stock'])
        order_qty = max(0.0, forecast - stock)

        return {
            "store_nbr": store_nbr,
            "item_nbr": int(item_nbr),
            "name": row['name'],
            "forecast_demand_16d": round(forecast, 2),
            "current_stock": stock,
            "needs_reorder": bool(order_qty > 0),
            "suggested_order_quantity": round(order_qty, 2)
        }
    except Exception as e:
        logger.error(f"Error in calculate_purchase_target: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        if conn:
            conn.close()


# ======================================================================
# 3. NHÓM MÔ PHỎNG KỊCH BẢN & ĐÁNH GIÁ THẤT THOÁT
# ======================================================================

def simulate_demand_multiplier(store_nbr: int, item_nbr: int, multiplier: float) -> Dict[str, Any]:
    """
    Mô phỏng tác động khi doanh số tăng/giảm (VD: 1.5 = tăng 50%, 0.8 = giảm 20%).
    """
    conn = None
    try:
        # Chặn NaN/Inf (JSON không hợp lệ khi serialize) và hệ số âm (demand âm vô nghĩa)
        try:
            multiplier = float(multiplier)
        except (TypeError, ValueError):
            return {"status": "error",
                    "message": f"Hệ số nhân không hợp lệ: {multiplier!r}. Vui lòng dùng số, ví dụ 1.5 = tăng 50%."}
        if not math.isfinite(multiplier) or multiplier < 0:
            return {"status": "error",
                    "message": f"Hệ số nhân phải là số dương hữu hạn (VD 1.5 = tăng 50%, 0.8 = giảm 20%), nhận được {multiplier!r}."}

        conn = get_db_connection()
        query = """
            SELECT SUM(f.predicted_sales) as old_forecast, i.current_stock, it.name
            FROM forecasts f
            JOIN inventory i ON f.store_nbr = i.store_nbr AND f.item_nbr = i.item_nbr
            LEFT JOIN items it ON f.item_nbr = it.item_nbr
            WHERE f.store_nbr = ? AND f.item_nbr = ?
            GROUP BY i.current_stock, it.name
        """
        df = pd.read_sql_query(query, conn, params=[store_nbr, item_nbr])

        if df.empty or df.iloc[0]['old_forecast'] is None:
            return {"status": "error", "message": "Không tìm thấy dữ liệu mặt hàng."}

        row = df.iloc[0]
        old_forecast = float(row['old_forecast'])
        new_forecast = old_forecast * multiplier
        current_stock = float(row['current_stock'])
        shortfall = max(0.0, new_forecast - current_stock)

        return {
            "store_nbr": store_nbr,
            "item_nbr": int(item_nbr),
            "name": row['name'],
            "multiplier": multiplier,
            "old_forecast_demand": round(old_forecast, 2),
            "new_forecast_demand": round(new_forecast, 2),
            "current_stock": current_stock,
            "will_stockout": bool(new_forecast > current_stock),
            "shortfall_quantity": round(shortfall, 2)
        }
    except Exception as e:
        logger.error(f"Error in simulate_demand_multiplier: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        if conn:
            conn.close()


def evaluate_stockout_loss(store_nbr: int, item_nbr: int, out_of_stock_days: int) -> Dict[str, Any]:
    """
    Ước tính thiệt hại doanh số khi bị đứt hàng trong N ngày.
    """
    conn = None
    try:
        # Chặn NaN/Inf và số ngày âm (thiệt hại âm vô nghĩa)
        try:
            out_of_stock_days = float(out_of_stock_days)
        except (TypeError, ValueError):
            return {"status": "error",
                    "message": f"Số ngày đứt hàng không hợp lệ: {out_of_stock_days!r}."}
        if not math.isfinite(out_of_stock_days) or out_of_stock_days < 0:
            return {"status": "error",
                    "message": f"Số ngày đứt hàng phải là số không âm hữu hạn, nhận được {out_of_stock_days!r}."}
        out_of_stock_days = int(out_of_stock_days)

        conn = get_db_connection()
        query = ("SELECT AVG(f.predicted_sales) as avg_daily_demand, it.name "
                 "FROM forecasts f LEFT JOIN items it ON f.item_nbr = it.item_nbr "
                 "WHERE f.store_nbr = ? AND f.item_nbr = ?")
        df = pd.read_sql_query(query, conn, params=[store_nbr, item_nbr])

        if df.empty or df.iloc[0]['avg_daily_demand'] is None:
            return {"status": "error", "message": "Không tìm thấy dữ liệu dự báo."}

        avg_demand = float(df.iloc[0]['avg_daily_demand'])
        lost_quantity = avg_demand * out_of_stock_days
        mock_price = 5.0  # Giá định mức giả lập USD
        lost_revenue = lost_quantity * mock_price

        return {
            "store_nbr": store_nbr,
            "item_nbr": int(item_nbr),
            "name": df.iloc[0]['name'],
            "out_of_stock_days": out_of_stock_days,
            "avg_daily_demand": round(avg_demand, 2),
            "lost_sales_quantity": round(lost_quantity, 2),
            "estimated_lost_revenue_usd": round(lost_revenue, 2),
            "impact_level": "Critical Negative Impact" if lost_revenue > 100 else "Moderate Impact"
        }
    except Exception as e:
        logger.error(f"Error in evaluate_stockout_loss: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        if conn:
            conn.close()


def run_scenario_analysis(
    store_nbr: int,
    family: str,
    demand_multiplier: float = 1.0,
    promo_days: Optional[int] = None,
    oil_price: Optional[float] = None,
    traffic_change_pct: Optional[float] = None,
    event_type: str = "none",
    event_days: int = 0,
    stock_override: Optional[float] = None,
    lead_time_override: Optional[float] = None,
    horizon_days: int = 16,
) -> Dict[str, Any]:
    """
    Chạy 1 kịch bản what-if trọn vẹn cho một ngành hàng: sửa số liệu -> dự báo
    lại bằng mô hình thật (ml_service, dự báo đệ quy 16 ngày) -> phân rã xuống
    SKU -> so với baseline -> trả KPI + kết luận + đề xuất.

    Kết quả đã chứa sẵn trường `analysis` (phân tích tiếng Việt) và
    `recommendation` (hành động đề xuất) - LLM chỉ cần trình bày lại,
    KHÔNG tự tính lại số liệu.
    """
    try:
        result = _run_scenario_impl(
            store_nbr=int(store_nbr),
            family=str(family),
            horizon_days=int(horizon_days) if horizon_days else 16,
            demand_multiplier=float(demand_multiplier if demand_multiplier is not None else 1.0),
            promo_days=promo_days,
            oil_price=oil_price,
            traffic_change_pct=traffic_change_pct,
            event_type=str(event_type or "none"),
            event_days=int(event_days or 0),
            stock_override=stock_override,
            lead_time_override=lead_time_override,
        )
        result["status"] = "success"
        return result
    except ScenarioError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        logger.error(f"Error in run_scenario_analysis: {e}")
        return {"status": "error", "message": f"Không chạy được kịch bản: {e}"}


# ======================================================================
# 4. NHÓM TỐI ƯU DANH MỤC & CỤM CỬA HÀNG
# ======================================================================

def find_cross_sell_items(store_nbr: int, item_nbr: int) -> Dict[str, Any]:
    """
    Tìm mặt hàng bán chạy nhất trong cùng ngành hàng để làm combo bán chéo.
    """
    conn = None
    try:
        conn = get_db_connection()
        item_info = pd.read_sql_query("SELECT name, family FROM items WHERE item_nbr = ?", conn, params=[item_nbr])
        if item_info.empty:
            return {"status": "error", "message": f"Không tìm thấy item_nbr {item_nbr}."}

        family = item_info.iloc[0]['family']

        query = """
            SELECT f.item_nbr, i.name, SUM(f.predicted_sales) as total_sales
            FROM forecasts f
            JOIN items i ON f.item_nbr = i.item_nbr
            WHERE f.store_nbr = ? AND i.family = ? AND f.item_nbr != ?
            GROUP BY f.item_nbr, i.name
            ORDER BY total_sales DESC
            LIMIT 3
        """
        df = pd.read_sql_query(query, conn, params=[store_nbr, family, item_nbr])

        return {
            "store_nbr": store_nbr,
            "target_item": int(item_nbr),
            "family": family,
            "suggested_cross_sell_items": df.to_dict(orient='records') if not df.empty else []
        }
    except Exception as e:
        logger.error(f"Error in find_cross_sell_items: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        if conn:
            conn.close()


def recommend_slow_mover_strategy(store_nbr: int) -> Dict[str, Any]:
    """
    Tìm các mặt hàng đọng vốn (tồn kho lớn hơn 2 lần nhu cầu 16 ngày) và đề xuất xả hàng.
    """
    conn = None
    try:
        conn = get_db_connection()
        query = """
            SELECT 
                f.item_nbr,
                it.name,
                it.family,
                ROUND(SUM(f.predicted_sales), 2) as forecast_16d, 
                i.current_stock
            FROM forecasts f
            JOIN inventory i ON f.store_nbr = i.store_nbr AND f.item_nbr = i.item_nbr
            LEFT JOIN items it ON f.item_nbr = it.item_nbr
            WHERE f.store_nbr = ?
            GROUP BY f.item_nbr, it.name, it.family, i.current_stock
            HAVING forecast_16d < (i.current_stock / 2.0) AND i.current_stock > 20
            ORDER BY i.current_stock DESC
            LIMIT 5
        """
        df = pd.read_sql_query(query, conn, params=[store_nbr])

        if df.empty:
            return {"status": "success", "message": f"Store {store_nbr} không có mặt hàng nào tồn đọng nghiêm trọng."}

        suggestions = []
        for _, row in df.iterrows():
            discount = 0.15 if row['forecast_16d'] > 5 else 0.30
            suggestions.append({
                "item_nbr": int(row['item_nbr']),
                "name": row['name'],
                "family": row['family'],
                "current_stock": float(row['current_stock']),
                "forecast_16d": float(row['forecast_16d']),
                "suggested_markdown_pct": int(discount * 100),
                "action": f"Giảm giá {int(discount*100)}% hoặc lập combo khuyến mãi để giải phóng tồn kho."
            })

        return {
            "store_nbr": store_nbr,
            "slow_movers_count": len(suggestions),
            "items_to_markdown": suggestions
        }
    except Exception as e:
        logger.error(f"Error in recommend_slow_mover_strategy: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        if conn:
            conn.close()


def compare_cluster_trends(cluster_1: int, cluster_2: int) -> Dict[str, Any]:
    """
    So sánh tổng doanh số dự báo giữa 2 cụm cửa hàng.
    """
    conn = None
    try:
        conn = get_db_connection()
        query = """
            SELECT s.cluster, SUM(f.predicted_sales) as total_sales
            FROM forecasts f
            JOIN stores s ON f.store_nbr = s.store_nbr
            WHERE s.cluster IN (?, ?)
            GROUP BY s.cluster
        """
        df = pd.read_sql_query(query, conn, params=[cluster_1, cluster_2])

        if len(df) < 2:
            return {"status": "error", "message": "Không đủ dữ liệu của cả 2 cluster để so sánh."}

        sales_1 = float(df[df['cluster'] == cluster_1]['total_sales'].values[0])
        sales_2 = float(df[df['cluster'] == cluster_2]['total_sales'].values[0])
        diff_pct = ((sales_1 - sales_2) / sales_2) * 100 if sales_2 != 0 else 0.0

        return {
            "cluster_1": int(cluster_1),
            "cluster_1_sales": round(sales_1, 2),
            "cluster_2": int(cluster_2),
            "cluster_2_sales": round(sales_2, 2),
            "difference_pct": round(diff_pct, 2),
            "stronger_cluster": cluster_1 if sales_1 > sales_2 else cluster_2
        }
    except Exception as e:
        logger.error(f"Error in compare_cluster_trends: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        if conn:
            conn.close()


# ======================================================================
# 5. NHÓM DOANH THU & KINH DOANH THỰC TẾ (dữ liệu lịch sử, không phải dự báo)
# ======================================================================

def get_monthly_revenue(store_nbr: Optional[int] = None, months: int = 1,
                        _allowed_stores: Optional[frozenset] = None) -> Dict[str, Any]:
    """
    Doanh thu THỰC TẾ theo tháng (USD tham chiếu) từ bảng agg_daily_business.
    Trả về các tháng MỚI NHẤT có dữ liệu trong DB - luôn kèm kỳ dữ liệu để
    LLM không gán nhầm cho tháng hiện tại (bộ dữ liệu là lịch sử).
    """
    conn = None
    try:
        months = max(1, min(int(months or 1), 12))
        conn = get_db_connection()
        if not _table_exists(conn, "agg_daily_business"):
            return {"status": "error",
                    "message": "Chưa có bảng agg_daily_business trong DB. Chạy một lần: "
                               "python backend/scripts/build_business_cache.py"}
        scope_sql, params, denied = _scope_filter_sql(store_nbr, _allowed_stores)
        if denied:
            return denied
        df = pd.read_sql_query(f"""
            SELECT strftime('%Y-%m', date) AS month,
                   ROUND(SUM(revenue), 2) AS revenue,
                   ROUND(SUM(returns), 2) AS returns,
                   ROUND(SUM(cogs), 2) AS cogs
            FROM agg_daily_business WHERE 1=1{scope_sql}
            GROUP BY month ORDER BY month DESC LIMIT ?
        """, conn, params=[*params, months])
        if df.empty:
            return {"status": "error", "message": "Không có dữ liệu doanh thu cho phạm vi này."}
        rng = conn.execute(
            f"SELECT MIN(date), MAX(date) FROM agg_daily_business WHERE 1=1{scope_sql}",
            params).fetchone()
        records = []
        for _, r in df.iterrows():  # df đang mới nhất trước -> đảo để trình bày cũ trước
            records.append({
                "month": r["month"],
                "revenue_usd": float(r["revenue"]),
                "returns_usd": float(r["returns"]),
                "cogs_usd": float(r["cogs"]),
                "gross_profit_usd": round(float(r["revenue"]) - float(r["returns"]) - float(r["cogs"]), 2),
            })
        return {
            "store_nbr": store_nbr if store_nbr else "Toàn hệ thống (trong phạm vi)",
            "months_requested": months,
            "monthly_revenue_newest_first": records,
            "data_period": {"from": rng[0], "to": rng[1]},
            "note": "Đây là DOANH THU THỰC TẾ lịch sử - trình bày đúng kỳ dữ liệu ở trên, "
                    "không được nói là tháng hiện tại.",
        }
    except Exception as e:
        logger.error(f"Error in get_monthly_revenue: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        if conn:
            conn.close()


def compare_stores_revenue(store_1: int, store_2: int, months: int = 3) -> Dict[str, Any]:
    """
    So sánh doanh thu THỰC TẾ giữa 2 cửa hàng theo các tháng gần nhất.
    RLS: store_1/store_2 được validate_tool_access kiểm tra trước khi gọi.
    """
    conn = None
    try:
        months = max(1, min(int(months or 3), 12))
        conn = get_db_connection()
        if not _table_exists(conn, "agg_daily_business"):
            return {"status": "error",
                    "message": "Chưa có bảng agg_daily_business trong DB. Chạy một lần: "
                               "python backend/scripts/build_business_cache.py"}
        df = pd.read_sql_query("""
            SELECT store_nbr, strftime('%Y-%m', date) AS month,
                   ROUND(SUM(revenue), 2) AS revenue
            FROM agg_daily_business
            WHERE store_nbr IN (?, ?)
            GROUP BY store_nbr, month ORDER BY month DESC
        """, conn, params=[int(store_1), int(store_2)])
        if df.empty:
            return {"status": "error", "message": f"Không có dữ liệu doanh thu cho cửa hàng {store_1} / {store_2}."}

        result = {"store_1": int(store_1), "store_2": int(store_2), "months": months}
        totals = {}
        for s in (store_1, store_2):
            sub = df[df["store_nbr"] == int(s)].head(months)
            result[f"store_{s}_monthly"] = [
                {"month": r["month"], "revenue_usd": float(r["revenue"])} for _, r in sub.iterrows()
            ]
            totals[s] = float(sub["revenue"].sum())
        result["store_1_total_usd"] = round(totals[store_1], 2)
        result["store_2_total_usd"] = round(totals[store_2], 2)
        if min(totals.values()) > 0:
            diff_pct = (totals[store_1] - totals[store_2]) / totals[store_2] * 100
            result["difference_pct"] = round(diff_pct, 2)
            result["stronger_store"] = int(store_1 if totals[store_1] > totals[store_2] else store_2)
        else:
            result["note"] = "Một trong hai cửa hàng không có dữ liệu doanh thu trong kỳ."
        return result
    except Exception as e:
        logger.error(f"Error in compare_stores_revenue: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        if conn:
            conn.close()


def get_top_selling_items(store_nbr: Optional[int] = None, top_n: int = 5,
                          _allowed_stores: Optional[frozenset] = None) -> Dict[str, Any]:
    """
    Top mặt hàng bán chạy THỰC TẾ theo tổng số lượng bán lịch sử (bảng tổng hợp
    agg_item_store_sales - KHÔNG phải dự báo). Nhanh nhờ pre-aggregation.
    """
    conn = None
    try:
        top_n = max(1, min(int(top_n or 5), 20))
        conn = get_db_connection()
        if not _table_exists(conn, "agg_item_store_sales"):
            return {"status": "error",
                    "message": "Chưa có bảng agg_item_store_sales trong DB. Chạy một lần: "
                               "python backend/scripts/build_sales_cache.py"}
        scope_sql, params, denied = _scope_filter_sql(store_nbr, _allowed_stores, column="a.store_nbr")
        if denied:
            return denied
        df = pd.read_sql_query(f"""
            SELECT a.item_nbr, it.name, it.family, it.perishable,
                   ROUND(SUM(a.unit_sales), 1) AS total_units_sold
            FROM agg_item_store_sales a
            LEFT JOIN items it ON a.item_nbr = it.item_nbr
            WHERE 1=1{scope_sql}
            GROUP BY a.item_nbr, it.name, it.family, it.perishable
            ORDER BY total_units_sold DESC LIMIT ?
        """, conn, params=[*params, top_n])
        if df.empty:
            return {"status": "error", "message": "Không có dữ liệu bán hàng cho phạm vi này."}
        return {
            "store_nbr": store_nbr if store_nbr else "Toàn hệ thống (trong phạm vi)",
            "top_items_actual_sales": df.to_dict(orient="records"),
            "note": "total_units_sold là SỐ LƯỢNG bán thực tế (unit), không phải doanh thu USD.",
        }
    except Exception as e:
        logger.error(f"Error in get_top_selling_items: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        if conn:
            conn.close()


def get_family_forecast(store_nbr: int, family: str, days: int = 7) -> Dict[str, Any]:
    """
    Chuỗi dự báo theo NGÀY của một ngành hàng (family) tại một cửa hàng,
    trong N ngày đầu của chu kỳ dự báo 16 ngày.
    """
    conn = None
    try:
        days = max(1, min(int(days or 7), 16))
        conn = get_db_connection()
        if not _table_exists(conn, "agg_forecast_date_family"):
            return {"status": "error",
                    "message": "Chưa có bảng agg_forecast_date_family trong DB. Chạy một lần: "
                               "python backend/scripts/build_sales_cache.py"}
        df = pd.read_sql_query(f"""
            SELECT date, ROUND(SUM(predicted_sales), 2) AS predicted_sales
            FROM agg_forecast_date_family
            WHERE store_nbr = ? AND UPPER(family) = UPPER(?)
              AND date IN (
                  SELECT DISTINCT date FROM agg_forecast_date_family
                  WHERE store_nbr = ? AND UPPER(family) = UPPER(?)
                  ORDER BY date LIMIT ?)
            GROUP BY date ORDER BY date
        """, conn, params=[int(store_nbr), str(family), int(store_nbr), str(family), days])
        if df.empty:
            avail = pd.read_sql_query(
                "SELECT DISTINCT family FROM agg_forecast_date_family WHERE store_nbr = ? ORDER BY family",
                conn, params=[int(store_nbr)])
            return {"status": "error",
                    "message": f"Không tìm thấy ngành hàng '{family}' tại cửa hàng {store_nbr}. "
                               f"Các ngành hàng hợp lệ: {avail['family'].tolist()}"}
        return {
            "store_nbr": int(store_nbr),
            "family": str(family).upper(),
            "forecast_days": days,
            "daily_forecast": df.to_dict(orient="records"),
            "total_forecast_sales": round(float(df["predicted_sales"].sum()), 2),
        }
    except Exception as e:
        logger.error(f"Error in get_family_forecast: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        if conn:
            conn.close()


def get_item_profile(item_nbr: int, _allowed_stores: Optional[frozenset] = None) -> Dict[str, Any]:
    """
    Hồ sơ mặt hàng: thông tin danh mục + tồn kho/dự báo theo từng cửa hàng
    trong phạm vi user + tổng số lượng bán thực tế.
    """
    conn = None
    try:
        conn = get_db_connection()
        item = conn.execute(
            "SELECT item_nbr, name, family, class, perishable FROM items WHERE item_nbr = ?",
            (int(item_nbr),)).fetchone()
        if item is None:
            return {"status": "error", "message": f"Không tìm thấy mặt hàng {item_nbr} trong danh mục."}
        result = {"item_nbr": int(item_nbr), "name": item["name"], "family": item["family"],
                  "class": item["class"],
                  "perishable": bool(item["perishable"])}

        scope_sql, params, denied = _scope_filter_sql(None, _allowed_stores)
        if denied:
            return denied

        inv = pd.read_sql_query(f"""
            SELECT store_nbr, current_stock, lead_time_days
            FROM inventory WHERE item_nbr = ?{scope_sql}
            ORDER BY store_nbr
        """, conn, params=[int(item_nbr), *params])
        fc = pd.read_sql_query(f"""
            SELECT store_nbr, ROUND(SUM(predicted_sales), 2) AS forecast_16d
            FROM forecasts WHERE item_nbr = ?{scope_sql}
            GROUP BY store_nbr
        """, conn, params=[int(item_nbr), *params])
        fc_map = {int(r["store_nbr"]): float(r["forecast_16d"]) for _, r in fc.iterrows()}
        stores_detail = []
        for _, r in inv.iterrows():
            s = int(r["store_nbr"])
            stores_detail.append({
                "store_nbr": s,
                "current_stock": float(r["current_stock"]),
                "lead_time_days": int(r["lead_time_days"]) if r["lead_time_days"] else None,
                "forecast_16d": fc_map.get(s, 0.0),
            })
        result["per_store"] = stores_detail
        result["total_current_stock"] = round(float(inv["current_stock"].sum()), 1) if not inv.empty else 0.0

        if _table_exists(conn, "agg_item_store_sales"):
            real = conn.execute(f"""
                SELECT ROUND(SUM(a.unit_sales), 1) FROM agg_item_store_sales a WHERE a.item_nbr = ?{scope_sql}
            """, (int(item_nbr), *params)).fetchone()
            result["total_units_sold_actual"] = float(real[0]) if real and real[0] is not None else 0.0
        return result
    except Exception as e:
        logger.error(f"Error in get_item_profile: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        if conn:
            conn.close()


def get_store_profile(store_nbr: Optional[int] = None,
                      _allowed_stores: Optional[frozenset] = None) -> Dict[str, Any]:
    """
    Không có store_nbr: danh sách cửa hàng trong phạm vi (kèm số mặt hàng quản lý).
    Có store_nbr: chi tiết 1 cửa hàng + doanh thu tháng gần nhất + top ngành hàng dự báo.
    """
    conn = None
    try:
        conn = get_db_connection()
        scope_sql, params, denied = _scope_filter_sql(store_nbr, _allowed_stores, column="s.store_nbr")
        if denied:
            return denied
        if store_nbr is None:
            df = pd.read_sql_query(f"""
                SELECT s.store_nbr, s.city, s.state, s.type, s.cluster,
                       COUNT(i.item_nbr) AS managed_items
                FROM stores s
                LEFT JOIN inventory i ON i.store_nbr = s.store_nbr
                WHERE 1=1{scope_sql}
                GROUP BY s.store_nbr, s.city, s.state, s.type, s.cluster
                ORDER BY s.store_nbr
            """, conn, params=params)
            return {"stores_in_scope": df.to_dict(orient="records"), "count": int(len(df))}

        row = conn.execute("""
            SELECT store_nbr, city, state, type, cluster FROM stores WHERE store_nbr = ?
        """, (int(store_nbr),)).fetchone()
        if row is None:
            return {"status": "error", "message": f"Không tồn tại cửa hàng {store_nbr}."}
        result = {"store_nbr": int(row["store_nbr"]), "city": row["city"], "state": row["state"],
                  "type": row["type"], "cluster": int(row["cluster"]) if row["cluster"] is not None else None}
        inv_count = conn.execute("SELECT COUNT(*) FROM inventory WHERE store_nbr = ?",
                                 (int(store_nbr),)).fetchone()[0]
        result["managed_items"] = int(inv_count)
        if _table_exists(conn, "agg_daily_business"):
            rev = conn.execute("""
                SELECT strftime('%Y-%m', date) AS month, ROUND(SUM(revenue), 2) AS revenue
                FROM agg_daily_business WHERE store_nbr = ?
                GROUP BY month ORDER BY month DESC LIMIT 1
            """, (int(store_nbr),)).fetchone()
            if rev:
                result["latest_month_revenue"] = {"month": rev["month"], "revenue_usd": float(rev["revenue"])}
        if _table_exists(conn, "agg_forecast_date_family"):
            top_fam = pd.read_sql_query("""
                SELECT family, ROUND(SUM(predicted_sales), 2) AS forecast_16d
                FROM agg_forecast_date_family WHERE store_nbr = ?
                GROUP BY family ORDER BY forecast_16d DESC LIMIT 3
            """, conn, params=[int(store_nbr)])
            result["top_3_families_by_forecast"] = top_fam.to_dict(orient="records")
        return result
    except Exception as e:
        logger.error(f"Error in get_store_profile: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        if conn:
            conn.close()


def get_store_traffic(store_nbr: Optional[int] = None, days: int = 30,
                      _allowed_stores: Optional[frozenset] = None) -> Dict[str, Any]:
    """
    Lượng khách (số hóa đơn/ngày) từ bảng daily_transactions: kỳ gần nhất vs
    kỳ trước đó để tính xu hướng tăng/giảm %.
    """
    conn = None
    try:
        days = max(1, min(int(days or 30), 180))
        conn = get_db_connection()
        if not _table_exists(conn, "daily_transactions"):
            return {"status": "error",
                    "message": "Chưa có bảng daily_transactions trong DB. Chạy một lần: "
                               "python backend/scripts/load_daily_transactions.py"}
        scope_sql, params, denied = _scope_filter_sql(store_nbr, _allowed_stores)
        if denied:
            return denied
        row = conn.execute("SELECT MAX(date), MIN(date) FROM daily_transactions").fetchone()
        max_date, min_date = row[0], row[1]
        window_days = 2 * days
        if pd.Timestamp(min_date) > pd.Timestamp(max_date) - pd.Timedelta(days=window_days - 1):
            days = max(1, days // 2)  # dữ liệu ngắn: thu hẹp cửa sổ so sánh
        df = pd.read_sql_query(f"""
            SELECT CASE WHEN date >= date(?, '-{days} day') THEN 'current' ELSE 'previous' END AS period,
                   SUM(n_invoices) AS total_invoices,
                   COUNT(DISTINCT date) AS active_days
            FROM daily_transactions
            WHERE date >= date(?, '-{2 * days - 1} day') AND date <= ?{scope_sql}
            GROUP BY period
        """, conn, params=[max_date, max_date, max_date, *params])
        stats = {r["period"]: {"total_invoices": int(r["total_invoices"] or 0),
                               "active_days": int(r["active_days"] or 0)}
                 for _, r in df.iterrows()}
        cur = stats.get("current", {"total_invoices": 0, "active_days": 0})
        prev = stats.get("previous", {"total_invoices": 0, "active_days": 0})
        result = {
            "store_nbr": store_nbr if store_nbr else "Toàn hệ thống (trong phạm vi)",
            "window_days": days,
            "current_period": cur,
            "previous_period": prev,
            "avg_invoices_per_day_current": round(cur["total_invoices"] / max(1, days), 1),
        }
        if prev["total_invoices"] > 0:
            result["trend_pct"] = round(
                (cur["total_invoices"] - prev["total_invoices"]) / prev["total_invoices"] * 100, 2)
            result["trend"] = "tăng" if result["trend_pct"] > 0 else ("giảm" if result["trend_pct"] < 0 else "đứng yên")
        return result
    except Exception as e:
        logger.error(f"Error in get_store_traffic: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        if conn:
            conn.close()


# ======================================================================
# 6. NHÓM KHUYẾN MÃI & RỦI RO HÀNG DỄ HỎNG
# ======================================================================

def evaluate_promotion_impact(store_nbr: int, family: Optional[str] = None) -> Dict[str, Any]:
    """
    So sánh doanh số ngày CÓ khuyến mãi (onpromotion=1) vs KHÔNG, tính lift %.
    Đọc bảng tổng hợp agg_promo_family_stats (dựng bởi build_promo_cache.py) -
    tuyệt đối KHÔNG quét historical_sales 59 triệu dòng trong lúc chat.
    """
    conn = None
    try:
        conn = get_db_connection()
        if not _table_exists(conn, "agg_promo_family_stats"):
            return {"status": "error",
                    "message": "Chưa có bảng agg_promo_family_stats trong DB. Chạy một lần: "
                               "python backend/scripts/build_promo_cache.py"}
        params = [int(store_nbr)]
        family_sql = ""
        if family:
            family_sql = " AND family = ?"
            params.append(str(family).upper())
        df = pd.read_sql_query(f"""
            SELECT onpromotion,
                   SUM(item_day_rows) AS item_day_rows,
                   SUM(total_units) AS total_units,
                   SUM(promo_days) AS promo_days
            FROM agg_promo_family_stats
            WHERE store_nbr = ?{family_sql}
            GROUP BY onpromotion
        """, conn, params=params)
        if df.empty:
            return {"status": "error",
                    "message": f"Không có dữ liệu bán hàng cho cửa hàng {store_nbr}" +
                               (f" / ngành {family}." if family else ".")}
        stats = {int(r["onpromotion"]): r for _, r in df.iterrows()}
        on, off = stats.get(1), stats.get(0)
        if on is None or off is None or float(off["total_units"]) <= 0 or int(off["item_day_rows"]) == 0:
            return {"status": "error",
                    "message": "Dữ liệu không đủ để so sánh (thiếu ngày có hoặc không khuyến mãi)."}
        avg_on = float(on["total_units"]) / max(1, int(on["item_day_rows"]))
        avg_off = float(off["total_units"]) / max(1, int(off["item_day_rows"]))
        lift_pct = (avg_on - avg_off) / avg_off * 100
        rng = conn.execute("SELECT MIN(date), MAX(date) FROM agg_daily_business").fetchone()
        return {
            "store_nbr": int(store_nbr),
            "family": str(family).upper() if family else "Tất cả ngành hàng",
            "period": {"from": rng[0], "to": rng[1]} if rng else None,
            "avg_units_per_item_day_on_promo": round(avg_on, 2),
            "avg_units_per_item_day_no_promo": round(avg_off, 2),
            "lift_pct": round(lift_pct, 2),
            "verdict": "Khuyến mãi HIỆU QUẢ (doanh số tăng rõ rệt)" if lift_pct >= 20 else
                       ("Khuyến mãi có tác động vừa phải" if lift_pct > 0 else "Khuyến mãi KHÔNG hiệu quả"),
            "days_with_sales_on_promo": int(on["promo_days"] or 0),
        }
    except Exception as e:
        logger.error(f"Error in evaluate_promotion_impact: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        if conn:
            conn.close()


def check_perishable_risk(store_nbr: int) -> Dict[str, Any]:
    """
    Mặt hàng DỄ HỎNG (items.perishable = 1) có nhu cầu dự báo vượt tồn kho,
    sắp xếp theo mức thiếu hụt, kèm số ngày tồn kho còn che phủ nhu cầu.
    """
    conn = None
    try:
        conn = get_db_connection()
        df = pd.read_sql_query("""
            SELECT f.item_nbr, it.name, it.family,
                   ROUND(SUM(f.predicted_sales), 2) AS forecast_16d,
                   i.current_stock, i.lead_time_days,
                   ROUND(SUM(f.predicted_sales) - i.current_stock, 2) AS deficit_quantity,
                   ROUND(i.current_stock / (SUM(f.predicted_sales) / 16.0), 1) AS days_of_cover
            FROM forecasts f
            JOIN inventory i ON f.store_nbr = i.store_nbr AND f.item_nbr = i.item_nbr
            JOIN items it ON f.item_nbr = it.item_nbr
            WHERE f.store_nbr = ? AND it.perishable = 1
            GROUP BY f.item_nbr, it.name, it.family, i.current_stock, i.lead_time_days
            HAVING forecast_16d > i.current_stock
            ORDER BY deficit_quantity DESC
            LIMIT 10
        """, conn, params=[int(store_nbr)])
        if df.empty:
            return {"status": "success",
                    "message": f"Các mặt hàng dễ hỏng tại Store {store_nbr} đều đủ tồn kho cho 16 ngày tới."}
        return {"store_nbr": int(store_nbr), "perishable_risk_count": int(len(df)),
                "items_at_risk": df.to_dict(orient="records")}
    except Exception as e:
        logger.error(f"Error in check_perishable_risk: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        if conn:
            conn.close()


# ======================================================================
# 7. MAPPING DICTIONARY & DISPATCHER
# ======================================================================

AVAILABLE_TOOLS = {
    "get_sales_summary": get_sales_summary,
    "check_stockout_risk": check_stockout_risk,
    "calculate_reorder_point": calculate_reorder_point,
    "calculate_purchase_target": calculate_purchase_target,
    "simulate_demand_multiplier": simulate_demand_multiplier,
    "evaluate_stockout_loss": evaluate_stockout_loss,
    "run_scenario_analysis": run_scenario_analysis,
    "find_cross_sell_items": find_cross_sell_items,
    "recommend_slow_mover_strategy": recommend_slow_mover_strategy,
    "compare_cluster_trends": compare_cluster_trends,
    "get_monthly_revenue": get_monthly_revenue,
    "compare_stores_revenue": compare_stores_revenue,
    "get_top_selling_items": get_top_selling_items,
    "get_family_forecast": get_family_forecast,
    "get_item_profile": get_item_profile,
    "get_store_profile": get_store_profile,
    "get_store_traffic": get_store_traffic,
    "evaluate_promotion_impact": evaluate_promotion_impact,
    "check_perishable_risk": check_perishable_risk,
}


# ======================================================================
# 8. JSON SCHEMAS CHO GROQ / QWEN 3.6 FUNCTION CALLING
# ======================================================================

GROQ_TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_sales_summary",
            "description": "Lấy tổng doanh số dự báo trong N ngày tới của một cửa hàng hoặc toàn bộ hệ thống.",
            "parameters": {
                "type": "object",
                "properties": {
                    "store_nbr": {"type": "integer", "description": "Mã cửa hàng (bỏ trống nếu muốn xem toàn chuỗi)."},
                    "days": {"type": "integer", "description": "Số ngày dự báo muốn xem (mặc định là 7 ngày)."}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_stockout_risk",
            "description": "Kiểm tra danh sách các mặt hàng có nguy cơ hết hàng, thiếu hụt tồn kho tại một cửa hàng.",
            "parameters": {
                "type": "object",
                "properties": {
                    "store_nbr": {"type": "integer", "description": "Mã số cửa hàng cần kiểm tra."}
                },
                "required": ["store_nbr"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_reorder_point",
            "description": "Tính điểm đặt hàng lại (ROP) và mức tồn kho an toàn cho một mặt hàng cụ thể tại một cửa hàng.",
            "parameters": {
                "type": "object",
                "properties": {
                    "store_nbr": {"type": "integer", "description": "Mã số cửa hàng."},
                    "item_nbr": {"type": "integer", "description": "Mã số mặt hàng."}
                },
                "required": ["store_nbr", "item_nbr"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_purchase_target",
            "description": "Tính số lượng cần đặt mua thêm để đáp ứng nhu cầu 16 ngày tới.",
            "parameters": {
                "type": "object",
                "properties": {
                    "store_nbr": {"type": "integer", "description": "Mã cửa hàng."},
                    "item_nbr": {"type": "integer", "description": "Mã mặt hàng."}
                },
                "required": ["store_nbr", "item_nbr"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "simulate_demand_multiplier",
            "description": "Mô phỏng thay đổi tồn kho khi doanh số tăng hoặc giảm theo hệ số (What-if scenario).",
            "parameters": {
                "type": "object",
                "properties": {
                    "store_nbr": {"type": "integer", "description": "Mã cửa hàng."},
                    "item_nbr": {"type": "integer", "description": "Mã mặt hàng."},
                    "multiplier": {"type": "number", "description": "Hệ số nhân (Ví dụ 1.5 là tăng 50%, 0.8 là giảm 20%)."}
                },
                "required": ["store_nbr", "item_nbr", "multiplier"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_stockout_loss",
            "description": "Ước tính số lượng bán mất và doanh thu thất thoát (USD) nếu một mặt hàng bị đứt hàng trong N ngày.",
            "parameters": {
                "type": "object",
                "properties": {
                    "store_nbr": {"type": "integer", "description": "Mã cửa hàng."},
                    "item_nbr": {"type": "integer", "description": "Mã mặt hàng."},
                    "out_of_stock_days": {"type": "integer", "description": "Số ngày hết kho cần mô phỏng thiệt hại."}
                },
                "required": ["store_nbr", "item_nbr", "out_of_stock_days"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_scenario_analysis",
            "description": (
                "Chạy kịch bản what-if cho một NGÀNH HÀNG (family) tại một cửa hàng: "
                "sửa số liệu (hệ số nhu cầu, khuyến mãi, giá dầu, lưu lượng khách, "
                "sự kiện bất ngờ, tồn kho) -> dự báo lại bằng mô hình thật 16 ngày -> "
                "so với hiện tại. Kết quả trả sẵn trường `analysis` (phân tích) và "
                "`recommendation` (đề xuất) - trình bày lại nguyên trạng, KHÔNG tự tính lại số."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "store_nbr": {"type": "integer", "description": "Mã cửa hàng."},
                    "family": {"type": "string", "description": "Tên ngành hàng, VD: GROCERY I, BEVERAGES."},
                    "demand_multiplier": {"type": "number", "description": "Hệ số nhu cầu (1.0 = giữ nguyên, 1.5 = tăng 50%, 0.8 = giảm 20%)."},
                    "promo_days": {"type": "integer", "description": "Số ngày khuyến mãi trong kỳ 0-16 (bỏ trống = theo lịch thật)."},
                    "oil_price": {"type": "number", "description": "Giá dầu USD (bỏ trống = giá thật)."},
                    "traffic_change_pct": {"type": "number", "description": "% thay đổi lưu lượng khách, VD -20 hoặc 30 (bỏ trống = giữ nguyên)."},
                    "event_type": {"type": "string", "enum": ["none", "holiday", "earthquake"], "description": "Sự kiện bất ngờ: none/holiday (ngày lễ)/earthquake (thiên tai)."},
                    "event_days": {"type": "integer", "description": "Số ngày diễn ra sự kiện (0 = không có)."},
                    "stock_override": {"type": "number", "description": "Tổng tồn kho muốn giả lập (bỏ trống = tồn thật của ngành)."},
                    "lead_time_override": {"type": "number", "description": "Lead time giả lập tính bằng ngày (bỏ trống = giá trị thật)."},
                    "horizon_days": {"type": "integer", "description": "Số ngày dự báo 7-16 (mặc định 16)."}
                },
                "required": ["store_nbr", "family"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_slow_mover_strategy",
            "description": "Phát hiện các mặt hàng bán chậm, đọng vốn cao và đề xuất mức giảm giá để xả hàng tồn.",
            "parameters": {
                "type": "object",
                "properties": {
                    "store_nbr": {"type": "integer", "description": "Mã cửa hàng."}
                },
                "required": ["store_nbr"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_cross_sell_items",
            "description": "Tìm các sản phẩm bán chạy cùng ngành hàng để lập chương trình bán kèm (Cross-selling).",
            "parameters": {
                "type": "object",
                "properties": {
                    "store_nbr": {"type": "integer", "description": "Mã cửa hàng."},
                    "item_nbr": {"type": "integer", "description": "Mã mặt hàng chính."}
                },
                "required": ["store_nbr", "item_nbr"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_cluster_trends",
            "description": "So sánh xu hướng và tổng doanh số giữa 2 cụm cửa hàng (Cluster).",
            "parameters": {
                "type": "object",
                "properties": {
                    "cluster_1": {"type": "integer", "description": "Mã cụm 1."},
                    "cluster_2": {"type": "integer", "description": "Mã cụm 2."}
                },
                "required": ["cluster_1", "cluster_2"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_monthly_revenue",
            "description": "Lấy DOANH THU THỰC TẾ (USD) theo tháng của một cửa hàng hoặc toàn hệ thống. "
                           "Dùng khi hỏi về doanh thu các kỳ ĐÃ QUA (tháng này, tháng trước, 3 tháng gần nhất...). "
                           "Kết quả là dữ liệu lịch sử - luôn đọc 'data_period' để nêu đúng kỳ.",
            "parameters": {
                "type": "object",
                "properties": {
                    "store_nbr": {"type": "integer", "description": "Mã cửa hàng (bỏ trống = toàn hệ thống trong phạm vi)."},
                    "months": {"type": "integer", "description": "Số tháng gần nhất muốn xem (mặc định 1, tối đa 12)."}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_stores_revenue",
            "description": "So sánh doanh thu thực tế giữa 2 cửa hàng theo các tháng gần nhất và kết luận cửa hàng nào mạnh hơn.",
            "parameters": {
                "type": "object",
                "properties": {
                    "store_1": {"type": "integer", "description": "Mã cửa hàng thứ nhất."},
                    "store_2": {"type": "integer", "description": "Mã cửa hàng thứ hai."},
                    "months": {"type": "integer", "description": "Số tháng gần nhất dùng để so sánh (mặc định 3)."}
                },
                "required": ["store_1", "store_2"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_selling_items",
            "description": "Top các mặt hàng bán chạy THỰC TẾ theo tổng số lượng bán lịch sử (không phải dự báo). "
                           "Dùng cho câu hỏi 'mặt hàng nào bán chạy nhất', 'sản phẩm chủ lực'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "store_nbr": {"type": "integer", "description": "Mã cửa hàng (bỏ trống = toàn hệ thống trong phạm vi)."},
                    "top_n": {"type": "integer", "description": "Số lượng mặt hàng cần xem (mặc định 5, tối đa 20)."}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_family_forecast",
            "description": "Lấy dự báo doanh số theo NGÀY của một ngành hàng (family, VD: PRODUCE, GROCERY I) "
                           "tại một cửa hàng trong N ngày đầu của chu kỳ dự báo 16 ngày.",
            "parameters": {
                "type": "object",
                "properties": {
                    "store_nbr": {"type": "integer", "description": "Mã cửa hàng."},
                    "family": {"type": "string", "description": "Tên ngành hàng (VD: PRODUCE, MEATS, BREAD/BAKERY)."},
                    "days": {"type": "integer", "description": "Số ngày dự báo muốn xem (mặc định 7, tối đa 16)."}
                },
                "required": ["store_nbr", "family"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_item_profile",
            "description": "Xem hồ sơ chi tiết một mặt hàng: ngành hàng, dễ hỏng hay không, tồn kho và dự báo "
                           "theo từng cửa hàng trong phạm vi, tổng số lượng bán thực tế.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_nbr": {"type": "integer", "description": "Mã số mặt hàng."}
                },
                "required": ["item_nbr"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_store_profile",
            "description": "Thông tin cửa hàng: thành phố, loại, cụm, số mặt hàng quản lý, doanh thu tháng gần nhất "
                           "và top ngành hàng dự báo. Nếu bỏ trống store_nbr sẽ trả về DANH SÁCH tất cả cửa hàng trong phạm vi.",
            "parameters": {
                "type": "object",
                "properties": {
                    "store_nbr": {"type": "integer", "description": "Mã cửa hàng (bỏ trống = liệt kê danh sách cửa hàng)."}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_store_traffic",
            "description": "Lượng khách ghé cửa hàng (số hóa đơn/ngày): trung bình mỗi ngày và xu hướng "
                           "tăng/giảm % so với kỳ trước đó. Dùng cho câu hỏi về khách hàng, lượt ghé thăm.",
            "parameters": {
                "type": "object",
                "properties": {
                    "store_nbr": {"type": "integer", "description": "Mã cửa hàng (bỏ trống = toàn hệ thống trong phạm vi)."},
                    "days": {"type": "integer", "description": "Độ dài kỳ so sánh theo ngày (mặc định 30, tối đa 180)."}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_promotion_impact",
            "description": "Đánh giá HIỆU QUẢ KHUYẾN MÃI: so sánh doanh số trung bình ngày có khuyến mãi vs không "
                           "trong 12 tháng lịch sử gần nhất, tính mức tăng (lift) %.",
            "parameters": {
                "type": "object",
                "properties": {
                    "store_nbr": {"type": "integer", "description": "Mã cửa hàng."},
                    "family": {"type": "string", "description": "Ngành hàng cụ thể (tùy chọn, VD: PRODUCE)."}
                },
                "required": ["store_nbr"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_perishable_risk",
            "description": "Danh sách mặt hàng DỄ HỎNG (rau quả, thịt, sữa...) có nhu cầu dự báo vượt tồn kho "
                           "trong 16 ngày tới, kèm số ngày tồn kho còn che phủ.",
            "parameters": {
                "type": "object",
                "properties": {
                    "store_nbr": {"type": "integer", "description": "Mã cửa hàng."}
                },
                "required": ["store_nbr"]
            }
        }
    }
]