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
        base_where = """
            WHERE f.date >= (SELECT MIN(date) FROM forecasts)
              AND f.date < date((SELECT MIN(date) FROM forecasts), '+' || ? || ' day')
        """
        params = [days]
        scope_sql = ""
        if store_nbr is not None:
            scope_sql = " AND f.store_nbr = ?"
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
            scope_sql = f" AND f.store_nbr IN ({marks})"
            params.extend(sorted(int(s) for s in _allowed_stores))

        query = f"""
            SELECT f.item_nbr, it.name, it.family, SUM(f.predicted_sales) as total_sales
            FROM forecasts f
            LEFT JOIN items it ON f.item_nbr = it.item_nbr
            {base_where}{scope_sql}
            GROUP BY f.item_nbr, it.name, it.family ORDER BY total_sales DESC
        """

        df = pd.read_sql_query(query, conn, params=params)

        if df.empty:
            return {"status": "error", "message": "Không có dữ liệu dự báo cho khoảng thời gian này."}

        total_sales = df['total_sales'].sum()
        top_items = df.head(3).to_dict(orient='records')

        # Mốc ngày thật của cửa sổ dự báo - model cần để nêu đúng kỳ,
        # không gán nhầm kết quả cho "tháng này" ngoài đời thực.
        win = conn.execute(
            f"SELECT MIN(f.date), MAX(f.date) FROM forecasts f{base_where}{scope_sql}",
            params).fetchone()

        return {
            "store_nbr": store_nbr if store_nbr else "Toàn hệ thống",
            "forecast_period_days": days,
            "forecast_window": {"from": win[0], "to": win[1]},
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
# 7. NHÓM PHÂN TÍCH TÀI CHÍNH FP&A (methodology: run-fpa skills)
# ======================================================================

def analyze_gross_margin(store_nbr: Optional[int] = None, months: int = 3,
                         _allowed_stores: Optional[frozenset] = None) -> Dict[str, Any]:
    """
    Biên lợi nhuận gộp (direct margin - cost-profitability): doanh thu − trả hàng − COGS.
    Có store_nbr: chuỗi biên gộp theo tháng + biến động pp. Không: xếp hạng biên gộp
    các cửa hàng trong phạm vi user.
    """
    conn = None
    try:
        months = max(1, min(int(months or 3), 12))
        conn = get_db_connection()
        if not _table_exists(conn, "agg_daily_business"):
            return {"status": "error",
                    "message": "Chưa có bảng agg_daily_business trong DB. Chạy một lần: "
                               "python backend/scripts/build_business_cache.py"}
        scope_sql, params, denied = _scope_filter_sql(store_nbr, _allowed_stores)
        if denied:
            return denied
        if store_nbr is not None:
            df = pd.read_sql_query(f"""
                SELECT strftime('%Y-%m', date) AS month,
                       ROUND(SUM(revenue), 2) AS revenue,
                       ROUND(SUM(returns), 2) AS returns,
                       ROUND(SUM(cogs), 2) AS cogs
                FROM agg_daily_business WHERE 1=1{scope_sql}
                GROUP BY month ORDER BY month DESC LIMIT ?
            """, conn, params=[*params, months])
            if df.empty:
                return {"status": "error", "message": f"Không có dữ liệu cho cửa hàng {store_nbr}."}
            rows = []
            for _, r in df.iterrows():  # mới nhất trước -> đảo để cũ trước
                rev, ret, cogs = float(r["revenue"]), float(r["returns"]), float(r["cogs"])
                gp = rev - ret - cogs
                rows.append({"month": r["month"], "revenue_usd": rev, "returns_usd": ret,
                             "cogs_usd": cogs, "gross_profit_usd": round(gp, 2),
                             "gross_margin_pct": round(gp / rev * 100, 2) if rev > 0 else None})
            result = {"store_nbr": int(store_nbr), "monthly_margin_oldest_first": rows,
                      "formula": "gross_profit = revenue - returns - cogs; margin = gp / revenue"}
            if len(rows) >= 2:
                result["margin_change_pp_vs_prev_month"] = round(
                    rows[-1]["gross_margin_pct"] - rows[-2]["gross_margin_pct"], 2)
            return result

        # Không chỉ định cửa hàng: xếp hạng biên gộp theo cửa hàng trong cửa sổ N tháng
        window_sql = (" AND date >= date((SELECT MAX(date) FROM agg_daily_business), "
                      f"'-{months} month')")
        df = pd.read_sql_query(f"""
            SELECT store_nbr,
                   ROUND(SUM(revenue), 2) AS revenue,
                   ROUND(SUM(returns), 2) AS returns,
                   ROUND(SUM(cogs), 2) AS cogs
            FROM agg_daily_business WHERE 1=1{scope_sql}{window_sql}
            GROUP BY store_nbr
        """, conn, params=params)
        if df.empty:
            return {"status": "error", "message": "Không có dữ liệu cho phạm vi này."}
        df["gross_profit_usd"] = (df["revenue"] - df["returns"] - df["cogs"]).round(2)
        df["gross_margin_pct"] = (df["gross_profit_usd"] / df["revenue"] * 100).round(2)
        df = df.sort_values("gross_margin_pct", ascending=False)
        return {
            "window_months": months,
            "store_margin_ranking_highest_first": df.head(10).to_dict(orient="records"),
            "formula": "gross_profit = revenue - returns - cogs; margin = gp / revenue",
        }
    except Exception as e:
        logger.error(f"Error in analyze_gross_margin: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        if conn:
            conn.close()


def analyze_revenue_change(store_nbr: int, months: int = 2) -> Dict[str, Any]:
    """
    Phân tách biến động doanh thu (budget-variance, price/volume decomposition):
    Volume = (Q_a − Q_b) × P_b với Q = số hóa đơn (daily_transactions);
    Rate   = (P_a − P_b) × Q_a với P = giá trị trung bình mỗi hóa đơn.
    Bridge 2 nhân tố đóng về 0 đúng theo construction - residual chỉ là sai số làm tròn.
    """
    conn = None
    try:
        months = max(2, min(int(months or 2), 6))
        conn = get_db_connection()
        if not _table_exists(conn, "agg_daily_business"):
            return {"status": "error",
                    "message": "Chưa có bảng agg_daily_business trong DB. Chạy một lần: "
                               "python backend/scripts/build_business_cache.py"}
        df = pd.read_sql_query("""
            SELECT strftime('%Y-%m', b.date) AS month,
                   ROUND(SUM(b.revenue), 2) AS revenue,
                   SUM(COALESCE(t.n_invoices, 0)) AS invoices
            FROM agg_daily_business b
            LEFT JOIN daily_transactions t ON t.date = b.date AND t.store_nbr = b.store_nbr
            WHERE b.store_nbr = ?
            GROUP BY month ORDER BY month DESC LIMIT ?
        """, conn, params=[int(store_nbr), months])
        if len(df) < 2:
            return {"status": "error",
                    "message": f"Cần ít nhất 2 tháng dữ liệu để phân tách (cửa hàng {store_nbr} chỉ có {len(df)})."}

        bridges = []
        rows = list(df.itertuples(index=False))  # mới nhất trước
        for newer, older in zip(rows, rows[1:]):
            rev_a, rev_b = float(newer.revenue), float(older.revenue)
            q_a, q_b = float(newer.invoices), float(older.invoices)
            p_a = rev_a / q_a if q_a > 0 else 0.0
            p_b = rev_b / q_b if q_b > 0 else 0.0
            delta = rev_a - rev_b
            volume_effect = (q_a - q_b) * p_b
            ticket_effect = (p_a - p_b) * q_a
            residual = delta - volume_effect - ticket_effect
            bridges.append({
                "month_newer": newer.month, "month_older": older.month,
                "revenue_newer_usd": rev_a, "revenue_older_usd": rev_b,
                "revenue_change_usd": round(delta, 2),
                "volume_effect_usd": round(volume_effect, 2),
                "invoice_count_newer": int(q_a), "invoice_count_older": int(q_b),
                "avg_ticket_newer_usd": round(p_a, 2), "avg_ticket_older_usd": round(p_b, 2),
                "ticket_effect_usd": round(ticket_effect, 2),
                "residual_usd": round(residual, 2),
                "verdict": ("Doanh thu TĂNG - chủ yếu do lượt khách" if delta > 0 and abs(volume_effect) >= abs(ticket_effect)
                            else "Doanh thu TĂNG - chủ yếu do giá trị mỗi hóa đơn" if delta > 0
                            else "Doanh thu GIẢM - chủ yếu do lượt khách" if abs(volume_effect) >= abs(ticket_effect)
                            else "Doanh thu GIẢM - chủ yếu do giá trị mỗi hóa đơn"),
            })
        return {
            "store_nbr": int(store_nbr),
            "formula": "volume_effect = (Q_a - Q_b) x P_b; ticket_effect = (P_a - P_b) x Q_a; "
                       "Q = số hóa đơn, P = doanh thu / hóa đơn",
            "bridges_oldest_pair_first": list(reversed(bridges)),
            "note": "Bridge 2 nhân tố đóng đúng về tổng biến động (residual chỉ là làm tròn).",
        }
    except Exception as e:
        logger.error(f"Error in analyze_revenue_change: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        if conn:
            conn.close()


def benchmark_store_vs_peers(store_nbr: int, months: int = 3,
                             _allowed_stores: Optional[frozenset] = None) -> Dict[str, Any]:
    """
    Benchmark nội bộ (peer-benchmark): so cửa hàng với trung bình các cửa hàng
    cùng TYPE trong phạm vi user (không tính chính nó) về doanh thu, biên gộp,
    giá trị mỗi hóa đơn. Peers lọc theo _allowed_stores - không rò rỉ ngoài phạm vi.
    """
    conn = None
    try:
        months = max(1, min(int(months or 3), 12))
        conn = get_db_connection()
        if not _table_exists(conn, "agg_daily_business"):
            return {"status": "error",
                    "message": "Chưa có bảng agg_daily_business trong DB. Chạy một lần: "
                               "python backend/scripts/build_business_cache.py"}
        me = conn.execute("SELECT store_nbr, city, type, cluster FROM stores WHERE store_nbr = ?",
                          (int(store_nbr),)).fetchone()
        if me is None:
            return {"status": "error", "message": f"Không tồn tại cửa hàng {store_nbr}."}

        peer_scope_sql, peer_params, denied = _scope_filter_sql(None, _allowed_stores)
        if denied:
            return denied
        peer_scope_sql = peer_scope_sql.replace(" AND store_nbr", " AND s.store_nbr", 1)
        peers = pd.read_sql_query(f"""
            SELECT s.store_nbr FROM stores s WHERE s.type = ? AND s.store_nbr != ?{peer_scope_sql}
        """, conn, params=[me["type"], int(store_nbr), *peer_params])
        if peers.empty:
            return {"status": "error",
                    "message": f"Không có cửa hàng cùng loại {me['type']} nào trong phạm vi để so sánh."}
        peer_ids = [int(x) for x in peers["store_nbr"].tolist()]
        peer_marks = ",".join("?" * len(peer_ids))

        window_sql = (" AND b.date >= date((SELECT MAX(date) FROM agg_daily_business), "
                      f"'-{months} month')")
        def _metrics(store_filter_sql: str, params_: list) -> Dict[str, float]:
            row = conn.execute(f"""
                SELECT COALESCE(SUM(b.revenue), 0) AS revenue,
                       COALESCE(SUM(b.returns), 0) AS returns,
                       COALESCE(SUM(b.cogs), 0) AS cogs,
                       COALESCE(SUM(t.n_invoices), 0) AS invoices
                FROM agg_daily_business b
                LEFT JOIN daily_transactions t ON t.date = b.date AND t.store_nbr = b.store_nbr
                WHERE 1=1{store_filter_sql}{window_sql}
            """, params_).fetchone()
            rev, ret, cogs = float(row[0]), float(row[1]), float(row[2])
            inv = float(row[3])
            gp = rev - ret - cogs
            return {"revenue_usd": round(rev, 2), "gross_profit_usd": round(gp, 2),
                    "gross_margin_pct": round(gp / rev * 100, 2) if rev > 0 else 0.0,
                    "invoices": int(inv),
                    "avg_ticket_usd": round(rev / inv, 2) if inv > 0 else 0.0}

        mine = _metrics(" AND b.store_nbr = ?", [int(store_nbr)])
        peer = _metrics(f" AND b.store_nbr IN ({peer_marks})", peer_ids)

        def _vs(metric: str) -> Dict[str, Any]:
            m = mine[metric]
            p = peer[metric]
            diff_pct = round((m - p) / p * 100, 2) if p else None
            return {"metric": metric, "store": m, "peer_avg": p, "vs_peer_avg_pct": diff_pct}

        return {
            "store_nbr": int(store_nbr), "city": me["city"], "type": me["type"],
            "cluster": me["cluster"], "peer_type": me["type"], "peer_count": len(peer_ids),
            "window_months": months,
            "store_totals": mine,
            "peer_avg_totals": peer,
            "comparison": [_vs("revenue_usd"), _vs("gross_margin_pct"), _vs("avg_ticket_usd"),
                           _vs("gross_profit_usd")],
            "note": "peer_avg chỉ tính các cửa hàng cùng loại trong phạm vi truy cập của bạn "
                    "(không gồm chính cửa hàng này), chia đều theo số cửa hàng.",
        }
    except Exception as e:
        logger.error(f"Error in benchmark_store_vs_peers: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        if conn:
            conn.close()


def analyze_reorder_profitability(store_nbr: int, top_n: int = 5) -> Dict[str, Any]:
    """
    Cơ hội lợi nhuận gộp tăng thêm (finance-bp decision support): với các mặt hàng
    dự báo 16 ngày vượt tồn kho, nếu đặt bổ sung đủ nhu cầu thì thu thêm bao nhiêu
    theo giá tham chiếu family_prices: gap x unit_price x (1 - cost_ratio).
    """
    conn = None
    try:
        top_n = max(1, min(int(top_n or 5), 10))
        conn = get_db_connection()
        if not _table_exists(conn, "family_prices"):
            return {"status": "error",
                    "message": "Chưa có bảng family_prices trong DB. Chạy một lần: "
                               "python backend/scripts/build_business_cache.py"}
        # Tổng hợp forecasts TRƯỚC trong subquery (đi qua index store_nbr) rồi mới
        # join - nếu join thẳng planner có thể quét toàn bộ 3.4M dòng forecasts (~40s).
        df = pd.read_sql_query("""
            SELECT agg.item_nbr, it.family, agg.forecast_16d,
                   i.current_stock,
                   fp.unit_price, fp.cost_ratio
            FROM (
                SELECT item_nbr, ROUND(SUM(predicted_sales), 2) AS forecast_16d
                FROM forecasts WHERE store_nbr = ?
                GROUP BY item_nbr
            ) agg
            JOIN inventory i ON i.store_nbr = ? AND i.item_nbr = agg.item_nbr
            JOIN items it ON agg.item_nbr = it.item_nbr
            JOIN family_prices fp ON fp.family = it.family
            WHERE agg.forecast_16d > i.current_stock
        """, conn, params=[int(store_nbr), int(store_nbr)])
        if df.empty:
            return {"status": "success",
                    "message": f"Không có mặt hàng nào tại Store {store_nbr} cần đặt bổ sung "
                               "(dự báo 16 ngày đều trong phạm vi tồn kho)."}
        df["gap_units"] = (df["forecast_16d"] - df["current_stock"]).round(2)
        df["gross_margin_per_unit_usd"] = (df["unit_price"] * (1 - df["cost_ratio"])).round(4)
        df["est_incremental_profit_usd"] = (df["gap_units"] * df["gross_margin_per_unit_usd"]).round(2)
        df = df.sort_values("est_incremental_profit_usd", ascending=False)
        total = float(df["est_incremental_profit_usd"].sum())
        return {
            "store_nbr": int(store_nbr),
            "items_needing_reorder": int(len(df)),
            "top_opportunities": df.head(top_n).to_dict(orient="records"),
            "total_est_incremental_profit_usd": round(total, 2),
            "formula": "est_profit = (forecast_16d - current_stock) x unit_price x (1 - cost_ratio)",
            "note": "Ước tính theo GIÁ THAM CHIẾU từng ngành hàng (dataset không có giá thật) - "
                    "dùng để xếp hạng ưu tiên đặt hàng, không phải con số cam kết.",
        }
    except Exception as e:
        logger.error(f"Error in analyze_reorder_profitability: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        if conn:
            conn.close()


# ======================================================================
# 7c. NHÓM QUẢN TRỊ BÁN LẺ: TỒN KHO - MERCHANDISING (dữ liệu cache, không quét lớn)
# ======================================================================

def _cost_price_expr(alias: str = "fp") -> str:
    """Biểu thức giá vốn tham chiếu mỗi unit: unit_price x cost_ratio."""
    return f"({alias}.unit_price * {alias}.cost_ratio)"


def analyze_inventory_health(store_nbr: int, months: int = 3) -> Dict[str, Any]:
    """
    Sức khỏe tồn kho: giá trị tồn (giá vốn tham chiếu), DOH (số ngày tồn che phủ
    doanh thu), vòng quay năm ước tính, và danh sách overstock (DOH > 30 ngày).
    """
    conn = None
    try:
        months = max(1, min(int(months or 3), 12))
        conn = get_db_connection()
        if not _table_exists(conn, "family_prices"):
            return {"status": "error",
                    "message": "Chưa có bảng family_prices trong DB. Chạy một lần: "
                               "python backend/scripts/build_business_cache.py"}

        # Tổng giá trị tồn kho theo giá vốn tham chiếu
        row = conn.execute(f"""
            SELECT COALESCE(SUM(i.current_stock * {_cost_price_expr()}), 0)
            FROM inventory i
            JOIN items it ON i.item_nbr = it.item_nbr
            JOIN family_prices fp ON fp.family = it.family
            WHERE i.store_nbr = ?
        """, (int(store_nbr),)).fetchone()
        stock_value = float(row[0] or 0)

        # COGS bình quân ngày trong cửa sổ `months` tháng
        cogs_row = conn.execute(f"""
            SELECT COALESCE(SUM(cogs), 0), COUNT(DISTINCT date)
            FROM agg_daily_business
            WHERE store_nbr = ? AND date >= date((SELECT MAX(date) FROM agg_daily_business), '-{months} month')
        """, (int(store_nbr),)).fetchone()
        total_cogs, active_days = float(cogs_row[0] or 0), int(cogs_row[1] or 0)
        if active_days == 0 or total_cogs <= 0:
            return {"status": "error", "message": f"Không có dữ liệu COGS cho cửa hàng {store_nbr}."}
        daily_cogs = total_cogs / active_days
        doh = stock_value / daily_cogs if daily_cogs > 0 else None
        turnover = (total_cogs / months * 12 / stock_value) if stock_value > 0 else None

        # Overstock: mặt hàng có DOH riêng > 30 ngày (nhu cầu ngày = forecast_16d / 16)
        over = pd.read_sql_query(f"""
            SELECT agg.item_nbr, it.family, agg.forecast_16d, i.current_stock,
                   ROUND(i.current_stock * {_cost_price_expr()}, 2) AS stock_value_usd,
                   ROUND(i.current_stock / (agg.forecast_16d / 16.0), 1) AS doh_days
            FROM (
                SELECT item_nbr, ROUND(SUM(predicted_sales), 2) AS forecast_16d
                FROM forecasts WHERE store_nbr = ?
                GROUP BY item_nbr
            ) agg
            JOIN inventory i ON i.store_nbr = ? AND i.item_nbr = agg.item_nbr
            JOIN items it ON agg.item_nbr = it.item_nbr
            JOIN family_prices fp ON fp.family = it.family
            WHERE i.current_stock > 0 AND agg.forecast_16d > 0
              AND i.current_stock / (agg.forecast_16d / 16.0) > 30
            ORDER BY stock_value_usd DESC
            LIMIT 3
        """, conn, params=[int(store_nbr), int(store_nbr)])

        return {
            "store_nbr": int(store_nbr),
            "window_months": months,
            "stock_value_usd": round(stock_value, 2),
            "daily_cogs_usd": round(daily_cogs, 2),
            "days_on_hand": round(doh, 1) if doh else None,
            "turnover_per_year_est": round(turnover, 2) if turnover else None,
            "overstock_count_doh_gt_30": int(len(over)),
            "top_overstock_items": over.to_dict(orient="records"),
            "formula": "stock_value = stock x unit_price x cost_ratio (giá vốn tham chiếu); "
                       "DOH = stock_value / daily_cogs; turnover = (cogs/tháng x 12) / stock_value",
            "note": "Giá vốn là GIÁ THAM CHIẾU theo ngành hàng (dataset không có giá thật).",
        }
    except Exception as e:
        logger.error(f"Error in analyze_inventory_health: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        if conn:
            conn.close()


def find_dead_stock(store_nbr: int, top_n: int = 10) -> Dict[str, Any]:
    """
    Hàng chết theo 2 định nghĩa:
    (a) chưa từng bán ở cửa hàng này (unit_sales trống/0 toàn kỳ lịch sử);
    (b) chết gần đây toàn chuỗi (sku_stats.avg_daily_45d ≈ 0 - 45 ngày cuối không bán).
    Giá trị vốn đọng = stock x giá vốn tham chiếu.
    """
    conn = None
    try:
        top_n = max(1, min(int(top_n or 10), 15))
        conn = get_db_connection()
        for tbl in ("agg_item_store_sales", "sku_stats"):
            if not _table_exists(conn, tbl):
                return {"status": "error",
                        "message": f"Chưa có bảng {tbl} trong DB. Chạy: "
                                   "python backend/scripts/build_sales_cache.py / build_sku_stats.py"}

        never_sold = pd.read_sql_query(f"""
            SELECT i.item_nbr, it.family, i.current_stock,
                   ROUND(i.current_stock * {_cost_price_expr()}, 2) AS stock_value_usd
            FROM inventory i
            LEFT JOIN agg_item_store_sales a ON a.store_nbr = i.store_nbr AND a.item_nbr = i.item_nbr
            JOIN items it ON i.item_nbr = it.item_nbr
            JOIN family_prices fp ON fp.family = it.family
            WHERE i.store_nbr = ? AND i.current_stock > 0
              AND (a.unit_sales IS NULL OR a.unit_sales = 0)
            ORDER BY stock_value_usd DESC LIMIT ?
        """, conn, params=[int(store_nbr), top_n])

        recent_dead = pd.read_sql_query(f"""
            SELECT i.item_nbr, it.family, i.current_stock,
                   ss.avg_daily_45d,
                   ROUND(i.current_stock * {_cost_price_expr()}, 2) AS stock_value_usd
            FROM inventory i
            JOIN sku_stats ss ON ss.item_nbr = i.item_nbr
            JOIN items it ON i.item_nbr = it.item_nbr
            JOIN family_prices fp ON fp.family = it.family
            WHERE i.store_nbr = ? AND i.current_stock > 0
              AND COALESCE(ss.avg_daily_45d, 0) < 0.01
            ORDER BY stock_value_usd DESC LIMIT ?
        """, conn, params=[int(store_nbr), top_n])

        return {
            "store_nbr": int(store_nbr),
            "never_sold_here_count": int(len(never_sold)),
            "never_sold_here": never_sold.to_dict(orient="records"),
            "recent_dead_chainwide_count": int(len(recent_dead)),
            "recent_dead_chainwide": recent_dead.to_dict(orient="records"),
            "note": "(a) chưa từng bán ở cửa hàng này trong toàn bộ lịch sử; "
                    "(b) trung bình 45 ngày gần nhất toàn chuỗi ≈ 0. Giá vốn là giá tham chiếu.",
        }
    except Exception as e:
        logger.error(f"Error in find_dead_stock: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        if conn:
            conn.close()


def get_abc_analysis(store_nbr: int, top_n: int = 5) -> Dict[str, Any]:
    """
    Phân loại ABC THEO CỬA HÀNG: giá trị bán = unit_sales x giá bán tham chiếu;
    cộng dồn A ≤ 80%, B ≤ 95%, C còn lại (khớp định nghĩa abc_class của sku_stats).
    """
    conn = None
    try:
        top_n = max(1, min(int(top_n or 5), 10))
        conn = get_db_connection()
        if not _table_exists(conn, "agg_item_store_sales"):
            return {"status": "error",
                    "message": "Chưa có bảng agg_item_store_sales trong DB. Chạy một lần: "
                               "python backend/scripts/build_sales_cache.py"}
        df = pd.read_sql_query(f"""
            SELECT a.item_nbr, it.family,
                   ROUND(a.unit_sales * fp.unit_price, 2) AS sales_value_usd
            FROM agg_item_store_sales a
            JOIN items it ON a.item_nbr = it.item_nbr
            JOIN family_prices fp ON fp.family = it.family
            WHERE a.store_nbr = ? AND a.unit_sales > 0
            ORDER BY sales_value_usd DESC
        """, conn, params=[int(store_nbr)])
        if df.empty:
            return {"status": "error", "message": f"Không có dữ liệu bán hàng cho cửa hàng {store_nbr}."}
        total = float(df["sales_value_usd"].sum())
        df["cum_share_pct"] = (df["sales_value_usd"].cumsum() / total * 100).round(2)
        df["abc_class"] = df["cum_share_pct"].apply(lambda c: "A" if c <= 80 else ("B" if c <= 95 else "C"))
        summary = {}
        for cls in ("A", "B", "C"):
            sub = df[df["abc_class"] == cls]
            summary[cls] = {
                "item_count": int(len(sub)),
                "value_share_pct": round(float(sub["sales_value_usd"].sum()) / total * 100, 2) if len(sub) else 0.0,
                "top_items": sub.head(top_n)[["item_nbr", "family", "sales_value_usd", "cum_share_pct"]].to_dict(orient="records"),
            }
        return {
            "store_nbr": int(store_nbr),
            "total_sales_value_usd": round(total, 2),
            "classes": summary,
            "thresholds": "A ≤ 80% giá trị cộng dồn, B ≤ 95%, C còn lại (giá bán tham chiếu)",
        }
    except Exception as e:
        logger.error(f"Error in get_abc_analysis: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        if conn:
            conn.close()


def analyze_weekly_pattern(store_nbr: int, weeks: int = 12) -> Dict[str, Any]:
    """
    Mẫu tuần: doanh thu bình quân theo ngày trong tuần (cửa sổ N tuần) - dùng để
    xếp ca nhân viên và chọn ngày chạy khuyến mãi.
    """
    conn = None
    try:
        weeks = max(2, min(int(weeks or 12), 52))
        conn = get_db_connection()
        if not _table_exists(conn, "agg_daily_business"):
            return {"status": "error",
                    "message": "Chưa có bảng agg_daily_business trong DB. Chạy một lần: "
                               "python backend/scripts/build_business_cache.py"}
        df = pd.read_sql_query(f"""
            SELECT CAST(strftime('%w', date) AS INTEGER) AS dow,
                   ROUND(AVG(revenue), 2) AS avg_daily_revenue_usd,
                   COUNT(*) AS days_observed
            FROM agg_daily_business
            WHERE store_nbr = ?
              AND date >= date((SELECT MAX(date) FROM agg_daily_business), '-{weeks * 7} day')
            GROUP BY dow ORDER BY dow
        """, conn, params=[int(store_nbr)])
        if df.empty:
            return {"status": "error", "message": f"Không có dữ liệu cho cửa hàng {store_nbr}."}
        dow_names = {0: "Chủ nhật", 1: "Thứ 2", 2: "Thứ 3", 3: "Thứ 4",
                     4: "Thứ 5", 5: "Thứ 6", 6: "Thứ 7"}
        df["day"] = df["dow"].map(dow_names)
        weekend_avg = float(df[df["dow"].isin([0, 6])]["avg_daily_revenue_usd"].mean())
        weekday_avg = float(df[~df["dow"].isin([0, 6])]["avg_daily_revenue_usd"].mean())
        best = df.loc[df["avg_daily_revenue_usd"].idxmax()]
        worst = df.loc[df["avg_daily_revenue_usd"].idxmin()]
        return {
            "store_nbr": int(store_nbr),
            "window_weeks": weeks,
            "by_day_of_week": df[["day", "avg_daily_revenue_usd", "days_observed"]].to_dict(orient="records"),
            "weekend_avg_usd": round(weekend_avg, 2),
            "weekday_avg_usd": round(weekday_avg, 2),
            "weekend_lift_pct": round((weekend_avg - weekday_avg) / weekday_avg * 100, 2) if weekday_avg > 0 else None,
            "best_day": best["day"], "worst_day": worst["day"],
        }
    except Exception as e:
        logger.error(f"Error in analyze_weekly_pattern: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        if conn:
            conn.close()


def compare_family_mix(store_nbr: int) -> Dict[str, Any]:
    """
    Cơ cấu ngành hàng: tỷ trọng giá trị bán theo ngành của cửa hàng so với toàn
    chuỗi - tìm ngành over-indexed (nên đẩy mạnh) / under-indexed (đang thiếu).
    """
    conn = None
    try:
        conn = get_db_connection()
        if not _table_exists(conn, "agg_item_store_sales"):
            return {"status": "error",
                    "message": "Chưa có bảng agg_item_store_sales trong DB. Chạy một lần: "
                               "python backend/scripts/build_sales_cache.py"}
        def _mix(where: str, params: list) -> Dict[str, float]:
            df = pd.read_sql_query(f"""
                SELECT it.family, SUM(a.unit_sales * fp.unit_price) AS value_usd
                FROM agg_item_store_sales a
                JOIN items it ON a.item_nbr = it.item_nbr
                JOIN family_prices fp ON fp.family = it.family
                {where}
                GROUP BY it.family
            """, conn, params=params)
            total = float(df["value_usd"].sum())
            return {r["family"]: float(r["value_usd"]) / total * 100 for _, r in df.iterrows()} if total > 0 else {}

        store_mix = _mix("WHERE a.store_nbr = ?", [int(store_nbr)])
        chain_mix = _mix("", [])
        if not store_mix or not chain_mix:
            return {"status": "error", "message": f"Không đủ dữ liệu cơ cấu ngành cho cửa hàng {store_nbr}."}
        rows = []
        for fam, s_share in store_mix.items():
            c_share = chain_mix.get(fam, 0.0)
            rows.append({"family": fam, "store_share_pct": round(s_share, 2),
                         "chain_share_pct": round(c_share, 2),
                         "diff_pp": round(s_share - c_share, 2)})
        rows.sort(key=lambda r: r["diff_pp"])
        return {
            "store_nbr": int(store_nbr),
            "under_indexed_worst_first": [r for r in rows if r["diff_pp"] < -1.0][:8],
            "over_indexed_top_first": [r for r in reversed(rows) if r["diff_pp"] > 1.0][:8],
            "note": "diff_pp = tỷ trọng cửa hàng - tỷ trọng chuỗi (điểm %); giá trị theo giá bán tham chiếu.",
        }
    except Exception as e:
        logger.error(f"Error in compare_family_mix: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        if conn:
            conn.close()


# ======================================================================
# 7b. MAPPING DICTIONARY & DISPATCHER
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
    "analyze_gross_margin": analyze_gross_margin,
    "analyze_revenue_change": analyze_revenue_change,
    "benchmark_store_vs_peers": benchmark_store_vs_peers,
    "analyze_reorder_profitability": analyze_reorder_profitability,
    "analyze_inventory_health": analyze_inventory_health,
    "find_dead_stock": find_dead_stock,
    "get_abc_analysis": get_abc_analysis,
    "analyze_weekly_pattern": analyze_weekly_pattern,
    "compare_family_mix": compare_family_mix,
}


# ======================================================================
# 8. JSON SCHEMAS CHO GROQ / QWEN 3.6 FUNCTION CALLING
# ======================================================================

def _fn(name: str, desc: str, props: dict, req: list = None) -> dict:
    """Schema function-calling gọn: props = {tên: (type, mô tả ngắn)} - tiết kiệm token TPM."""
    return {"type": "function", "function": {
        "name": name,
        "description": desc,
        "parameters": {
            "type": "object",
            "properties": {k: {"type": t, "description": d} for k, (t, d) in props.items()},
            **({"required": req} if req else {}),
        },
    }}


_STORE = ("integer", "Mã cửa hàng.")
_ITEM = ("integer", "Mã mặt hàng.")
_MONTHS = ("integer", "Số tháng gần nhất (mặc định 3).")
_TOPN = ("integer", "Số mặt hàng cần xem (mặc định 5).")

GROQ_TOOL_DEFINITIONS = [
    _fn("get_sales_summary", "Tổng doanh số DỰ BÁO trong N ngày tới của một cửa hàng hoặc toàn chuỗi.",
        {"store_nbr": _STORE, "days": ("integer", "Số ngày dự báo (mặc định 7).")}),
    _fn("check_stockout_risk", "Mặt hàng có nguy cơ HẾT HÀNG tại một cửa hàng (dự báo 16 ngày vượt tồn).",
        {"store_nbr": _STORE}, ["store_nbr"]),
    _fn("calculate_reorder_point", "Điểm đặt hàng lại (ROP) và tồn kho an toàn cho 1 mặt hàng.",
        {"store_nbr": _STORE, "item_nbr": _ITEM}, ["store_nbr", "item_nbr"]),
    _fn("calculate_purchase_target", "Số lượng cần đặt thêm cho 1 mặt hàng theo nhu cầu 16 ngày.",
        {"store_nbr": _STORE, "item_nbr": _ITEM}, ["store_nbr", "item_nbr"]),
    _fn("simulate_demand_multiplier",
        "Mô phỏng khi cầu tăng/giảm theo hệ số cho 1 mặt hàng (1.5 = +50%).",
        {"store_nbr": _STORE, "item_nbr": _ITEM, "multiplier": ("number", "Hệ số cầu.")},
        ["store_nbr", "item_nbr", "multiplier"]),
    _fn("evaluate_stockout_loss", "Thiệt hại (unit + USD) nếu 1 mặt hàng đứt hàng N ngày.",
        {"store_nbr": _STORE, "item_nbr": _ITEM, "out_of_stock_days": ("integer", "Số ngày đứt hàng.")},
        ["store_nbr", "item_nbr", "out_of_stock_days"]),
    _fn("run_scenario_analysis",
        "Kịch bản what-if cho 1 NGÀNH HÀNG: sửa cầu/khuyến mãi/giá dầu/lưu lượng/sự kiện/tồn kho "
        "-> dự báo lại bằng model thật. Kết quả có sẵn `analysis` + `recommendation`: dùng nguyên "
        "trạng, KHÔNG tự tính lại số.",
        {"store_nbr": _STORE, "family": ("string", "Tên ngành hàng, VD: GROCERY I."),
         "demand_multiplier": ("number", "Hệ số cầu (bỏ trống = giữ)."),
         "promo_days": ("integer", "Số ngày khuyến mãi 0-16 (bỏ trống = thật)."),
         "oil_price": ("number", "Giá dầu USD (bỏ trống = thật)."),
         "traffic_change_pct": ("number", "% thay đổi khách (bỏ trống = giữ)."),
         "event_type": ("string", "none/holiday/earthquake."),
         "event_days": ("integer", "Số ngày sự kiện."),
         "stock_override": ("number", "Tồn kho giả lập (bỏ trống = thật)."),
         "lead_time_override": ("number", "Lead time giả lập ngày."),
         "horizon_days": ("integer", "Số ngày dự báo 7-16.")},
        ["store_nbr", "family"]),
    _fn("recommend_slow_mover_strategy",
        "Mặt hàng đọng vốn (tồn > 2x nhu cầu 16 ngày) + mức giảm giá đề xuất.",
        {"store_nbr": _STORE}, ["store_nbr"]),
    _fn("find_cross_sell_items", "Top mặt hàng cùng ngành để bán kèm cho 1 mặt hàng.",
        {"store_nbr": _STORE, "item_nbr": _ITEM}, ["store_nbr", "item_nbr"]),
    _fn("compare_cluster_trends", "So tổng doanh số dự báo giữa 2 cụm cửa hàng (cluster).",
        {"cluster_1": ("integer", "Mã cụm 1."), "cluster_2": ("integer", "Mã cụm 2.")},
        ["cluster_1", "cluster_2"]),
    _fn("get_monthly_revenue",
        "DOANH THU THỰC TẾ (USD) theo tháng của kỳ ĐÃ QUA; đọc data_period để nêu đúng kỳ.",
        {"store_nbr": _STORE, "months": ("integer", "Số tháng gần nhất (mặc định 1).")}),
    _fn("compare_stores_revenue", "So doanh thu thực tế 2 cửa hàng theo tháng + cửa hàng mạnh hơn.",
        {"store_1": ("integer", "Mã cửa hàng 1."), "store_2": ("integer", "Mã cửa hàng 2."),
         "months": _MONTHS}, ["store_1", "store_2"]),
    _fn("get_top_selling_items", "Top bán chạy THỰC TẾ theo unit (không phải dự báo).",
        {"store_nbr": _STORE, "top_n": _TOPN}),
    _fn("get_family_forecast", "Dự báo theo NGÀY của 1 ngành hàng (family) tại 1 cửa hàng.",
        {"store_nbr": _STORE, "family": ("string", "Tên ngành, VD: PRODUCE."),
         "days": ("integer", "Số ngày (mặc định 7, tối đa 16).")}, ["store_nbr", "family"]),
    _fn("get_item_profile", "Hồ sơ 1 mặt hàng: family, dễ hỏng, tồn + dự báo theo từng cửa hàng.",
        {"item_nbr": _ITEM}, ["item_nbr"]),
    _fn("get_store_profile",
        "Thông tin cửa hàng (địa điểm, loại, cụm, doanh thu tháng gần nhất); bỏ trống = danh sách cửa hàng.",
        {"store_nbr": _STORE}),
    _fn("get_store_traffic", "Lượt khách (số hóa đơn/ngày) + xu hướng tăng/giảm % so kỳ trước.",
        {"store_nbr": _STORE, "days": ("integer", "Độ dài kỳ (mặc định 30).")}),
    _fn("evaluate_promotion_impact", "Hiệu quả khuyến mãi: doanh số ngày có KM vs không + lift %.",
        {"store_nbr": _STORE, "family": ("string", "Ngành cụ thể (tùy chọn).")}, ["store_nbr"]),
    _fn("check_perishable_risk", "Mặt hàng DỄ HỎNG có dự báo vượt tồn trong 16 ngày tới.",
        {"store_nbr": _STORE}, ["store_nbr"]),
    _fn("analyze_gross_margin",
        "Biên lợi nhuận gộp (doanh thu - trả hàng - giá vốn) theo tháng; bỏ trống store_nbr = xếp hạng cửa hàng.",
        {"store_nbr": _STORE, "months": _MONTHS}),
    _fn("analyze_revenue_change", "Phân tách Δdoanh thu 2 tháng: lượt khách vs giá trị mỗi hóa đơn.",
        {"store_nbr": _STORE, "months": ("integer", "Số tháng (mặc định 2).")}, ["store_nbr"]),
    _fn("benchmark_store_vs_peers", "So cửa hàng với trung bình cửa hàng cùng loại: doanh thu, biên gộp, ticket.",
        {"store_nbr": _STORE, "months": _MONTHS}, ["store_nbr"]),
    _fn("analyze_reorder_profitability", "Cơ hội lợi nhuận nếu đặt thêm hàng đủ nhu cầu, xếp hạng mặt hàng.",
        {"store_nbr": _STORE, "top_n": _TOPN}, ["store_nbr"]),
    _fn("analyze_inventory_health", "Giá trị tồn, DOH, vòng quay, overstock > 30 ngày.",
        {"store_nbr": _STORE, "months": _MONTHS}, ["store_nbr"]),
    _fn("find_dead_stock", "Hàng chết: còn tồn nhưng không bán (chưa từng bán ở đây / 45 ngày toàn chuỗi ≈ 0).",
        {"store_nbr": _STORE, "top_n": _TOPN}, ["store_nbr"]),
    _fn("get_abc_analysis", "Phân loại ABC theo cửa hàng: A ≤ 80%, B ≤ 95% giá trị cộng dồn.",
        {"store_nbr": _STORE, "top_n": _TOPN}, ["store_nbr"]),
    _fn("analyze_weekly_pattern", "Doanh thu theo ngày trong tuần + lift cuối tuần vs ngày thường.",
        {"store_nbr": _STORE, "weeks": ("integer", "Cửa sổ tuần (mặc định 12).")}, ["store_nbr"]),
    _fn("compare_family_mix", "Cơ cấu ngành của cửa hàng vs chuỗi: thiếu/dư ngành nào.",
        {"store_nbr": _STORE}, ["store_nbr"]),
]
