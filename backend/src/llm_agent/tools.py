"""
tools.py
Bộ công cụ Function Calling & Phân tích Tồn kho cho LLM Agent.
"""

import os
import sqlite3
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

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


def get_db_connection() -> sqlite3.Connection:
    """Tạo kết nối SQLite an toàn với chế độ Read-Only / WAL."""
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at {DB_PATH}. Vui lòng chạy init_database.py trước.")
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


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
            SELECT f.item_nbr, it.family, SUM(f.predicted_sales) as total_sales
            FROM forecasts f
            LEFT JOIN items it ON f.item_nbr = it.item_nbr
            WHERE f.date >= (SELECT MIN(date) FROM forecasts)
              AND f.date < date((SELECT MIN(date) FROM forecasts), '+' || ? || ' day')
        """
        params = [days]

        if store_nbr is not None:
            query += " AND f.store_nbr = ?"
            params.append(store_nbr)
        elif _allowed_stores:
            # RLS: giới hạn tổng hợp trong phạm vi cửa hàng của user
            marks = ",".join("?" * len(_allowed_stores))
            query += f" AND f.store_nbr IN ({marks})"
            params.extend(sorted(int(s) for s in _allowed_stores))

        query += " GROUP BY f.item_nbr, it.family ORDER BY total_sales DESC"

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
                it.family,
                ROUND(SUM(f.predicted_sales), 2) as forecast_demand,
                i.current_stock,
                i.lead_time_days,
                ROUND(SUM(f.predicted_sales) - i.current_stock, 2) as deficit_quantity
            FROM forecasts f
            JOIN inventory i ON f.store_nbr = i.store_nbr AND f.item_nbr = i.item_nbr
            LEFT JOIN items it ON f.item_nbr = it.item_nbr
            WHERE f.store_nbr = ?
            GROUP BY f.item_nbr, it.family, i.current_stock, i.lead_time_days
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
                it.family
            FROM inventory i
            JOIN forecasts f ON i.store_nbr = f.store_nbr AND i.item_nbr = f.item_nbr
            LEFT JOIN items it ON i.item_nbr = it.item_nbr
            WHERE i.store_nbr = ? AND i.item_nbr = ?
            GROUP BY i.current_stock, i.lead_time_days, f.item_nbr, it.family
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
                i.lead_time_days
            FROM forecasts f
            JOIN inventory i ON f.store_nbr = i.store_nbr AND f.item_nbr = i.item_nbr
            WHERE f.store_nbr = ? AND f.item_nbr = ?
            GROUP BY i.current_stock, i.lead_time_days
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
        conn = get_db_connection()
        query = """
            SELECT SUM(f.predicted_sales) as old_forecast, i.current_stock
            FROM forecasts f
            JOIN inventory i ON f.store_nbr = i.store_nbr AND f.item_nbr = i.item_nbr
            WHERE f.store_nbr = ? AND f.item_nbr = ?
            GROUP BY i.current_stock
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
        conn = get_db_connection()
        query = "SELECT AVG(predicted_sales) as avg_daily_demand FROM forecasts WHERE store_nbr = ? AND item_nbr = ?"
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
        item_info = pd.read_sql_query("SELECT family FROM items WHERE item_nbr = ?", conn, params=[item_nbr])
        if item_info.empty:
            return {"status": "error", "message": f"Không tìm thấy item_nbr {item_nbr}."}
        
        family = item_info.iloc[0]['family']

        query = """
            SELECT f.item_nbr, SUM(f.predicted_sales) as total_sales
            FROM forecasts f
            JOIN items i ON f.item_nbr = i.item_nbr
            WHERE f.store_nbr = ? AND i.family = ? AND f.item_nbr != ?
            GROUP BY f.item_nbr
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
                it.family,
                ROUND(SUM(f.predicted_sales), 2) as forecast_16d, 
                i.current_stock
            FROM forecasts f
            JOIN inventory i ON f.store_nbr = i.store_nbr AND f.item_nbr = i.item_nbr
            LEFT JOIN items it ON f.item_nbr = it.item_nbr
            WHERE f.store_nbr = ?
            GROUP BY f.item_nbr, it.family, i.current_stock
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
# 5. MAPPING DICTIONARY & DISPATCHER
# ======================================================================

AVAILABLE_TOOLS = {
    "get_sales_summary": get_sales_summary,
    "check_stockout_risk": check_stockout_risk,
    "calculate_reorder_point": calculate_reorder_point,
    "calculate_purchase_target": calculate_purchase_target,
    "simulate_demand_multiplier": simulate_demand_multiplier,
    "evaluate_stockout_loss": evaluate_stockout_loss,
    "find_cross_sell_items": find_cross_sell_items,
    "recommend_slow_mover_strategy": recommend_slow_mover_strategy,
    "compare_cluster_trends": compare_cluster_trends
}


# ======================================================================
# 6. JSON SCHEMAS CHO GROQ / QWEN 3.6 FUNCTION CALLING
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
    }
]