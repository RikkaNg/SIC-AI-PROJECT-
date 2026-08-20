"""
supply_chain_tools.py
Bộ công cụ Python (Function Calling Tools) cho LLM Agent.
Thực hiện các tính toán Toán học & Truy vấn Database.
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "database" / "retail.db"


def get_db_connection():
    """Tạo kết nối SQLite."""
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at {DB_PATH}. Run init_database.py first.")
    conn = sqlite3.connect(DB_PATH)
    return conn


# ======================================================================
# 1. NHÓM THỐNG KÊ & SO SÁNH DOANH SỐ
# ======================================================================

def get_sales_summary(store_nbr: int = None, days: int = 7) -> dict: # type: ignore
    """
    Tính tổng doanh số dự báo trong N ngày tới của một cửa hàng hoặc toàn hệ thống.
        Args:
            store_nbr (int, optional): Mã cửa hàng. Nếu None thì lấy tất cả.
            days (int): Số ngày dự báo muốn tính (mặc định 7 ngày).
        Returns:
            dict: Tổng doanh số, số mặt hàng, và Top 3 mặt hàng bán chạy nhất.
    """
    try:
        conn = get_db_connection()
        query = """
                SELECT item_nbr, SUM(predicted_sales) as total_sales
                FROM forecasts
                WHERE date <= (SELECT MAX (date) FROM forecasts)
                  AND date \
                    > (SELECT MAX (date) FROM forecasts) - ? \
                """
        params = [days]

        if store_nbr:
            query += " AND store_nbr = ?"
            params.append(store_nbr)

        query += " GROUP BY item_nbr ORDER BY total_sales DESC"

        df = pd.read_sql_query(query, conn, params=tuple(params))
        conn.close()

        if df.empty:
            return {"status": "error", "message": "Không có dữ liệu dự báo."}

        total_sales = df['total_sales'].sum()
        top_items = df.head(3).to_dict(orient='records')

        return {
            "total_forecast_sales": round(float(total_sales), 2),
            "total_items": len(df),
            "top_3_items": top_items
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ======================================================================
# 2. NHÓM QUẢN TRỊ TỒN KHO & CUNG ỨNG
# ======================================================================

def check_stockout_risk(store_nbr: int) -> dict:
    """
    Kiểm tra các mặt hàng có rủi ro cháy hàng (hết kho) trong 16 ngày tới.
        Args:
            store_nbr (int): Mã cửa hàng cần kiểm tra.
        Returns:
            dict: Danh sách các mặt hàng rủi ro hết hàng, số lượng tồn kho và dự báo tiêu thụ.
    """
    try:
        conn = get_db_connection()
        query = """
                SELECT f.item_nbr,
                       SUM(f.predicted_sales) as forecast_demand,
                       i.current_stock
                FROM forecasts f
                         JOIN inventory i ON f.store_nbr = i.store_nbr AND f.item_nbr = i.item_nbr
                WHERE f.store_nbr = ?
                GROUP BY f.item_nbr, i.current_stock
                HAVING forecast_demand > i.current_stock
                ORDER BY (forecast_demand - i.current_stock) DESC LIMIT 10 \
                """
        df = pd.read_sql_query(query, conn, params=(store_nbr,))
        conn.close()

        if df.empty:
            return {"status": "success", "message": f"Không có mặt hàng nào rủi ro hết kho tại cửa hàng {store_nbr}."}

        return {
            "store_nbr": store_nbr,
            "risk_items_count": len(df),
            "items_at_risk": df.to_dict(orient='records')
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def calculate_reorder_point(store_nbr: int, item_nbr: int) -> dict:
    """
    Tính Điểm đặt hàng lại (Reorder Point - ROP) và Tồn kho an toàn (Safety Stock).
        Args:
            store_nbr (int): Mã cửa hàng.
            item_nbr (int): Mã mặt hàng.
        Returns:
            dict: Tồn kho hiện tại, nhu cầu trung bình ngày, Lead time, Safety Stock, ROP.
    """
    try:
        conn = get_db_connection()
        query = """
                SELECT i.current_stock, \
                       i.lead_time_days,
                       AVG(f.predicted_sales) as avg_daily_demand,
                       f.item_nbr
                FROM inventory i
                         JOIN forecasts f ON i.store_nbr = f.store_nbr AND i.item_nbr = f.item_nbr
                WHERE i.store_nbr = ? \
                  AND i.item_nbr = ?
                GROUP BY i.current_stock, i.lead_time_days, f.item_nbr \
                """
        df = pd.read_sql_query(query, conn, params=[store_nbr, item_nbr])
        conn.close()

        if df.empty:
            return {"status": "error", "message": "Không tìm thấy mặt hàng này trong kho."}

        row = df.iloc[0]
        lead_time = row['lead_time_days']
        avg_demand = row['avg_daily_demand']

        # Công thức Safety Stock: Z * σ * sqrt(L) (Z=1.65 cho 95% service level)
        # Ở đây dùng biến động của dự báo để xấp xỉ std_dev
        std_demand = 0.2 * avg_demand  # Giả định độ lệch chuẩn 20% nhu cầu
        safety_stock = 1.65 * std_demand * np.sqrt(lead_time)

        # ROP = Avg Demand * Lead Time + Safety Stock
        rop = (avg_demand * lead_time) + safety_stock

        return {
            "item_nbr": int(item_nbr),
            "current_stock": float(row['current_stock']),
            "avg_daily_demand": round(float(avg_demand), 2),
            "lead_time_days": int(lead_time),
            "safety_stock": round(float(safety_stock), 2),
            "reorder_point": round(float(rop), 2),
            "need_reorder": bool(row['current_stock'] <= rop)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ======================================================================
# 3. NHÓM MÔ PHỎNG KỊCH BẢN (WHAT-IF SCENARIOS)
# ======================================================================

def simulate_demand_multiplier(store_nbr: int, item_nbr: int, multiplier: float) -> dict:
    """
    Mô phỏng kịch bản doanh số tăng/giảm (VD: x1.5 nếu có siêu khuyến mãi, x0.8 nếu kinh tế suy thoái).
        Args:
            store_nbr (int): Mã cửa hàng.
            item_nbr (int): Mã mặt hàng.
            multiplier (float): Hệ số nhân (VD: 1.5 = tăng 50%).
        Returns:
            dict: Tổng nhu cầu mới, tồn kho hiện tại, và chênh lệch so với dự báo cũ.
    """
    try:
        conn = get_db_connection()
        query = """
                SELECT SUM(f.predicted_sales) as old_forecast, i.current_stock
                FROM forecasts f
                         JOIN inventory i ON f.store_nbr = i.store_nbr AND f.item_nbr = i.item_nbr
                WHERE f.store_nbr = ? \
                  AND f.item_nbr = ? \
                """
        df = pd.read_sql_query(query, conn, params=[store_nbr, item_nbr])
        conn.close()

        if df.empty or df.iloc[0]['old_forecast'] is None:
            return {"status": "error", "message": "Không có dữ liệu cho mặt hàng này."}

        row = df.iloc[0]
        old_forecast = row['old_forecast']
        new_forecast = old_forecast * multiplier
        current_stock = row['current_stock']

        return {
            "item_nbr": int(item_nbr),
            "old_forecast_demand": round(float(old_forecast), 2),
            "new_forecast_demand": round(float(new_forecast), 2),
            "current_stock": float(current_stock),
            "will_stockout": bool(new_forecast > current_stock),
            "shortfall_quantity": round(float(max(0, new_forecast - current_stock)), 2)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ======================================================================
# 4. NHÓM ĐÁNH GIÁ CHẤT LƯỢNG GIAO DỊCH
# ======================================================================

def analyze_transaction_anomaly(expected_sales: float, actual_sales: float) -> dict:
    """
    Đánh giá xem một giao dịch thực tế là tích cực (bán đột biến) hay tiêu cực (sụt giảm) so với dự báo.
        Args:
            expected_sales (float): Doanh số dự báo kỳ vọng.
            actual_sales (float): Doanh số thực tế xảy ra.
        Returns:
            dict: Phân loại bất thường, tỷ lệ chênh lệch.
    """
    try:
        if expected_sales == 0:
            deviation = 0.0 if actual_sales == 0 else 100.0
        else:
            deviation = ((actual_sales - expected_sales) / expected_sales) * 100

        if abs(deviation) < 15.0:
            status = "Normal"
        elif deviation > 0:
            status = "Positive Spike (Bán chạy bất thường)"
        else:
            status = "Negative Drop (Sụt giảm bất thường)"

        return {
            "expected_sales": float(expected_sales),
            "actual_sales": float(actual_sales),
            "deviation_pct": round(float(deviation), 2),
            "anomaly_status": status
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def evaluate_stockout_loss(store_nbr: int, item_nbr: int, out_of_stock_days: int) -> dict:
    """
    Đánh giá thiệt hại doanh thu tiềm năng do một mặt hàng bị cháy hàng (Out of Stock).
        Args:
            store_nbr (int): Mã cửa hàng.
            item_nbr (int): Mã mặt hàng.
            out_of_stock_days (int): Số ngày mặt hàng bị hết kho.
        Returns:
            dict: Số lượng thất thoát, ước tính doanh thu mất đi.
    """
    try:
        conn = get_db_connection()
        query = """
                SELECT AVG(predicted_sales) as avg_daily_demand
                FROM forecasts
                WHERE store_nbr = ? \
                  AND item_nbr = ? \
                """
        df = pd.read_sql_query(query, conn, params=[store_nbr, item_nbr])
        conn.close()

        if df.empty or df.iloc[0]['avg_daily_demand'] is None:
            return {"status": "error", "message": "Không tìm thấy dữ liệu dự báo cho mặt hàng này."}

        avg_demand = df.iloc[0]['avg_daily_demand']
        lost_quantity = avg_demand * out_of_stock_days

        # Giả lập giá bán trung bình = 5 USD/sản phẩm (Vì DB chưa có dữ liệu giá vốn)
        mock_price = 5.0
        lost_revenue = lost_quantity * mock_price

        return {
            "item_nbr": int(item_nbr),
            "out_of_stock_days": int(out_of_stock_days),
            "avg_daily_demand": round(float(avg_demand), 2),
            "lost_sales_quantity": round(float(lost_quantity), 2),
            "estimated_lost_revenue_usd": round(float(lost_revenue), 2),
            "impact_level": "Critical Negative Impact" if lost_revenue > 100 else "Moderate Impact"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ======================================================================
# MAPPING DICTIONARY CHO LLM AGENT
# ======================================================================
# Khai báo các hàm ở trên thành dạng mapping để file Agent (câu sau) dễ dàng gọi
AVAILABLE_FUNCTIONS = {
    "get_sales_summary": get_sales_summary,
    "check_stockout_risk": check_stockout_risk,
    "calculate_reorder_point": calculate_reorder_point,
    "simulate_demand_multiplier": simulate_demand_multiplier,
    "analyze_transaction_anomaly": analyze_transaction_anomaly,
    "evaluate_stockout_loss": evaluate_stockout_loss
}

# Test thử hàm khi chạy file trực tiếp
if __name__ == "__main__":
    print("Đang test các tools...\n")

    # Test 1: Lấy tổng doanh số 7 ngày tới của toàn hệ thống
    print("1. Get Sales Summary (Toàn hệ thống, 7 ngày):")
    print(get_sales_summary(days=7))

    # Test 2: Kiểm tra rủi ro cháy hàng ở cửa hàng 25
    print("\n2. Check Stockout Risk (Cửa hàng 25):")
    print(check_stockout_risk(store_nbr=25))

    # Test 3: Tính ROP cho item 1041 tại store 25
    print("\n3. Calculate Reorder Point (Store 25, Item 1041):")
    print(calculate_reorder_point(store_nbr=25, item_nbr=1041))

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_sales_summary",
            "description": "Tính tổng doanh số dự báo trong N ngày tới của một cửa hàng hoặc toàn hệ thống.",
            "parameters": {
                "type": "object",
                "properties": {
                    "store_nbr": {"type": "integer", "description": "Mã cửa hàng (Nếu không nhập sẽ lấy toàn hệ thống)"},
                    "days": {"type": "integer", "description": "Số ngày dự báo muốn tính (mặc định 7 ngày)", "default": 7}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_stockout_risk",
            "description": "Kiểm tra các mặt hàng có rủi ro cháy hàng (hết kho) trong 16 ngày tới tại 1 cửa hàng.",
            "parameters": {
                "type": "object",
                "properties": {
                    "store_nbr": {"type": "integer", "description": "Mã cửa hàng cần kiểm tra"}
                },
                "required": ["store_nbr"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_reorder_point",
            "description": "Tính Điểm đặt hàng lại (Reorder Point - ROP) và Tồn kho an toàn (Safety Stock) cho 1 mặt hàng.",
            "parameters": {
                "type": "object",
                "properties": {
                    "store_nbr": {"type": "integer", "description": "Mã cửa hàng"},
                    "item_nbr": {"type": "integer", "description": "Mã mặt hàng"}
                },
                "required": ["store_nbr", "item_nbr"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "simulate_demand_multiplier",
            "description": "Mô phỏng kịch bản doanh số tăng/giảm (VD: x1.5 nếu có khuyến mãi, x0.8 nếu suy thoái).",
            "parameters": {
                "type": "object",
                "properties": {
                    "store_nbr": {"type": "integer", "description": "Mã cửa hàng"},
                    "item_nbr": {"type": "integer", "description": "Mã mặt hàng"},
                    "multiplier": {"type": "number", "description": "Hệ số nhân (VD: 1.5 = tăng 50%)"}
                },
                "required": ["store_nbr", "item_nbr", "multiplier"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_transaction_anomaly",
            "description": "Đánh giá xem một giao dịch thực tế là tích cực (bán đột biến) hay tiêu cực (sụt giảm) so với dự báo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expected_sales": {"type": "number", "description": "Doanh số dự báo kỳ vọng"},
                    "actual_sales": {"type": "number", "description": "Doanh số thực tế xảy ra"}
                },
                "required": ["expected_sales", "actual_sales"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_stockout_loss",
            "description": "Đánh giá thiệt hại doanh thu tiềm năng do một mặt hàng bị cháy hàng (Out of Stock).",
            "parameters": {
                "type": "object",
                "properties": {
                    "store_nbr": {"type": "integer", "description": "Mã cửa hàng"},
                    "item_nbr": {"type": "integer", "description": "Mã mặt hàng"},
                    "out_of_stock_days": {"type": "integer", "description": "Số ngày mặt hàng bị hết kho"}
                },
                "required": ["store_nbr", "item_nbr", "out_of_stock_days"]
            }
        }
    }
]

# ==========================================
# MAPPING DICTIONARY
# ==========================================
# Đảm bảo tên biến này là AVAILABLE_FUNCTIONS
AVAILABLE_FUNCTIONS = {
    "get_sales_summary": get_sales_summary,
    "check_stockout_risk": check_stockout_risk,
    "calculate_reorder_point": calculate_reorder_point,
    "simulate_demand_multiplier": simulate_demand_multiplier,
    "analyze_transaction_anomaly": analyze_transaction_anomaly,
    "evaluate_stockout_loss": evaluate_stockout_loss
}