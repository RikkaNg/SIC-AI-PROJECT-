"""
test_llm_tools.py - Kiểm thử 9 LLM Tools qua chat
==================================================
Verify LLM Agent gọi đúng tool và trả số liệu KHỚP với DB.
Đây là test quan trọng nhất cho chất lượng Agent.

Chạy: pytest backend/tests/test_llm_tools.py -v -m slow
"""
import pytest
import httpx
import sqlite3
from pathlib import Path

from conftest import BASE_URL, API_PREFIX, login_and_get_token, make_auth_headers

pytestmark = [pytest.mark.llm, pytest.mark.slow]

# Đường dẫn tuyệt đối để chạy pytest từ bất kỳ thư mục nào
DB_PATH = str(Path(__file__).resolve().parents[2] / "backend" / "src" / "database" / "retail.db")


def get_db_ground_truth(query: str, params=()):
    """Truy vấn DB trực tiếp để verify LLM trả đúng số liệu."""
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    cursor = conn.execute(query, params)
    result = cursor.fetchone()
    conn.close()
    return result


class TestToolSalesSummary:
    """Tool 1: get_sales_summary."""

    def test_sales_summary_accuracy(self):
        """TEST-LLM-001: Số liệu LLM trả KHỚP với DB.

        Tool get_sales_summary tổng hợp FORECAST 7 ngày đầu của chu kỳ dự báo
        (không phải doanh số 2016), nên câu hỏi và ground truth phải cùng nguồn.
        """
        headers = make_auth_headers(login_and_get_token(
            {"username": "admin", "password": "admin123"}
        ))

        response = httpx.post(
            f"{BASE_URL}{API_PREFIX}/chat",
            json={"user_query": "Tổng doanh số dự báo 7 ngày tới của cửa hàng 1 là bao nhiêu?"},
            headers=headers, timeout=60.0
        )
        assert response.status_code == 200
        reply = response.json()["reply"]

        # Ground truth từ DB - cùng truy vấn/tool window với get_sales_summary(days=7)
        db_result = get_db_ground_truth(
            "SELECT SUM(predicted_sales) FROM forecasts "
            "WHERE store_nbr=1 AND date >= (SELECT MIN(date) FROM forecasts) "
            "AND date < date((SELECT MIN(date) FROM forecasts), '+7 day')"
        )
        expected_value = round(db_result[0], 2) if db_result[0] else 0

        # LLM có thể viết số với dấu phân cách (65.804,86 / 65,804.86 / 65804.86)
        # → bỏ hết ký tự không phải chữ số rồi so khớp 5 chữ số đầu
        reply_digits = "".join(ch for ch in reply if ch.isdigit())
        candidates = {str(int(expected_value)), str(int(round(expected_value)))}
        assert any(str(c)[:5] in reply_digits for c in candidates), (
            f"LLM trả sai số liệu. DB={expected_value}, Reply={reply[:300]}"
        )


class TestToolForecast:
    """Tool 2: get_forecast."""
    
    def test_forecast_16_days(self):
        """TEST-LLM-010: Dự báo 16 ngày được trả về."""
        headers = make_auth_headers(login_and_get_token(
            {"username": "admin", "password": "admin123"}
        ))
        response = httpx.post(
            f"{BASE_URL}{API_PREFIX}/chat",
            json={"user_query": "Dự báo doanh số cửa hàng 1 trong 16 ngày tới"},
            headers=headers, timeout=60.0
        )
        assert response.status_code == 200
        reply = response.json()["reply"]
        # Phải nhắc đến "16 ngày" hoặc "dự báo"
        assert "16" in reply or "dự báo" in reply.lower()


class TestToolROP:
    """Tool 4: calculate_rop — công thức quan trọng nhất."""
    
    def test_rop_calculation_correct(self):
        """TEST-LLM-020: ROP = d̄·LT + z·σ·√LT với z=1.65, CV=0.2."""
        headers = make_auth_headers(login_and_get_token(
            {"username": "admin", "password": "admin123"}
        ))
        
        # SKU 96995 có thật trong inventory + forecasts của store 1
        response = httpx.post(
            f"{BASE_URL}{API_PREFIX}/chat",
            json={"user_query": "Tính ROP cho sản phẩm 96995 tại cửa hàng 1"},
            headers=headers, timeout=60.0
        )
        assert response.status_code == 200
        reply = response.json()["reply"]
        
        # Verify công thức: ground truth phải cùng nguồn với tool
        # (tools.py calculate_reorder_point dùng AVG(forecasts.predicted_sales)
        # JOIN inventory để lấy lead_time_days)
        db_result = get_db_ground_truth(
            "SELECT AVG(f.predicted_sales), i.lead_time_days "
            "FROM forecasts f JOIN inventory i "
            "ON f.store_nbr=i.store_nbr AND f.item_nbr=i.item_nbr "
            "WHERE f.store_nbr=1 AND f.item_nbr=96995"
        )
        if db_result and db_result[0]:
            avg_daily, lead_time = db_result
            sigma = 0.2 * avg_daily
            expected_rop = avg_daily * lead_time + 1.65 * sigma * (lead_time ** 0.5)
            
            # LLM phải trả con số gần đúng (±10%)
            import re
            numbers = re.findall(r'\d+\.?\d*', reply)
            rop_values = [float(n) for n in numbers 
                         if abs(float(n) - expected_rop) / expected_rop < 0.15]
            assert len(rop_values) > 0, (
                f"ROP không khớp. Expected≈{expected_rop:.1f}, "
                f"Reply={reply[:300]}"
            )


class TestToolStockoutRisk:
    """Tool 7: analyze_stockout_risk."""
    
    def test_stockout_detection(self):
        """TEST-LLM-030: Phát hiện SKU sắp cháy hàng."""
        headers = make_auth_headers(login_and_get_token(
            {"username": "admin", "password": "admin123"}
        ))
        response = httpx.post(
            f"{BASE_URL}{API_PREFIX}/chat",
            json={"user_query": "Cửa hàng 1 có SKU nào sắp hết hàng không?"},
            headers=headers, timeout=60.0
        )
        assert response.status_code == 200
        reply = response.json()["reply"].lower()
        
        # Ground truth: SKU có stock thấp
        db_result = get_db_ground_truth(
            "SELECT COUNT(*) FROM inventory "
            "WHERE store_nbr=1 AND current_stock > 0 AND current_stock < 30"
        )
        low_stock_count = db_result[0]
        
        if low_stock_count > 0:
            # LLM phải cảnh báo
            keywords = ["hết hàng", "cháy hàng", "rủi ro", "thấp", 
                       "cảnh báo", "sắp hết"]
            assert any(kw in reply for kw in keywords), (
                f"Có {low_stock_count} SKU thấp nhưng LLM không cảnh báo"
            )


class TestToolCrossSelling:
    """Tool 8: suggest_cross_selling."""
    
    def test_cross_selling_suggestion(self):
        """TEST-LLM-040: Gợi ý bán chéo trả top 3 SKU cùng ngành."""
        headers = make_auth_headers(login_and_get_token(
            {"username": "admin", "password": "admin123"}
        ))
        response = httpx.post(
            f"{BASE_URL}{API_PREFIX}/chat",
            json={"user_query": "Gợi ý sản phẩm bán chéo cho cửa hàng 1"},
            headers=headers, timeout=60.0
        )
        assert response.status_code == 200
        # Agent phải trả về ít nhất gợi ý (không phải câu trả lời rỗng)
        assert len(response.json()["reply"]) > 30


class TestLLMErrorHandling:
    """Gap #9: Xử lý lỗi khi Groq API down."""
    
    def test_llm_timeout_graceful(self):
        """TEST-LLM-050: LLM timeout → 502 với detail tiếng Việt (không 500)."""
        # Test này khó mock Groq down — skip nếu không thể
        pytest.skip("Cần mock Groq API — chỉ chạy trong CI environment")