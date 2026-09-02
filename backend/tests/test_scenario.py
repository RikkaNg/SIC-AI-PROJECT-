"""
test_scenario.py - Kiểm thử Scenario Lab (kịch bản what-if)
============================================================
1. Unit test analyze_scenario: toán phân tích đúng trên series giả (không cần server).
2. API test: auth, RLS, validation của /api/scenario/run + /api/scenario/meta.

Chạy: pytest backend/tests/test_scenario.py -v
"""
import sys
from pathlib import Path

import pytest
import httpx

# Cho phép import backend.* khi pytest được chạy từ thư mục gốc
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from conftest import (
    BASE_URL, API_PREFIX, DEFAULT_TIMEOUT,
    MANAGER1_CREDENTIALS, login_and_get_token,
)
from backend.src.services.scenario_service import analyze_scenario

pytestmark = pytest.mark.api


# ============================================================
# 1. UNIT TEST analyze_scenario (không cần server/DB)
# ============================================================
def _series(values):
    return [{"date": f"2017-08-{16 + i:02d}", "predicted_sales": v} for i, v in enumerate(values)]


class TestAnalyzeScenario:
    """Phân tích deterministic trên dữ liệu giả."""

    def test_stockout_when_demand_exceeds_stock(self):
        """Kịch bản tăng nhu cầu vượt tồn -> đứt hàng, thiếu đúng bằng chênh lệch."""
        result = analyze_scenario(
            "GROCERY I", _series([10.0] * 16), _series([20.0] * 16),
            stock=100.0, lead_time=5.0, unit_price=2.0, horizon_days=16,
        )
        kpi = result["kpi"]
        assert kpi["baseline_total"] == 160.0
        assert kpi["scenario_total"] == 320.0
        assert kpi["delta_pct"] == 100.0
        assert kpi["will_stockout"] is True
        assert kpi["shortfall"] == 220.0  # 320 - 100
        assert kpi["lost_revenue_usd"] == 440.0  # 220 * 2.0
        assert "SẮP ĐỨT HÀNG" in result["analysis"]
        assert "nhập tối thiểu 220.0" in result["recommendation"]

    def test_overstock_when_demand_drops(self):
        """Kịch bản giảm mạnh nhu cầu -> dư tồn khi phủ quá 30 ngày."""
        result = analyze_scenario(
            "GROCERY I", _series([10.0] * 16), _series([1.0] * 16),
            stock=1000.0, lead_time=3.0, unit_price=1.0, horizon_days=16,
        )
        kpi = result["kpi"]
        assert kpi["will_stockout"] is False
        assert kpi["overstock"] is True  # 1000 / (16/16 = 1 đơn vị/ngày) = 1000 ngày > 30
        assert kpi["excess"] == 984.0
        assert "DƯ TỒN" in result["analysis"]
        assert "khuyến mãi" in result["recommendation"].lower()

    def test_adequate_when_within_stock(self):
        """Nhu cầu tăng nhẹ vẫn nằm trong tồn kho -> đủ hàng, không đề xuất nhập gấp."""
        result = analyze_scenario(
            "GROCERY I", _series([10.0] * 16), _series([12.0] * 16),
            stock=300.0, lead_time=4.0, unit_price=2.0, horizon_days=16,
        )
        kpi = result["kpi"]
        assert kpi["will_stockout"] is False
        assert kpi["overstock"] is False
        assert kpi["verdict"] == "ĐỦ HÀNG"
        assert kpi["shortfall"] == 0.0

    def test_zero_baseline_delta_pct_is_none(self):
        """Baseline = 0 (SKU mới/ngành trống) -> delta_pct None, không chia 0."""
        result = analyze_scenario(
            "GROCERY I", _series([0.0] * 16), _series([5.0] * 16),
            stock=200.0, lead_time=2.0, unit_price=1.0, horizon_days=16,
        )
        assert result["kpi"]["delta_pct"] is None
        assert result["kpi"]["will_stockout"] is False

    def test_risk_levels_by_shortfall_ratio(self):
        """Mức rủi ro theo tỷ lệ thiếu/tồn (scenario=320): >2 Rất cao, >1 Cao, >0.3 Trung bình."""
        for stock, expected in [(50.0, "Rất cao"), (150.0, "Cao"), (200.0, "Trung bình")]:
            result = analyze_scenario(
                "GROCERY I", _series([10.0] * 16), _series([20.0] * 16),
                stock=stock, lead_time=3.0, unit_price=1.0, horizon_days=16,
            )
            assert result["kpi"]["risk_level"] == expected, f"stock={stock}"

    def test_analysis_is_json_safe(self):
        """Kết quả phải serialize được JSON chuẩn (không NaN/Infinity)."""
        import json
        result = analyze_scenario(
            "GROCERY I", _series([0.0] * 16), _series([0.0] * 16),
            stock=0.0, lead_time=0.0, unit_price=1.0, horizon_days=16,
        )
        json.dumps(result, allow_nan=False)  # raise nếu có NaN/Inf


# ============================================================
# 2. API TEST (cần backend :8000 đang chạy)
# ============================================================
class TestScenarioAPI:
    """Auth/RLS/validation cho /api/scenario/*."""

    def test_meta_missing_auth_401(self):
        response = httpx.get(
            f"{BASE_URL}{API_PREFIX}/scenario/meta",
            params={"store_nbr": 1, "family": "GROCERY I"},
            timeout=DEFAULT_TIMEOUT,
        )
        assert response.status_code == 401

    def test_run_missing_auth_401(self):
        response = httpx.post(
            f"{BASE_URL}{API_PREFIX}/scenario/run",
            json={"store_nbr": 1, "family": "GROCERY I"},
            timeout=DEFAULT_TIMEOUT,
        )
        assert response.status_code == 401

    def test_run_out_of_scope_store_403(self):
        """manager1 (stores 1-10) chạy kịch bản cho store 15 -> 403, không tốn ML."""
        token = httpx.post(
            f"{BASE_URL}{API_PREFIX}/auth/login",
            json=MANAGER1_CREDENTIALS, timeout=DEFAULT_TIMEOUT,
        ).json()["access_token"]
        response = httpx.post(
            f"{BASE_URL}{API_PREFIX}/scenario/run",
            json={"store_nbr": 15, "family": "GROCERY I"},
            headers={"Authorization": f"Bearer {token}"},
            timeout=DEFAULT_TIMEOUT,
        )
        assert response.status_code == 403

    def test_run_invalid_body_422(self):
        """Thiếu store_nbr/family hoặc sai pattern event_type -> 422."""
        token = login_and_get_token({"username": "admin", "password": "admin123"})
        response = httpx.post(
            f"{BASE_URL}{API_PREFIX}/scenario/run",
            json={"family": "GROCERY I"},
            headers={"Authorization": f"Bearer {token}"},
            timeout=DEFAULT_TIMEOUT,
        )
        assert response.status_code == 422

    def test_run_multiplier_out_of_range_422(self):
        """demand_multiplier > 10 -> 422 (chặn trước khi gọi ML)."""
        token = httpx.post(
            f"{BASE_URL}{API_PREFIX}/auth/login",
            json={"username": "admin", "password": "admin123"}, timeout=DEFAULT_TIMEOUT,
        ).json()["access_token"]
        response = httpx.post(
            f"{BASE_URL}{API_PREFIX}/scenario/run",
            json={"store_nbr": 1, "family": "GROCERY I", "demand_multiplier": 99.0},
            headers={"Authorization": f"Bearer {token}"},
            timeout=DEFAULT_TIMEOUT,
        )
        assert response.status_code == 422

    def test_run_unknown_family_404(self):
        """Family không tồn tại -> 404 (không crash 500)."""
        token = httpx.post(
            f"{BASE_URL}{API_PREFIX}/auth/login",
            json={"username": "admin", "password": "admin123"}, timeout=DEFAULT_TIMEOUT,
        ).json()["access_token"]
        response = httpx.post(
            f"{BASE_URL}{API_PREFIX}/scenario/run",
            json={"store_nbr": 1, "family": "KHONG_TON_TAI_XYZ"},
            headers={"Authorization": f"Bearer {token}"},
            timeout=60.0,
        )
        assert response.status_code == 404
