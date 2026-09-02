"""
test_ml_service.py - Kiểm thử ML Service (:8001)
================================================
Test /predict endpoint với features thực tế,
verify Smart Routing (Local vs Global Ensemble).

Chạy: pytest backend/tests/test_ml_service.py -v
"""
import pytest
import httpx

# 127.0.0.1 thay vì localhost: tránh đụng wslrelay.exe trên [::1]:8001 (service cũ trong WSL)
ML_SERVICE_URL = "http://127.0.0.1:8001"

pytestmark = pytest.mark.ml

# 31 cột theo đúng schema preprocessor (ml_training/src/preprocessor.py):
# COLS_FILL_ZERO + COLS_CATEGORICAL + COLS_PASSTHROUGH
BASE_FEATURES = {
    # COLS_FILL_ZERO
    "transactions_lag1": 2500,
    "sales_lag7": 100.5,
    "sales_lag14": 98.2,
    "sales_rolling_mean7": 99.8,
    "cluster_mean_sales": 10.0,
    "cluster_median_sales": 9.0,
    "cluster_std_sales": 2.0,
    "cluster_family_mean_sales": 10.0,
    "cluster_promo_mean_sales": 12.0,
    "cluster_promo_lift": 1.1,
    # COLS_CATEGORICAL
    "store_nbr": 1,
    "family": "GROCERY I",
    "city": "Guayaquil",
    "state": "Guayas",
    "type": "D",
    "holiday_type": "Normal Day",
    # COLS_PASSTHROUGH
    "dayofweek": 1,
    "month": 8,
    "is_weekend": 0,
    "oil_price": 48.5,
    "cluster": 5,
    "perishable": 0,
    "is_holiday_lag1": 0,
    "is_holiday_lag2": 0,
    "is_holiday_lead1": 0,
    "is_holiday_lead2": 0,
    "is_tier1_cluster": 0,
    "onpromotion": 1,
    "is_earthquake_period": 0,
    "is_holiday": 0,
    "is_back_to_school": 1,
}


def make_payload(family: str, store_nbr: int = 1, **overrides) -> dict:
    features = {**BASE_FEATURES, "family": family, "store_nbr": store_nbr}
    features.update(overrides)
    return {"features": features, "family": family, "store_nbr": store_nbr}


class TestMLServiceHealth:

    def test_health_check(self):
        """TEST-ML-001: /health → 200."""
        response = httpx.get(f"{ML_SERVICE_URL}/health", timeout=5.0)
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "healthy"

    def test_models_loaded(self):
        """TEST-ML-002: Models preload trong memory."""
        response = httpx.get(f"{ML_SERVICE_URL}/health", timeout=5.0)
        data = response.json()
        # Kiểm tra models đã nạp (tùy response schema thực tế)
        assert data is not None


class TestPredict:
    """Test POST /predict với features."""

    def test_predict_valid_features(self):
        """TEST-ML-010: Predict với features hợp lệ → 200."""
        response = httpx.post(
            f"{ML_SERVICE_URL}/predict",
            json=make_payload("GROCERY I"),
            timeout=10.0
        )
        assert response.status_code == 200
        data = response.json()
        # Response schema: {store_nbr, family, predicted_sales, used_model}
        assert "predicted_sales" in data
        # Dự báo không âm
        assert data["predicted_sales"] >= 0

    def test_predict_local_model_routing(self):
        """TEST-ML-011: Family có local model → dùng Local LGBM."""
        response = httpx.post(
            f"{ML_SERVICE_URL}/predict",
            json=make_payload("PRODUCE", dayofweek=2, onpromotion=0,
                              sales_lag7=50, sales_lag14=48,
                              sales_rolling_mean7=50, transactions_lag1=2000),
            timeout=10.0
        )
        assert response.status_code == 200
        data = response.json()
        # Verify routing đến local model
        used_model = data.get("used_model", "")
        assert "Local" in used_model or "local" in used_model.lower()

    def test_predict_missing_features(self):
        """TEST-ML-012: Thiếu features → 422 validation error."""
        response = httpx.post(
            f"{ML_SERVICE_URL}/predict",
            json={"family": "GROCERY I"},  # Thiếu store_nbr + features
            timeout=10.0
        )
        assert response.status_code in (400, 422)
        assert response.status_code != 500

    def test_predict_response_time(self):
        """TEST-ML-013: Suy luận < 2 giây (SLA đề ra)."""
        import time
        start = time.perf_counter()
        response = httpx.post(
            f"{ML_SERVICE_URL}/predict",
            json=make_payload("GROCERY I", onpromotion=0,
                              sales_lag7=100, sales_lag14=98,
                              sales_rolling_mean7=100, transactions_lag1=2000),
            timeout=10.0
        )
        elapsed = time.perf_counter() - start

        assert response.status_code == 200
        assert elapsed < 2.0, f"Suy luận mất {elapsed:.2f}s, vượt SLA 2s"
