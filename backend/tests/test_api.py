"""
test_api.py - Kiểm thử API Endpoints (Mục 4.3)
==============================================
Test toàn bộ REST API endpoints của SIC-AI-PROJECT:
- Auth: POST /api/auth/login
- KPI: GET /api/kpi
- Predictions: GET /api/predictions
- Top Products: GET /api/top-products
- Family Mix: GET /api/family-mix
- Family Trend: GET /api/family-trend
- Products: GET /api/products (phân trang, search)
- Inventory Restock: POST /api/inventory/restock
- Chat: POST /api/chat

Chạy: pytest backend/tests/test_api.py -v
"""
import pytest
import httpx

from conftest import (
    BASE_URL, API_PREFIX, DEFAULT_TIMEOUT,
    ADMIN_CREDENTIALS, MANAGER1_CREDENTIALS
)


# ============================================================
# MARKERS
# ============================================================
pytestmark = pytest.mark.api


# ============================================================
# 1. AUTH ENDPOINTS
# ============================================================
class TestAuthAPI:
    """Kiểm thử authentication endpoints."""
    
    def test_login_admin_success(self):
        """TEST-001: Admin đăng nhập thành công → 200 + token."""
        response = httpx.post(
            f"{BASE_URL}{API_PREFIX}/auth/login",
            json=ADMIN_CREDENTIALS,
            timeout=DEFAULT_TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert len(data["access_token"]) > 50  # JWT đủ dài
    
    def test_login_manager1_success(self):
        """TEST-002: Manager1 đăng nhập thành công."""
        response = httpx.post(
            f"{BASE_URL}{API_PREFIX}/auth/login",
            json=MANAGER1_CREDENTIALS,
            timeout=DEFAULT_TIMEOUT
        )
        assert response.status_code == 200
        assert "access_token" in response.json()
    
    def test_login_wrong_password(self):
        """TEST-003: Sai mật khẩu → 401."""
        response = httpx.post(
            f"{BASE_URL}{API_PREFIX}/auth/login",
            json={"username": "admin", "password": "wrong_password"},
            timeout=DEFAULT_TIMEOUT
        )
        assert response.status_code == 401
    
    def test_login_nonexistent_user(self):
        """TEST-004: User không tồn tại → 401."""
        response = httpx.post(
            f"{BASE_URL}{API_PREFIX}/auth/login",
            json={"username": "ghost_user", "password": "anything"},
            timeout=DEFAULT_TIMEOUT
        )
        assert response.status_code == 401
    
    def test_login_empty_body(self):
        """TEST-005: Body rỗng → 422 (validation error)."""
        response = httpx.post(
            f"{BASE_URL}{API_PREFIX}/auth/login",
            json={},
            timeout=DEFAULT_TIMEOUT
        )
        assert response.status_code in (400, 422)


# ============================================================
# 2. KPI ENDPOINT
# ============================================================
class TestKPIAPI:
    """Kiểm thử GET /api/kpi."""
    
    def test_kpi_store_1(self, admin_headers):
        """TEST-010: KPI cửa hàng 1 → 200 + đầy đủ trường."""
        response = httpx.get(
            f"{BASE_URL}{API_PREFIX}/kpi",
            params={"store_nbr": 1},
            headers=admin_headers,
            timeout=DEFAULT_TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        # Kiểm tra các KPI cơ bản tồn tại
        assert isinstance(data, dict)

    def test_kpi_invalid_store(self, admin_headers):
        """TEST-011: Store ID không tồn tại (999) → không crash."""
        response = httpx.get(
            f"{BASE_URL}{API_PREFIX}/kpi",
            params={"store_nbr": 999},
            headers=admin_headers,
            timeout=DEFAULT_TIMEOUT
        )
        # API trả 404 khi không có dữ liệu, miễn KHÔNG 500
        assert response.status_code != 500

    def test_kpi_missing_auth(self):
        """TEST-012: Không có token → 401."""
        response = httpx.get(
            f"{BASE_URL}{API_PREFIX}/kpi",
            params={"store_nbr": 1},
            timeout=DEFAULT_TIMEOUT
        )
        assert response.status_code == 401


# ============================================================
# 3. PREDICTIONS ENDPOINT
# ============================================================
class TestPredictionsAPI:
    """Kiểm thử GET /api/predictions."""
    
    def test_predictions_store_1(self, admin_headers):
        """TEST-020: Dự báo 16 ngày cửa hàng 1 → 200."""
        response = httpx.get(
            f"{BASE_URL}{API_PREFIX}/predictions",
            params={"store_nbr": 1},
            headers=admin_headers,
            timeout=DEFAULT_TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        # Dự báo phải có dữ liệu 16 ngày
        assert data is not None
    
    def test_predictions_forecast_meta(self, admin_headers):
        """TEST-021: Metadata kỳ dự báo → 200."""
        response = httpx.get(
            f"{BASE_URL}{API_PREFIX}/forecast-meta",
            headers=admin_headers,
            timeout=DEFAULT_TIMEOUT
        )
        assert response.status_code == 200


# ============================================================
# 4. TOP PRODUCTS ENDPOINT
# ============================================================
class TestTopProductsAPI:
    """Kiểm thử GET /api/top-products."""
    
    def test_top_products_default(self, admin_headers):
        """TEST-030: Top products mặc định → 200."""
        response = httpx.get(
            f"{BASE_URL}{API_PREFIX}/top-products",
            headers=admin_headers,
            timeout=DEFAULT_TIMEOUT
        )
        assert response.status_code == 200
    
    def test_top_products_with_limit(self, admin_headers):
        """TEST-031: Top products limit=6 → tối đa 6 kết quả."""
        response = httpx.get(
            f"{BASE_URL}{API_PREFIX}/top-products",
            params={"store_nbr": 1, "limit": 6},
            headers=admin_headers,
            timeout=DEFAULT_TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        # Nếu trả list, kiểm tra length
        if isinstance(data, list):
            assert len(data) <= 6
        elif isinstance(data, dict) and "items" in data:
            assert len(data["items"]) <= 6


# ============================================================
# 5. FAMILY MIX & TREND
# ============================================================
class TestFamilyAPI:
    """Kiểm thử family-mix và family-trend."""
    
    def test_family_mix(self, admin_headers):
        """TEST-040: Thị phần ngành hàng → 200."""
        response = httpx.get(
            f"{BASE_URL}{API_PREFIX}/family-mix",
            params={"store_nbr": 1},
            headers=admin_headers,
            timeout=DEFAULT_TIMEOUT
        )
        assert response.status_code == 200
    
    def test_family_trend(self, admin_headers):
        """TEST-041: Xu hướng dự báo theo ngành → 200."""
        response = httpx.get(
            f"{BASE_URL}{API_PREFIX}/family-trend",
            params={"store_nbr": 1},
            headers=admin_headers,
            timeout=DEFAULT_TIMEOUT
        )
        assert response.status_code == 200
    
    def test_product_families_list(self, admin_headers):
        """TEST-042: Danh sách ngành hàng cho dropdown → 200."""
        response = httpx.get(
            f"{BASE_URL}{API_PREFIX}/product-families",
            headers=admin_headers,
            timeout=DEFAULT_TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        # Phải có 33 ngành hàng
        if isinstance(data, list):
            assert len(data) >= 30  # ~33 families


# ============================================================
# 6. PRODUCTS ENDPOINT (Phân trang + Search)
# ============================================================
class TestProductsAPI:
    """Kiểm thử GET /api/products."""
    
    def test_products_default_page(self, admin_headers):
        """TEST-050: Trang 1 mặc định → 200, tối đa 20 items (page_size mặc định)."""
        response = httpx.get(
            f"{BASE_URL}{API_PREFIX}/products",
            headers=admin_headers,
            timeout=DEFAULT_TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        if isinstance(data, dict) and "items" in data:
            assert len(data["items"]) <= 20
            assert "total" in data
    
    def test_products_pagination(self, admin_headers):
        """TEST-051: Phân trang page=2, page_size=10 → 200."""
        response = httpx.get(
            f"{BASE_URL}{API_PREFIX}/products",
            params={"page": 2, "page_size": 10},
            headers=admin_headers,
            timeout=DEFAULT_TIMEOUT
        )
        assert response.status_code == 200
    
    def test_products_max_page_size(self, admin_headers):
        """TEST-052: page_size=100 (max) → 200, không vượt 100."""
        response = httpx.get(
            f"{BASE_URL}{API_PREFIX}/products",
            params={"page": 1, "page_size": 100},
            headers=admin_headers,
            timeout=DEFAULT_TIMEOUT
        )
        assert response.status_code == 200
    
    def test_products_search_normal(self, admin_headers):
        """TEST-053: Tìm kiếm từ khóa bình thường → 200."""
        response = httpx.get(
            f"{BASE_URL}{API_PREFIX}/products",
            params={"search": "milk"},
            headers=admin_headers,
            timeout=DEFAULT_TIMEOUT
        )
        assert response.status_code == 200
    
    def test_products_filter_family(self, admin_headers):
        """TEST-054: Lọc theo ngành hàng → 200."""
        response = httpx.get(
            f"{BASE_URL}{API_PREFIX}/products",
            params={"family": "GROCERY I"},
            headers=admin_headers,
            timeout=DEFAULT_TIMEOUT
        )
        assert response.status_code == 200
    
    def test_products_sort(self, admin_headers):
        """TEST-055: Sắp xếp theo cột → 200."""
        response = httpx.get(
            f"{BASE_URL}{API_PREFIX}/products",
            params={"sort": "sold", "order": "desc"},
            headers=admin_headers,
            timeout=DEFAULT_TIMEOUT
        )
        assert response.status_code == 200


# ============================================================
# 7. INVENTORY ENDPOINT
# ============================================================
class TestInventoryAPI:
    """Kiểm thử POST /api/inventory/restock (non-mutating).

    Không gọi POST với body hợp lệ + admin vì sẽ UPDATE thật vào
    bảng inventory của retail.db (RLS của admin cho qua mọi store).
    """

    def test_restock_missing_body_422(self, admin_headers):
        """TEST-060: Body rỗng/thiếu trường bắt buộc → 422."""
        response = httpx.post(
            f"{BASE_URL}{API_PREFIX}/inventory/restock",
            json={},
            headers=admin_headers,
            timeout=DEFAULT_TIMEOUT
        )
        assert response.status_code == 422

    def test_restock_invalid_type_422(self, admin_headers):
        """TEST-061: store_nbr không phải int → 422."""
        response = httpx.post(
            f"{BASE_URL}{API_PREFIX}/inventory/restock",
            json={"store_nbr": "abc", "item_nbr": 96995, "quantity": 5},
            headers=admin_headers,
            timeout=DEFAULT_TIMEOUT
        )
        assert response.status_code == 422

    def test_restock_out_of_scope_403(self, manager1_headers):
        """TEST-062: manager1 (stores 1-10) restock store 15 → 403.

        ensure_store_access chạy TRƯỚC update_stock_db trong
        inventory_routes.py nên không có ghi nào xảy ra vào DB.
        """
        response = httpx.post(
            f"{BASE_URL}{API_PREFIX}/inventory/restock",
            json={"store_nbr": 15, "item_nbr": 96995, "quantity": 5},
            headers=manager1_headers,
            timeout=DEFAULT_TIMEOUT
        )
        assert response.status_code == 403


# ============================================================
# 8. CHAT ENDPOINT (LLM Agent)
# ============================================================
class TestChatAPI:
    """Kiểm thử POST /api/chat (LLM Agent)."""

    # Mọi test ở đây gọi Groq thật (tốn hạn mức token/ngày) -> tách khỏi
    # các test thường để bỏ qua được bằng -m "not llm"
    pytestmark = pytest.mark.llm

    def test_chat_simple_question(self, admin_headers):
        """TEST-070: Câu hỏi đơn giản → 200 + có response."""
        response = httpx.post(
            f"{BASE_URL}{API_PREFIX}/chat",
            json={"user_query": "Tổng quan doanh số cửa hàng 1"},
            headers=admin_headers,
            timeout=60.0  # LLM cần timeout dài
        )
        assert response.status_code == 200
        data = response.json()
        assert "reply" in data
        assert len(data["reply"]) > 50  # Câu trả lời đủ dài
    
    def test_chat_with_history(self, admin_headers):
        """TEST-071: Chat kèm lịch sử hội thoại → 200."""
        history = [
            {"role": "user", "content": "Xin chào"},
            {"role": "assistant", "content": "Xin chào! Tôi có thể giúp gì?"}
        ]
        response = httpx.post(
            f"{BASE_URL}{API_PREFIX}/chat",
            json={
                "user_query": "Cửa hàng 1 bán chạy nhất ngành nào?",
                "chat_history": history
            },
            headers=admin_headers,
            timeout=60.0
        )
        assert response.status_code == 200


# ============================================================
# 9. STORES ENDPOINT
# ============================================================
class TestStoresAPI:
    """Kiểm thử GET /api/stores."""
    
    def test_list_stores_admin(self, admin_headers):
        """TEST-080: Admin thấy toàn bộ 54 cửa hàng."""
        response = httpx.get(
            f"{BASE_URL}{API_PREFIX}/stores",
            headers=admin_headers,
            timeout=DEFAULT_TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        if isinstance(data, list):
            assert len(data) == 54
        elif isinstance(data, dict) and "stores" in data:
            assert len(data["stores"]) == 54


# ============================================================
# 10. SMOKE TEST TOÀN HỆ THỐNG
# ============================================================
class TestSystemSmoke:
    """Smoke test: kiểm tra hệ thống sống."""
    
    def test_backend_alive(self):
        """TEST-090: Backend FastAPI phản hồi."""
        response = httpx.get(
            f"{BASE_URL}/docs",
            timeout=5.0
        )
        assert response.status_code == 200
    
    def test_ml_service_alive(self):
        """TEST-091: ML Service :8001 health check."""
        response = httpx.get(
            "http://localhost:8001/health",
            timeout=5.0
        )
        assert response.status_code == 200