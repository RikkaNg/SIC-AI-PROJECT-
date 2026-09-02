"""
test_edge_cases.py - Kiểm thử biên (Edge Cases)
===============================================
Test các input bất thường: số âm, 0, chuỗi, null, giá trị cực đại.

Chạy: pytest backend/tests/test_edge_cases.py -v
"""
import pytest
import httpx

from conftest import BASE_URL, API_PREFIX, DEFAULT_TIMEOUT

pytestmark = pytest.mark.edge


class TestEdgeInputs:
    
    @pytest.mark.parametrize("store_id", [0, -1, -100, 99999, 1.5, "abc", "", None])
    def test_kpi_invalid_store_ids(self, admin_headers, store_id):
        """TEST-EDGE-001..008: Store ID bất thường → không 500."""
        response = httpx.get(
            f"{BASE_URL}{API_PREFIX}/kpi",
            params={"store_nbr": store_id},
            headers=admin_headers,
            timeout=DEFAULT_TIMEOUT
        )
        assert response.status_code != 500, (
            f"Server crash với store_id={store_id!r}"
        )
        assert response.status_code in (200, 400, 404, 422)
    
    @pytest.mark.parametrize("page,page_size", [
        (0, 15),           # Page 0
        (-1, 15),          # Page âm
        (99999, 15),       # Page cực lớn
        (1, 0),            # PageSize 0
        (1, -5),           # PageSize âm
        (1, 10001),        # PageSize vượt max
    ])
    def test_products_edge_pagination(self, admin_headers, page, page_size):
        """TEST-EDGE-020..025: Pagination bất thường → không crash."""
        response = httpx.get(
            f"{BASE_URL}{API_PREFIX}/products",
            params={"page": page, "page_size": page_size},
            headers=admin_headers,
            timeout=DEFAULT_TIMEOUT
        )
        assert response.status_code != 500
    
    def test_search_empty_string(self, admin_headers):
        """TEST-EDGE-030: Search rỗng → trả tất cả."""
        response = httpx.get(
            f"{BASE_URL}{API_PREFIX}/products",
            params={"search": ""},
            headers=admin_headers,
            timeout=DEFAULT_TIMEOUT
        )
        assert response.status_code == 200
    
    def test_search_very_long(self, admin_headers):
        """TEST-EDGE-031: Search 10.000 ký tự → không crash."""
        response = httpx.get(
            f"{BASE_URL}{API_PREFIX}/products",
            params={"search": "a" * 10000},
            headers=admin_headers,
            timeout=DEFAULT_TIMEOUT
        )
        assert response.status_code != 500
    
    def test_search_special_unicode(self, admin_headers):
        """TEST-EDGE-032: Unicode/emoji trong search."""
        response = httpx.get(
            f"{BASE_URL}{API_PREFIX}/products",
            params={"search": "🚀🎉 Vietnamese: tiếng Việt"},
            headers=admin_headers,
            timeout=DEFAULT_TIMEOUT
        )
        assert response.status_code == 200
    
    def test_chat_empty_message(self, admin_headers):
        """TEST-EDGE-040: Chat message rỗng → 422 hoặc xử lý an toàn."""
        response = httpx.post(
            f"{BASE_URL}{API_PREFIX}/chat",
            json={"user_query": ""},
            headers=admin_headers,
            timeout=60.0
        )
        assert response.status_code != 500
    
    def test_chat_very_long_message(self, admin_headers):
        """TEST-EDGE-041: Message 50.000 ký tự → không crash."""
        response = httpx.post(
            f"{BASE_URL}{API_PREFIX}/chat",
            json={"user_query": "test " * 10000},
            headers=admin_headers,
            timeout=60.0
        )
        assert response.status_code != 500


class TestConcurrentAccess:
    """Gap #5: Truy cập đồng thời."""
    
    def test_concurrent_reads(self, admin_headers):
        """TEST-EDGE-050: 20 request đồng thời → tất cả 200 (WAL mode)."""
        import concurrent.futures
        
        def make_request(_):
            return httpx.get(
                f"{BASE_URL}{API_PREFIX}/products",
                params={"page": 1, "page_size": 15},
                headers=admin_headers,
                timeout=DEFAULT_TIMEOUT
            )
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            responses = list(executor.map(make_request, range(20)))
        
        for r in responses:
            assert r.status_code == 200, (
                f"Concurrent read failed: {r.status_code}"
            )
    
    def test_concurrent_mixed_roles(self, admin_headers, manager1_headers):
        """TEST-EDGE-051: Admin + manager1 đồng thời → isolation vẫn đúng."""
        import concurrent.futures
        
        def admin_request(_):
            return httpx.get(
                f"{BASE_URL}{API_PREFIX}/kpi",
                params={"store_nbr": 50},
                headers=admin_headers,
                timeout=DEFAULT_TIMEOUT
            ).status_code
        
        def manager1_request(_):
            return httpx.get(
                f"{BASE_URL}{API_PREFIX}/kpi",
                params={"store_nbr": 50},
                headers=manager1_headers,
                timeout=DEFAULT_TIMEOUT
            ).status_code
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
            admin_codes = list(ex.map(admin_request, range(5)))
            manager_codes = list(ex.map(manager1_request, range(5)))
        
        assert all(c == 200 for c in admin_codes)
        assert all(c == 403 for c in manager_codes), (
            "Isolation bị phá dưới concurrent load!"
        )