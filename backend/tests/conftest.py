"""
conftest.py - Shared fixtures cho toàn bộ test suite
SIC-AI-PROJECT Backend Tests
"""
import pytest
import httpx
import os

# ============================================================
# CẤU HÌNH
# ============================================================
# Dùng 127.0.0.1 thay vì localhost: trên máy này wslrelay.exe chiếm [::1]:8000/8001
# (service cũ trong WSL), nên resolve "localhost" sang IPv6 sẽ trúng nhầm service đó.
BASE_URL = os.getenv("TEST_BASE_URL", "http://127.0.0.1:8000")
API_PREFIX = "/api"

# Tài khoản test (seed từ init_auth.py)
ADMIN_CREDENTIALS = {"username": "admin", "password": "admin123"}
MANAGER1_CREDENTIALS = {"username": "manager1", "password": "manager123"}
MANAGER2_CREDENTIALS = {"username": "manager2", "password": "manager123"}

# Timeout cho các request
DEFAULT_TIMEOUT = 10.0
LLM_TIMEOUT = 60.0  # LLM chat cần timeout dài hơn


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def login_and_get_token(credentials: dict) -> str:
    """
    Đăng nhập và trả về JWT access token.
    
    Args:
        credentials: dict {"username": ..., "password": ...}
    
    Returns:
        str: JWT access token
    
    Raises:
        AssertionError: Nếu đăng nhập thất bại
    """
    response = httpx.post(
        f"{BASE_URL}{API_PREFIX}/auth/login",
        json=credentials,
        timeout=DEFAULT_TIMEOUT
    )
    assert response.status_code == 200, (
        f"Login failed for {credentials['username']}: "
        f"status={response.status_code}, body={response.text}"
    )
    data = response.json()
    assert "access_token" in data, f"Response thiếu access_token: {data}"
    return data["access_token"]


def make_auth_headers(token: str) -> dict:
    """Tạo headers Authorization Bearer từ token."""
    return {"Authorization": f"Bearer {token}"}


# ============================================================
# FIXTURES
# ============================================================
@pytest.fixture(scope="session")
def admin_token() -> str:
    """JWT token cho admin (toàn bộ 54 cửa hàng)."""
    return login_and_get_token(ADMIN_CREDENTIALS)


@pytest.fixture(scope="session")
def manager1_token() -> str:
    """JWT token cho manager1 (cửa hàng 1-10)."""
    return login_and_get_token(MANAGER1_CREDENTIALS)


@pytest.fixture(scope="session")
def manager2_token() -> str:
    """JWT token cho manager2 (cửa hàng 11-20)."""
    return login_and_get_token(MANAGER2_CREDENTIALS)


@pytest.fixture(scope="session")
def admin_headers(admin_token) -> dict:
    """Auth headers cho admin."""
    return make_auth_headers(admin_token)


@pytest.fixture(scope="session")
def manager1_headers(manager1_token) -> dict:
    """Auth headers cho manager1."""
    return make_auth_headers(manager1_token)


@pytest.fixture(scope="session")
def manager2_headers(manager2_token) -> dict:
    """Auth headers cho manager2."""
    return make_auth_headers(manager2_token)


@pytest.fixture(scope="session")
def http_client() -> httpx.Client:
    """
    HTTP client dùng chung cho cả session test.
    Tự động đóng sau khi test xong.
    """
    with httpx.Client(
        base_url=BASE_URL,
        timeout=DEFAULT_TIMEOUT
    ) as client:
        yield client


@pytest.fixture(scope="session")
def llm_client() -> httpx.Client:
    """HTTP client riêng cho LLM chat (timeout dài hơn)."""
    with httpx.Client(
        base_url=BASE_URL,
        timeout=LLM_TIMEOUT
    ) as client:
        yield client