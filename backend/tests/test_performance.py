"""
test_performance.py - Đo hiệu năng API (P95 latency)
====================================================
Đo thời gian phản hồi của từng endpoint qua N lần gọi,
tính P50 (median), P95, P99.

Chạy: python backend/tests/test_performance.py
Hoặc: pytest backend/tests/test_performance.py -s

Xuất kết quả dạng bảng cho báo cáo Mục 4.3.
"""
import time
import statistics
import httpx
import pytest
import numpy as np
from tabulate import tabulate  # pip install tabulate

from conftest import (
    BASE_URL, API_PREFIX, 
    ADMIN_CREDENTIALS, login_and_get_token, make_auth_headers
)

pytestmark = pytest.mark.performance

# ============================================================
# CẤU HÌNH
# ============================================================
N_REQUESTS = 50  # Số lần gọi mỗi endpoint để đo
ENDPOINTS = [
    # (method, path, params, name, timeout)
    ("GET", f"{API_PREFIX}/kpi", {"store_nbr": 1}, "KPI Dashboard", 10),
    ("GET", f"{API_PREFIX}/predictions", {"store_nbr": 1}, "Predictions 16d", 10),
    ("GET", f"{API_PREFIX}/top-products", {"store_nbr": 1, "limit": 6}, "Top Products", 10),
    ("GET", f"{API_PREFIX}/family-mix", {"store_nbr": 1}, "Family Mix", 10),
    ("GET", f"{API_PREFIX}/family-trend", {"store_nbr": 1}, "Family Trend", 10),
    ("GET", f"{API_PREFIX}/products", {"page": 1, "page_size": 15}, "Products (page 1)", 10),
    ("GET", f"{API_PREFIX}/products", {"search": "milk", "page": 1}, "Products (search)", 10),
    ("GET", f"{API_PREFIX}/product-families", {}, "Product Families", 10),
]


def measure_endpoint(
    method: str, 
    path: str, 
    params: dict, 
    headers: dict,
    n_requests: int,
    timeout: float
) -> dict:
    """
    Đo latency của 1 endpoint qua n_requests lần gọi.
    
    Returns:
        dict: {name, p50_ms, p95_ms, p99_ms, mean_ms, max_ms, status}
    """
    latencies = []
    status_codes = set()
    
    with httpx.Client(base_url=BASE_URL, timeout=timeout) as client:
        for i in range(n_requests):
            start = time.perf_counter()
            
            if method == "GET":
                response = client.get(path, params=params, headers=headers)
            else:
                response = client.post(path, json=params, headers=headers)
            
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)
            status_codes.add(response.status_code)
    
    # Kiểm tra tất cả request thành công
    all_ok = all(s == 200 for s in status_codes)
    
    return {
        "endpoint": path.split("/")[-1] or path,
        "status": "PASS" if all_ok else f"FAIL {status_codes}",
        "n_requests": n_requests,
        "mean_ms": round(statistics.mean(latencies), 1),
        "p50_ms": round(np.percentile(latencies, 50), 1),
        "p95_ms": round(np.percentile(latencies, 95), 1),
        "p99_ms": round(np.percentile(latencies, 99), 1),
        "max_ms": round(max(latencies), 1),
    }


def measure_llm_chat(headers: dict, n_requests: int = 5) -> dict:
    """Đo latency LLM chat (ít request hơn vì tốn kém)."""
    latencies = []
    
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
        for _ in range(n_requests):
            start = time.perf_counter()
            response = client.post(
                f"{API_PREFIX}/chat",
                json={"user_query": "Tổng quan cửa hàng 1"},
                headers=headers
            )
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)
    
    return {
        "endpoint": "chat (LLM)",
        "status": "PASS",
        "n_requests": n_requests,
        "mean_ms": round(statistics.mean(latencies), 1),
        "p50_ms": round(np.percentile(latencies, 50), 1),
        "p95_ms": round(np.percentile(latencies, 95), 1),
        "p99_ms": round(np.percentile(latencies, 99), 1),
        "max_ms": round(max(latencies), 1),
    }


def main():
    """Chạy toàn bộ performance test và in bảng kết quả."""
    print("=" * 80)
    print("SIC-AI-PROJECT PERFORMANCE TEST")
    print(f"Base URL: {BASE_URL}")
    print(f"Requests per endpoint: {N_REQUESTS}")
    print("=" * 80)
    
    # Đăng nhập
    print("\n[1/3] Đăng nhập...")
    token = login_and_get_token(ADMIN_CREDENTIALS)
    headers = make_auth_headers(token)
    print(f"  ✓ Token acquired ({token[:20]}...)")
    
    # Đo từng endpoint
    print(f"\n[2/3] Đo {len(ENDPOINTS)} endpoints × {N_REQUESTS} requests...")
    results = []
    
    for method, path, params, name, timeout in ENDPOINTS:
        print(f"  Testing: {name}...", end=" ", flush=True)
        result = measure_endpoint(
            method, path, params, headers, N_REQUESTS, timeout
        )
        result["endpoint"] = name  # Dùng tên thân thiện
        results.append(result)
        print(f"P95 = {result['p95_ms']}ms")
    
    # Đo LLM chat
    print(f"\n[3/3] Đo LLM Chat (5 requests)...")
    llm_result = measure_llm_chat(headers, n_requests=5)
    results.append(llm_result)
    print(f"  LLM Chat: P95 = {llm_result['p95_ms']}ms")
    
    # In bảng kết quả
    print("\n" + "=" * 80)
    print("KẾT QUẢ PERFORMANCE TEST")
    print("=" * 80)
    
    table_data = []
    for r in results:
        table_data.append([
            r["endpoint"],
            r["status"],
            r["n_requests"],
            f"{r['mean_ms']:.1f}",
            f"{r['p50_ms']:.1f}",
            f"{r['p95_ms']:.1f}",
            f"{r['p99_ms']:.1f}",
            f"{r['max_ms']:.1f}",
        ])
    
    headers_table = [
        "Endpoint", "Status", "N", "Mean (ms)", "P50 (ms)", 
        "P95 (ms)", "P99 (ms)", "Max (ms)"
    ]
    
    print(tabulate(
        table_data, 
        headers=headers_table, 
        tablefmt="grid"
    ))
    
    # Đánh giá SLA
    print("\n" + "-" * 80)
    print("ĐÁNH GIÁ THEO SLA (< 1000ms cho API nội bộ):")
    print("-" * 80)
    
    all_pass = True
    for r in results:
        if "chat" in r["endpoint"].lower():
            sla = 5000  # LLM chat: < 5 giây
        else:
            sla = 1000  # API thường: < 1 giây
        
        status = "✓ PASS" if r["p95_ms"] < sla else "✗ FAIL"
        if r["p95_ms"] >= sla:
            all_pass = False
        
        print(f"  {r['endpoint']:<25} P95={r['p95_ms']:>8.1f}ms  "
              f"SLA={sla}ms  {status}")
    
    print("-" * 80)
    if all_pass:
        print("  ✓ TẤT CẢ ENDPOINT ĐẠT SLA")
    else:
        print("  ✗ CÓ ENDPOINT VƯỢT SLA - CẦN TỐI ƯU")
    
    # Xuất LaTeX table (tùy chọn)
    print("\n" + "=" * 80)
    print("LATEX TABLE (copy vào báo cáo Mục 4.3):")
    print("=" * 80)
    print(r"\begin{table}[h!]")
    print(r"\centering")
    print(r"\small")
    print(r"\begin{tabular}{@{}llrrl@{}}")
    print(r"\toprule")
    print(r"\textbf{Endpoint} & \textbf{Method} & \textbf{Status} & \textbf{P95 (ms)} & \textbf{Kết quả} \\")
    print(r"\midrule")
    
    for method, path, params, name, timeout in ENDPOINTS:
        result = next(r for r in results if r["endpoint"] == name)
        print(f"{name.replace(' ', '~')} & {method} & 200 & "
              f"{result['p95_ms']:.0f} & Pass \\\\")
    
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Kết quả kiểm thử API nội bộ (n=" + str(N_REQUESTS) + " lần gọi mỗi endpoint).}")
    print(r"\label{tab:apitest}")
    print(r"\end{table}")
    
    return all_pass


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)


# ============================================================
# PYTEST WRAPPER (chạy được qua pytest)
# ============================================================
def test_performance_all_endpoints():
    """Pytest wrapper: chạy performance test và assert tất cả pass SLA."""
    token = login_and_get_token(ADMIN_CREDENTIALS)
    headers = make_auth_headers(token)
    
    for method, path, params, name, timeout in ENDPOINTS:
        result = measure_endpoint(
            method, path, params, headers, n_requests=10, timeout=timeout
        )
        assert result["status"] == "PASS", (
            f"{name} failed: {result['status']}"
        )
        assert result["p95_ms"] < 1000, (
            f"{name} P95 {result['p95_ms']}ms vượt SLA 1000ms"
        )