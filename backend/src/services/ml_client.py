# backend/src/services/ml_client.py
"""
Client gọi ML Service với connection reuse + retry + TTL cache (§4.1, §4.2).

- Singleton httpx.AsyncClient: khởi tạo 1 lần trong lifespan của backend
  (backend/src/main.py), tái sử dụng connection pool cho mọi request thay vì
  tạo client mới từng lần gọi.
- HTTPTransport(retries=2): tự retry lỗi kết nối tạm thời; vòng retry ở trên
  xử lý thêm 429/5xx với backoff ngắn.
- TTL cache: dự báo cho cùng một (store, family, features) trong ngày là
  bất biến -> cache kết quả thành công để cắt bớt lời gọi lặp lại.
"""
import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)


class MLServiceError(Exception):
    """Lỗi gọi ML Service - forecast_routes bắt lỗi này để fallback về bảng precomputed."""

# URL của ml_service (chạy ở port 8001)
ML_SERVICE_URL = os.environ.get("ML_SERVICE_URL", "http://localhost:8001")

# Timeout mỗi request (giây) - khớp giá trị 15s trước đây
ML_TIMEOUT_SECONDS = float(os.environ.get("ML_TIMEOUT_SECONDS", "15"))

# TTL cache dự báo (giây). Dự báo trong ngày là bất biến nên 1 giờ là an toàn;
# sẽ bị làm mới tự nhiên khi pipeline retrain tạo dữ liệu mới.
ML_PREDICT_TTL_SECONDS = int(os.environ.get("ML_PREDICT_TTL_SECONDS", "3600"))

# Giới hạn số entry cache để không phình RAM (evict cũ nhất khi vượt)
ML_CACHE_MAX_ENTRIES = 1024

# Số lần thử tối đa cho lỗi 429/5xx (kèm backoff ngắn giữa các lần)
ML_MAX_ATTEMPTS = 3
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_RETRY_BACKOFF_SECONDS = 0.5

_client: Optional[httpx.AsyncClient] = None

# TTL cache in-process (pattern giống product_routes.py:62-76)
_TTL_CACHE: Dict[Any, Any] = {}


def init_ml_client() -> None:
    """Khởi tạo singleton client - gọi 1 lần trong lifespan của backend."""
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            base_url=ML_SERVICE_URL,
            timeout=ML_TIMEOUT_SECONDS,
            # AsyncHTTPTransport (không phải HTTPTransport sync) - client.aclose()
            # gọi transport.aclose() lúc shutdown.
            transport=httpx.AsyncHTTPTransport(
                retries=2,
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            ),
        )
        logger.info(f"ML client ready -> {ML_SERVICE_URL}")


async def close_ml_client() -> None:
    """Đóng connection pool khi backend shutdown."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
        logger.info("ML client closed.")


def _cache_key(store_nbr: int, family: str, features: Dict[str, Any]) -> tuple:
    """Key bất biến cho cache: (store, family, features đã chuẩn hóa thứ tự key)."""
    canonical = json.dumps(features, sort_keys=True, default=str)
    return (int(store_nbr), str(family), canonical)


def _ttl_get(key: tuple) -> Optional[Any]:
    hit = _TTL_CACHE.get(key)
    if hit and time.monotonic() - hit[0] < ML_PREDICT_TTL_SECONDS:
        return hit[1]
    return None


def _ttl_put(key: tuple, value: Any) -> None:
    if len(_TTL_CACHE) >= ML_CACHE_MAX_ENTRIES:
        oldest_key = min(_TTL_CACHE, key=lambda k: _TTL_CACHE[k][0])
        _TTL_CACHE.pop(oldest_key, None)
    _TTL_CACHE[key] = (time.monotonic(), value)


def clear_ml_cache() -> None:
    """Xóa cache (hữu ích sau khi pipeline retrain chạy xong)."""
    _TTL_CACHE.clear()


async def get_ml_forecast(store_nbr: int, family: str, features: Dict[str, Any]):
    """
    Gọi sang ML Service để lấy dự báo doanh số cho 1 sản phẩm (family) tại 1 cửa hàng.
    Khớp với SinglePredictRequest của ML Service. Kết quả thành công được cache TTL.

    Raises:
        MLServiceError: khi ml_service không phản hồi hoặc trả lỗi sau khi đã retry.
    """
    key = _cache_key(store_nbr, family, features)
    hit = _ttl_get(key)
    if hit is not None:
        return hit

    if _client is None:
        # Phòng khi route được gọi ngoài lifespan (VD test chạy thẳng hàm)
        init_ml_client()

    # Cấu trúc payload gửi đi phải khớp 100% với main.py bên ml_service
    payload = {
        "store_nbr": store_nbr,
        "family": family,
        "features": features
    }

    last_error: Optional[Exception] = None
    for attempt in range(1, ML_MAX_ATTEMPTS + 1):
        try:
            response = await _client.post("/predict", json=payload)
            response.raise_for_status()  # Báo lỗi nếu status code là 4xx hoặc 5xx
            data = response.json()
            _ttl_put(key, data)
            return data

        except httpx.HTTPStatusError as e:
            last_error = e
            if e.response.status_code in _RETRYABLE_STATUS and attempt < ML_MAX_ATTEMPTS:
                logger.warning(
                    "ML Service trả %s (lần %s/%s) - retry sau %.1fs",
                    e.response.status_code, attempt, ML_MAX_ATTEMPTS, _RETRY_BACKOFF_SECONDS,
                )
                await asyncio.sleep(_RETRY_BACKOFF_SECONDS * attempt)
                continue
            logger.error(f"Lỗi từ ML Service: {e.response.text}")
            raise MLServiceError(
                f"ML Service trả về lỗi: {e.response.status_code} - {e.response.text}"
            ) from e

        except httpx.RequestError as e:
            # Lỗi kết nối - transport đã retry 2 lần phía HTTP, thêm vòng chờ dài hơn
            last_error = e
            if attempt < ML_MAX_ATTEMPTS:
                logger.warning(
                    "Không kết nối được ML Service (lần %s/%s) - retry sau %.1fs",
                    attempt, ML_MAX_ATTEMPTS, _RETRY_BACKOFF_SECONDS,
                )
                await asyncio.sleep(_RETRY_BACKOFF_SECONDS * attempt)
                continue
            logger.error(f"Lỗi kết nối tới ML Service: {e}")
            raise MLServiceError(
                "Không thể kết nối tới ML Service. Có thể service chưa chạy hoặc bị sập."
            ) from e

    raise MLServiceError(f"ML Service không phản hồi sau {ML_MAX_ATTEMPTS} lần thử: {last_error}")
