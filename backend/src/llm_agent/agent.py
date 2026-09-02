# backend/src/llm_agent/agent.py
import functools
import json
import logging
import re
import time
from typing import Optional, Set

from groq import APIStatusError, APIConnectionError

from backend.src.security import validate_tool_access
from .config import (
    client, MODEL_NAME, LLM_REASONING_EFFORT,
    LLM_HISTORY_TOKEN_BUDGET, LLM_HISTORY_MAX_MESSAGES, LLM_HISTORY_MAX_CHARS_PER_MSG,
)
from .prompts import SYSTEM_PROMPT_SUPPLY_CHAIN as SYSTEM_PROMPT
from .tools import (
    GROQ_TOOL_DEFINITIONS as TOOLS_SCHEMA,
    AVAILABLE_TOOLS as AVAILABLE_FUNCTIONS,
)

logger = logging.getLogger(__name__)

# Groq gói free giới hạn TPM thấp: gọi 2 lần liên tiếp (call 1 + call sau tool
# result) dễ chạm limit -> tự thử lại thay vì trả lỗi cho người dùng.
RATE_LIMIT_RETRIES = 3


class AgentError(RuntimeError):
    """Lỗi điều phối agent - được chat_routes chuyển thành HTTP 502 kèm message rõ ràng."""


def _sanitize_history(chat_history) -> list:
    """
    Làm sạch lịch sử client gửi lên - backend KHÔNG tin tuyệt đối input:
    - Chỉ chấp nhận role 'user'/'assistant' với content là string: chặn client
      chèn role 'system'/'tool' giả vào messages (prompt injection).
    - Cắt mỗi tin nhắn theo LLM_HISTORY_MAX_CHARS_PER_MSG, giới hạn tổng số tin.
    """
    if not chat_history or not isinstance(chat_history, list):
        return []
    cleaned = []
    for m in chat_history[: LLM_HISTORY_MAX_MESSAGES * 2]:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        content = content.strip()[:LLM_HISTORY_MAX_CHARS_PER_MSG]
        if content:
            cleaned.append({"role": role, "content": content})
    # Giữ tối đa LLM_HISTORY_MAX_MESSAGES tin MỚI NHẤT
    return cleaned[-LLM_HISTORY_MAX_MESSAGES:]


def _estimate_tokens(text: str) -> int:
    """
    Ước lượng token bảo thủ (~2.5 ký tự/token): tiếng Việt có dấu thường tốn
    token hơn tiếng Anh, ước lượng thấp giúp luôn nằm dưới ngân sách thật.
    """
    return int(len(text) / 2.5) + 1


def _trim_history_to_budget(history: list, budget_tokens: int) -> list:
    """
    Giữ các lượt trao đổi MỚI NHẤT sao cho tổng ước lượng token <= ngân sách.
    Duyệt từ mới -> cũ để ưu tiên ngữ cảnh gần, rồi đảo lại thứ tự gốc.
    """
    if budget_tokens <= 0:
        return []
    kept, used = [], 0
    for m in reversed(history):
        cost = _estimate_tokens(m["content"])
        if kept and used + cost > budget_tokens:
            break  # giữ nguyên cặp hỏi-đáp: dừng trước khi cắt giữa chừng
        if not kept and cost > budget_tokens:
            m = {"role": m["role"], "content": m["content"][: int(budget_tokens * 2.5)]}
            cost = _estimate_tokens(m["content"])
        kept.append(m)
        used += cost
    kept.reverse()
    if len(kept) != len(history):
        logger.info(f"History trim: giữ {len(kept)}/{len(history)} tin (~{used} tokens / budget {budget_tokens}).")
    return kept


def _rate_limit_wait(e: APIStatusError) -> Optional[float]:
    """
    Đọc số giây Groq khuyên chờ từ message 429. Groq dùng 2 dạng:
    'Please try again in 27.9975s' (phút - TPM) và 'in 14m58s' (ngày - TPD).
    """
    m = re.search(r"try again in ([\d.]+)s", str(e))
    if m:
        return float(m.group(1))
    m = re.search(r"try again in (\d+)m([\d.]+)s", str(e))
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    return None


# Không retry khi Groq yêu cầu chờ quá ngưỡng này (hết hạn mức NGÀY - TPD):
# chờ 15+ phút trong 1 request là vô lý, trả lỗi rõ ràng cho người dùng ngay.
_MAX_RETRY_WAIT_SECONDS = 120


def _chat_completion_with_retry(**kwargs):
    """Gọi chat.completions.create với retry cho lỗi 429 rate limit TPM."""
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except APIStatusError as e:
            if getattr(e, "status_code", None) == 429 and attempt < RATE_LIMIT_RETRIES:
                wait = _rate_limit_wait(e)
                if wait is not None and wait > _MAX_RETRY_WAIT_SECONDS:
                    logger.warning(f"Groq 429: cần chờ {wait:.0f}s (hết hạn mức ngày) - không retry.")
                    raise
                wait = wait or 10.0 * (attempt + 1)
                logger.warning(f"Groq 429 rate limit - thử lại sau {wait:.0f}s (lần {attempt + 1}/{RATE_LIMIT_RETRIES}).")
                time.sleep(wait)
                continue
            raise


def _friendly_llm_error(e: Exception) -> str:
    """Chuyển lỗi SDK Groq thành thông điệp tiếng Việt dễ hiểu cho người dùng cuối."""
    status_code = getattr(e, "status_code", None)
    body = getattr(e, "body", None) or {}
    api_msg = ""
    if isinstance(body, dict):
        api_msg = str(body.get("error", {}).get("message", "") if isinstance(body.get("error"), dict) else body.get("error", ""))
    if status_code == 404 or "does not exist" in api_msg or "model_not_found" in api_msg:
        return (f"Model '{MODEL_NAME}' không tồn tại trên Groq hoặc tài khoản không có quyền. "
                f"Vui lòng đổi LLM_MODEL_NAME trong .env (VD: qwen/qwen3.6-27b) rồi khởi động lại backend. "
                f"Chi tiết: {api_msg or e}")
    if status_code == 401:
        return ("API key không hợp lệ hoặc đã bị thu hồi. Vui lòng kiểm tra lại GROQ_API_KEY "
                "và GROQ_BASE_URL (nếu dùng endpoint trung gian như OrcaRouter) trong .env.")
    if status_code == 429:
        if "per day" in api_msg or "TPD" in api_msg:
            return ("Groq đã hết hạn mức token NGÀY (gói free 200K token/ngày). "
                    "Vui lòng thử lại sau hoặc nâng cấp gói Groq.")
        return "Groq đang giới hạn tốc độ (rate limit). Vui lòng thử lại sau ít giây."
    if isinstance(e, APIConnectionError):
        return "Không kết nối được tới Groq Cloud. Kiểm tra mạng của server backend."
    if isinstance(e, APIStatusError):
        return f"Lỗi từ Groq Cloud (HTTP {status_code}): {api_msg or e}"
    return f"Lỗi không xác định khi gọi LLM: {e}"


def _extra_model_kwargs() -> dict:
    """
    Tham số bổ sung cho model Qwen 3 trên Groq.
    - LLM_REASONING_EFFORT được set (VD: "none") -> tắt thinking cho phản hồi nhanh.
    - Không set -> không gửi gì (an toàn với mọi model).
    """
    if LLM_REASONING_EFFORT:
        return {"extra_body": {"reasoning_effort": LLM_REASONING_EFFORT}}
    return {}


# Tool có store_nbr TÙY CHỌN + tham số ẩn _allowed_stores: khi user không chỉ định
# cửa hàng, agent chèn scope RLS để filter WHERE IN (các tool có store_nbr bắt buộc
# đã được validate_tool_access kiểm tra trước rồi).
TOOLS_WITH_STORE_SCOPE = frozenset({
    "get_sales_summary",
    "get_monthly_revenue",
    "get_top_selling_items",
    "get_item_profile",
    "get_store_profile",
    "get_store_traffic",
})


def _bind_allowed_stores(function_name: str, function: callable,
                         allowed_stores: Optional[Set[int]]) -> callable:
    """
    Row-Level Isolation: với tool truy vấn toàn hệ thống khi thiếu store_nbr,
    chèn tham số ẩn _allowed_stores để filter WHERE IN.
    """
    if allowed_stores is not None and function_name in TOOLS_WITH_STORE_SCOPE:
        return functools.partial(function, _allowed_stores=frozenset(allowed_stores))
    return function


def run_agent(user_query: str, chat_history: list = None,
              allowed_stores: Optional[Set[int]] = None) -> str:
    """
    Điều phối luồng hội thoại giữa User, LLM và Python Tools.

    allowed_stores: phạm vi cửa hàng của user hiện tại (None = toàn hệ thống).
    Mọi tool call vi phạm phạm vi sẽ bị chặn và LLM diễn giải lời từ chối.
    """
    if client is None:
        return ("⚠️ GROQ_API_KEY chưa được cấu hình trên server. "
                "Vui lòng đặt biến môi trường GROQ_API_KEY (xem .env.example) rồi khởi động lại backend.")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    # Ngữ cảnh hội thoại: làm sạch (chặn role giả) rồi trim theo ngân sách token
    history = _trim_history_to_budget(
        _sanitize_history(chat_history), LLM_HISTORY_TOKEN_BUDGET)
    messages.extend(history)

    messages.append({"role": "user", "content": user_query})

    # Bước 1: Gửi câu hỏi cho LLM kèm theo danh sách Tools
    try:
        response = _chat_completion_with_retry(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
            temperature=0.6,
            max_tokens=2048,
            **_extra_model_kwargs(),
        )
    except (APIStatusError, APIConnectionError) as e:
        logger.error(f"Groq API error (lần 1): {e}")
        raise AgentError(_friendly_llm_error(e)) from e

    response_message = response.choices[0].message
    messages.append(response_message)

    # Bước 2: Kiểm tra xem LLM có muốn gọi Tool không
    if response_message.tool_calls:
        logger.info(f"LLM yêu cầu gọi {len(response_message.tool_calls)} tool(s).")

        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            try:
                function_args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError as e:
                logger.warning(f"Tool '{function_name}' trả args không phải JSON hợp lệ: {tool_call.function.arguments!r}")
                result_str = json.dumps({"error": f"Tham số tool không hợp lệ: {e}"}, ensure_ascii=False)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": result_str,
                })
                continue

            logger.info(f"Thực thi: {function_name} | Args: {function_args}")

            # Row-Level Isolation: chặn tool nếu vượt phạm vi cửa hàng của user
            forbidden = validate_tool_access(function_name, function_args, allowed_stores)
            if forbidden is not None:
                result_str = json.dumps(forbidden, ensure_ascii=False)
                logger.warning(f"Chặn tool '{function_name}' - vượt phạm vi cửa hàng của user.")
            elif function_name in AVAILABLE_FUNCTIONS:
                try:
                    fn = _bind_allowed_stores(function_name, AVAILABLE_FUNCTIONS[function_name], allowed_stores)
                    result_payload = fn(**function_args)
                    # allow_nan=False: kết quả chứa NaN/Infinity sẽ sinh JSON không hợp lệ
                    # khiến Groq trả 400 -> lỗi 502 cho user. Đưa về error dict sạch.
                    result_str = json.dumps(result_payload, ensure_ascii=False, default=str, allow_nan=False)
                    logger.info(f"Kết quả Tool: {result_str}")
                except Exception as e:
                    result_str = json.dumps({"error": f"Lỗi thực thi tool: {str(e)}"})
            else:
                result_str = json.dumps({"error": "Tool không tồn tại."})

            # Đưa kết quả Tool về lại cho LLM
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": result_str,
            })

        # Bước 3: Gọi LLM lần 2 để nó đọc kết quả Tool và trả lời tự nhiên
        try:
            second_response = _chat_completion_with_retry(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.6,
                **_extra_model_kwargs(),
            )
        except (APIStatusError, APIConnectionError) as e:
            logger.error(f"Groq API error (lần 2): {e}")
            raise AgentError(_friendly_llm_error(e)) from e

        return second_response.choices[0].message.content

    else:
        # Nếu LLM trả lời trực tiếp không dùng Tool
        return response_message.content
