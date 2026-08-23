# backend/src/llm_agent/agent.py
import functools
import json
import logging
from typing import Optional, Set

from groq import APIStatusError, APIConnectionError

from backend.src.security import validate_tool_access
from .config import client, MODEL_NAME, LLM_REASONING_EFFORT
from .prompts import SYSTEM_PROMPT_SUPPLY_CHAIN as SYSTEM_PROMPT
from .tools import (
    GROQ_TOOL_DEFINITIONS as TOOLS_SCHEMA,
    AVAILABLE_TOOLS as AVAILABLE_FUNCTIONS,
)

logger = logging.getLogger(__name__)


class AgentError(RuntimeError):
    """Lỗi điều phối agent - được chat_routes chuyển thành HTTP 502 kèm message rõ ràng."""


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
        return "GROQ_API_KEY không hợp lệ hoặc đã bị thu hồi. Vui lòng kiểm tra lại .env."
    if status_code == 429:
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


def _bind_allowed_stores(function_name: str, function: callable,
                         allowed_stores: Optional[Set[int]]) -> callable:
    """
    Row-Level Isolation: với tool truy vấn toàn hệ thống khi thiếu store_nbr
    (get_sales_summary), chèn tham số ẩn _allowed_stores để filter WHERE IN.
    Các tool khác có store_nbr bắt buộc đã được validate_tool_access chặn trước.
    """
    if allowed_stores is not None and function_name == "get_sales_summary":
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

    if chat_history:
        messages.extend(chat_history)

    messages.append({"role": "user", "content": user_query})

    # Bước 1: Gửi câu hỏi cho LLM kèm theo danh sách Tools
    try:
        response = client.chat.completions.create(
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
                    result_str = json.dumps(result_payload, ensure_ascii=False, default=str)
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
            second_response = client.chat.completions.create(
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
