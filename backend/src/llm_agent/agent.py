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
    LLM_GROUNDING_MIN_MATCH, LLM_GROUNDING_RETRIES,
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
    """Gọi chat.completions.create với retry cho lỗi 429 rate limit TPM + log latency/usage."""
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        t0 = time.time()
        try:
            response = client.chat.completions.create(**kwargs)
            u = getattr(response, "usage", None)
            logger.info(f"LLM call {time.time() - t0:.1f}s | prompt={getattr(u, 'prompt_tokens', '?')} "
                        f"completion={getattr(u, 'completion_tokens', '?')} total={getattr(u, 'total_tokens', '?')}")
            return response
        except APIStatusError as e:
            if getattr(e, "status_code", None) == 429 and attempt < RATE_LIMIT_RETRIES:
                wait = _rate_limit_wait(e)
                if wait is not None and wait > _MAX_RETRY_WAIT_SECONDS:
                    logger.warning(f"Groq 429: cần chờ {wait:.0f}s (hết hạn mức ngày) - không retry.")
                    raise
                wait = wait or 10.0 * (attempt + 1)
                logger.warning(f"Groq 429 rate limit sau {time.time() - t0:.1f}s - thử lại sau {wait:.0f}s "
                               f"(lần {attempt + 1}/{RATE_LIMIT_RETRIES}).")
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


# ====================== CHỐNG BỊA SỐ LIỆU (grounding check) ======================
_NUMBER_RE = re.compile(r"-?\d[\d.,]*")

_GROUNDING_WARN_WITH_TOOLS = (
    "CẢNH BÁO CHỐNG BỊA: những số sau KHÔNG có trong kết quả tool: {unmatched}. "
    "Viết lại câu trả lời CHỈ dùng số liệu có trong kết quả tool ở trên (được phép tính "
    "chênh lệch/% từ số đã có nhưng ghi rõ cách tính). Số nào không có thì nói rõ là "
    "chưa có dữ liệu - KHÔNG được nêu số tự suy ra."
)
_GROUNDING_WARN_NO_TOOLS = (
    "Câu trả lời trước chứa số liệu nhưng bạn CHƯA gọi tool nào. Nếu câu hỏi cần dữ liệu "
    "cửa hàng/hàng hóa: hãy gọi tool phù hợp rồi mới trả lời. Nếu hệ thống không có dữ liệu "
    "đó: trả lời rõ là chưa có dữ liệu và gợi ý câu hỏi gần nhất được hỗ trợ. "
    "KHÔNG được nêu bất kỳ con số nào không đến từ kết quả tool."
)


def _number_candidates(raw: str) -> set:
    """Sinh mọi cách đọc hợp lệ của một token số (dấu . , kiểu VN và US đảo nhau)."""
    s = raw.strip().strip(".,")
    if not s or not any(c.isdigit() for c in s):
        return set()
    cands = set()
    try:  # đọc kiểu US: ',' là nghìn, '.' là thập phân (payload JSON)
        cands.add(float(s.replace(",", "")))
    except ValueError:
        pass
    try:  # đọc kiểu VN: '.' là nghìn, ',' là thập phân (câu trả lời tiếng Việt)
        cands.add(float(s.replace(".", "").replace(",", ".")))
    except ValueError:
        pass
    # abs: model nói "giảm 95,86%" trong khi payload lưu -95.86 là diễn giải hợp lệ,
    # không phải bịa - so sánh bỏ qua dấu.
    cands |= {abs(c) for c in cands}
    return cands


def _number_tokens(text: str) -> list:
    """Danh sách token số (chuỗi gốc) trong một đoạn văn."""
    return [raw.strip().strip(".,") for raw in _NUMBER_RE.findall(text or "")
            if any(c.isdigit() for c in raw)]


def _numbers_in_text(text: str) -> set:
    """Tập giá trị số (mọi cách đọc) trong một đoạn văn."""
    vals = set()
    for raw in _number_tokens(text):
        vals |= _number_candidates(raw)
    return vals


def _tool_payloads_text(messages: list) -> str:
    """Ghép nội dung toàn bộ message role='tool' trong hội thoại hiện tại."""
    return "\n".join(m.get("content", "") for m in messages
                     if isinstance(m, dict) and m.get("role") == "tool")


def _payload_variants(tool_payloads_text: str) -> set:
    """Tập giá trị có thể khớp từ payload JSON: gốc, abs, làm tròn int/1dp/2dp."""
    vals = set()
    for raw in _NUMBER_RE.findall(tool_payloads_text or ""):
        try:
            v = float(raw.replace(",", ""))  # payload là JSON chuẩn US
        except ValueError:
            continue
        for x in (v, abs(v)):
            vals.update({x, float(round(x)), round(x, 1), round(x, 2)})
    return vals


def _significant_tokens(text: str) -> list:
    """
    Token số đáng neo: loại bỏ số nguyên nhỏ (<10 ở mọi cách đọc) - đó chủ yếu là
    số đếm trong văn xuôi ("2 cửa hàng", "top 5"), không phải con số kinh doanh
    cần đối chiếu, giữ lại sẽ gây retry giả.
    """
    out = []
    for t in _number_tokens(text):
        cands = _number_candidates(t)
        if cands and max(cands) >= 10:
            out.append(t)
    return out


def _grounding_ratio(reply: str, tool_payloads_text: str) -> tuple:
    """
    Tỉ lệ TOKEN số đáng neo trong reply được 'neo' vào payload tool: một token được
    tính khớp nếu BẤT KỲ cách đọc nào của nó trùng payload (kèm abs/làm tròn).
    Số derived (VD: % model tự tính) được chấp nhận ở mức ngưỡng 0.6.
    Trả về (ratio, danh sách token không khớp). Reply có < 3 token -> (1.0, []).
    """
    tokens = _significant_tokens(reply)
    if len(tokens) < 3:
        return 1.0, []
    payload_vals = _payload_variants(tool_payloads_text)
    unmatched = [t for t in tokens if not (_number_candidates(t) & payload_vals)]
    ratio = (len(tokens) - len(unmatched)) / len(tokens)
    return ratio, unmatched


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
    "analyze_gross_margin",
    # benchmark_store_vs_peers: store_nbr bắt buộc nhưng peers cùng type phải lọc
    # theo phạm vi user -> vẫn cần _allowed_stores để không rò rỉ dữ liệu nhóm.
    "benchmark_store_vs_peers",
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
            max_tokens=512,  # output chọn tool rất nhỏ; Groq tính max_tokens vào hạn mức TPM
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
        _execute_tool_calls(messages, allowed_stores)
        return _synthesize_answer(messages)

    # LLM trả lời trực tiếp không dùng Tool - chặn đường bịa số (đường B):
    # reply chứa nhiều con số mà KHÔNG gọi tool nào -> bắt gọi tool hoặc thừa nhận thiếu dữ liệu.
    content = response_message.content or ""
    if LLM_GROUNDING_RETRIES > 0 and len(_significant_tokens(content)) >= 3:
        logger.warning("Reply chứa số liệu nhưng không gọi tool nào - retry chống bịa (đường B).")
        messages.append({"role": "system", "content": _GROUNDING_WARN_NO_TOOLS})
        try:
            retry_response = _chat_completion_with_retry(
                model=MODEL_NAME,
                messages=messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
                temperature=0.6,
                max_tokens=512,  # như call 1: hoặc chọn tool, hoặc trả lời ngắn
                **_extra_model_kwargs(),
            )
        except (APIStatusError, APIConnectionError) as e:
            logger.error(f"Groq API error (retry đường B): {e}")
            raise AgentError(_friendly_llm_error(e)) from e
        retry_message = retry_response.choices[0].message
        messages.append(retry_message)
        if retry_message.tool_calls:
            _execute_tool_calls(messages, allowed_stores)
            return _synthesize_answer(messages)
        return retry_message.content
    return content


def _execute_tool_calls(messages: list, allowed_stores: Optional[Set[int]]) -> None:
    """Thực thi toàn bộ tool_calls của message assistant cuối cùng, append kết quả role='tool'."""
    for tool_call in messages[-1].tool_calls:
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


def _synthesize_answer(messages: list) -> str:
    """
    Gọi LLM đọc kết quả Tool và trả lời tự nhiên, kèm kiểm tra grounding (đường A):
    reply chứa nhiều số mà phần lớn không có trong payload tool -> retry 1 lần
    ép model chỉ dùng số từ tool. Chọn giữa 2 bản theo tỉ lệ khớp cao hơn.
    """
    try:
        second_response = _chat_completion_with_retry(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.3,  # thấp hơn call 1: giảm 'sáng tạo' khi tổng hợp số liệu
            max_tokens=1200,  # chặn cứng độ dài câu trả lời (style ngắn gọn, đậm đặc)
            **_extra_model_kwargs(),
        )
    except (APIStatusError, APIConnectionError) as e:
        logger.error(f"Groq API error (lần 2): {e}")
        raise AgentError(_friendly_llm_error(e)) from e

    content = second_response.choices[0].message.content or ""
    if LLM_GROUNDING_RETRIES <= 0:
        return content

    ratio, unmatched = _grounding_ratio(content, _tool_payloads_text(messages))
    if ratio >= LLM_GROUNDING_MIN_MATCH:
        return content

    logger.warning(f"Grounding thấp ({ratio:.0%}, {len(unmatched)} số không khớp) - retry ép dùng số từ tool.")
    messages.append({"role": "system", "content": _GROUNDING_WARN_WITH_TOOLS.format(
        unmatched=unmatched[:10])})
    try:
        fixed_response = _chat_completion_with_retry(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.3,
            max_tokens=1200,
            **_extra_model_kwargs(),
        )
    except (APIStatusError, APIConnectionError) as e:
        logger.error(f"Groq API error (retry grounding): {e}")
        raise AgentError(_friendly_llm_error(e)) from e

    fixed = fixed_response.choices[0].message.content or ""
    fixed_ratio, _ = _grounding_ratio(fixed, _tool_payloads_text(messages))
    if fixed_ratio < ratio:
        logger.warning(f"Retry grounding không tốt hơn ({fixed_ratio:.0%} < {ratio:.0%}) - giữ câu trả lời đầu.")
        return content
    if fixed_ratio < LLM_GROUNDING_MIN_MATCH:
        logger.warning(f"Grounding vẫn thấp sau retry ({fixed_ratio:.0%}) - đã trả kết quả tốt nhất có thể.")
    return fixed
