# backend/src/llm_agent/config.py
import os
import logging
from groq import Groq

logger = logging.getLogger(__name__)

# Lấy API key từ biến môi trường
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY chưa được set trong biến môi trường!")

# Base URL tùy chọn cho nhà cung cấp OpenAI-compatible khác Groq (VD: OrcaRouter).
# Để trống -> dùng Groq Cloud chính thức (https://api.groq.com/openai/v1).
GROQ_BASE_URL = os.environ.get("GROQ_BASE_URL", "").strip()

# Khởi tạo Groq Client - chỉ khi có API key để tránh crash toàn bộ app lúc import
# (các API không liên quan LLM vẫn phải chạy được khi thiếu key)
client = None
if GROQ_API_KEY:
    if not GROQ_BASE_URL:
        # SDK Groq tự đọc os.environ["GROQ_BASE_URL"] khi không được truyền base_url;
        # chuỗi rỗng (từ .env/compose) sẽ đè mặc định và làm hỏng kết nối -> phải gỡ bỏ.
        os.environ.pop("GROQ_BASE_URL", None)
        client = Groq(api_key=GROQ_API_KEY)
    else:
        client = Groq(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
        logger.info(f"LLM client dùng endpoint tùy chỉnh: {GROQ_BASE_URL}")
else:
    logger.warning("GROQ_API_KEY chưa được set - /api/chat sẽ trả về lỗi hướng dẫn thay vì làm sập service.")

# Model Qwen 3 trên Groq (hỗ trợ Function Calling).
# Lưu ý: "qwen/qwen3-32b" đã bị Groq ngừng phục vụ (404 model_not_found) ->
# mặc định dùng bản kế nhiệm "qwen/qwen3.6-27b". Có thể override qua biến môi trường.
MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "qwen/qwen3.6-27b")

# (Tùy chọn) Tắt/giảm thinking của Qwen 3 để phản hồi nhanh hơn, VD: "none".
# Để trống -> không gửi tham số này lên Groq.
LLM_REASONING_EFFORT = os.environ.get("LLM_REASONING_EFFORT", "")

# ---- Ngân sách ngữ cảnh hội thoại (chat history) ----
# Lịch sử chỉ được giữ trong giới hạn token để không chạm rate-limit TPM của
# Groq (gói free: 8.000 token/phút) với request đã nặng ~5-6K token
# (system prompt + schema 18 tools). Backend luôn là nơi cắt cuối cùng.
LLM_HISTORY_TOKEN_BUDGET = max(0, int(os.environ.get("LLM_HISTORY_TOKEN_BUDGET", "1200")))
LLM_HISTORY_MAX_MESSAGES = max(1, int(os.environ.get("LLM_HISTORY_MAX_MESSAGES", "40")))
# Chặn 1 tin nhắn dài bất tận nhồi vào lịch sử.
LLM_HISTORY_MAX_CHARS_PER_MSG = max(100, int(os.environ.get("LLM_HISTORY_MAX_CHARS_PER_MSG", "4000")))
