# backend/src/llm_agent/config.py
import os
import logging
from groq import Groq

logger = logging.getLogger(__name__)

# Lấy API key từ biến môi trường
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY chưa được set trong biến môi trường!")

# Khởi tạo Groq Client - chỉ khi có API key để tránh crash toàn bộ app lúc import
# (các API không liên quan LLM vẫn phải chạy được khi thiếu key)
client = None
if GROQ_API_KEY:
    client = Groq(api_key=GROQ_API_KEY)
else:
    logger.warning("GROQ_API_KEY chưa được set - /api/chat sẽ trả về lỗi hướng dẫn thay vì làm sập service.")

# Model Qwen 3 trên Groq (hỗ trợ Function Calling).
# Lưu ý: "qwen/qwen3-32b" đã bị Groq ngừng phục vụ (404 model_not_found) ->
# mặc định dùng bản kế nhiệm "qwen/qwen3.6-27b". Có thể override qua biến môi trường.
MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "qwen/qwen3.6-27b")

# (Tùy chọn) Tắt/giảm thinking của Qwen 3 để phản hồi nhanh hơn, VD: "none".
# Để trống -> không gửi tham số này lên Groq.
LLM_REASONING_EFFORT = os.environ.get("LLM_REASONING_EFFORT", "")
