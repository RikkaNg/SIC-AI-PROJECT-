# backend/src/llm_agent/config.py
import os
import logging
from groq import Groq

logger = logging.getLogger(__name__)

# Lấy API key từ biến môi trường
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY chưa được set trong biến môi trường!")

# Khởi tạo Groq Client
client = Groq(api_key=GROQ_API_KEY)

# Model của Groq hỗ trợ Function Calling tốt nhất
MODEL_NAME = "llama-3.3-70b-versatile"