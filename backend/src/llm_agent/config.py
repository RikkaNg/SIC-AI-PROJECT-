# backend/src/llm_agent/config.py
import os
import logging
from groq import Groq
from dotenv import load_dotenv

load_dotenv() # Tự động đọc file .env ở thư mục gốc
logger = logging.getLogger(__name__)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY chưa được set!")

client = Groq(api_key=GROQ_API_KEY)
MODEL_NAME = "llama-3.1-8b-instant"