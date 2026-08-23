# backend/src/routes/chat_routes.py
import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.src.llm_agent.agent import AgentError, run_agent
from backend.src.security import Identity, get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()

class ChatRequest(BaseModel):
    user_query: str
    chat_history: list = []

@router.post("/chat")
def chat_endpoint(req: ChatRequest,
                  user: Identity = Depends(get_current_user)):
    """
    Row-Level Isolation: toàn bộ tool của Agent chỉ được truy cập
    các cửa hàng trong phạm vi của user (admin/ERP = toàn hệ thống).

    Lỗi LLM/Gateway được trả về 502 kèm `detail` rõ ràng để frontend hiển thị,
    thay vì 500 "Internal Server Error" trống không có thông tin.
    """
    try:
        answer = run_agent(
            user_query=req.user_query,
            chat_history=req.chat_history,
            allowed_stores=user.allowed_stores,
        )
    except AgentError as e:
        logger.error(f"AgentError: {e}")
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        logger.exception("Lỗi không xử lý được trong /api/chat")
        raise HTTPException(status_code=502, detail=f"Lỗi xử lý AI Agent: {e}")
    return {"reply": answer}
