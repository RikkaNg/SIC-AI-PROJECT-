# backend/src/routes/chat_routes.py
from fastapi import APIRouter
from pydantic import BaseModel
from backend.src.llm_agent.agent import run_agent

router = APIRouter()

class ChatRequest(BaseModel):
    user_query: str
    chat_history: list = []

@router.post("/chat")
def chat_endpoint(req: ChatRequest):
    answer = run_agent(user_query=req.user_query, chat_history=req.chat_history)
    return {"reply": answer}