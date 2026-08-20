# backend/src/llm_agent/agent.py
import json
import logging
from typing import Optional 
from .config import client, MODEL_NAME  
from .prompts import SYSTEM_PROMPT
from .tools import TOOLS_SCHEMA, AVAILABLE_FUNCTIONS
from groq.types.chat import ChatCompletionMessageParam

logger = logging.getLogger(__name__)

def run_agent(user_query: str, chat_history: Optional[list] = None) -> str:
    """
    Điều phối luồng hội thoại giữa User, LLM và Python Tools.
    """
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    
    if chat_history:
        messages.extend(chat_history)
        
    messages.append({"role": "user", "content": user_query})

    # Bước 1: Gửi câu hỏi cho LLM kèm theo danh sách Tools
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=TOOLS_SCHEMA, # type: ignore
        tool_choice="auto",
        temperature=0.6,
        max_tokens=2048
    )

    response_message = response.choices[0].message
    messages.append(response_message) # type: ignore

    # Bước 2: Kiểm tra xem LLM có muốn gọi Tool không
    if response_message.tool_calls:
        logger.info(f"LLM yêu cầu gọi {len(response_message.tool_calls)} tool(s).")
        
        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            logger.info(f"Thực thi: {function_name} | Args: {function_args}")
            
            if function_name in AVAILABLE_FUNCTIONS:
                try:
                    result_str = AVAILABLE_FUNCTIONS[function_name](**function_args)
                    logger.info(f"Kết quả Tool: {result_str}")
                except Exception as e:
                    result_str = json.dumps({"error": f"Lỗi thực thi tool: {str(e)}"})
            else:
                result_str = json.dumps({"error": "Tool không tồn tại."})
            
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": result_str,
            }) # type: ignore
        
        # Bước 3: Gọi LLM lần 2 để nó đọc kết quả Tool và trả lời tự nhiên
        second_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.6
        )
        
        return second_response.choices[0].message.content # type: ignore

    else:
        # Nếu LLM trả lời trực tiếp không dùng Tool
        return response_message.content # type: ignore