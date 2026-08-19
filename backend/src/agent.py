"""
llm_agent.py
Lớp điều phối LLM Agent sử dụng Groq API (Llama-3.3-70B).
Kết nối LLM với các Python Tools (Function Calling) để truy vấn Database.
"""

import os
import json
import logging
from groq import Groq
# Import toàn bộ các hàm Python Tools và Schema mà ta đã viết
from backend.src.llm_agent.tools import AVAILABLE_TOOLS, TOOLS_SCHEMA

logger = logging.getLogger(__name__)

# Khởi tạo Groq Client (Cần set GROQ_API_KEY trong biến môi trường)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Model của Groq hỗ trợ Function Calling tốt nhất hiện nay
MODEL_NAME = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """
Bạn là một Trợ lý AI chuyên gia về Quản lý Chuỗi cung ứng và Bán lẻ.
Nhiệm vụ của bạn là giúp các Quản lý cửa hàng kiểm tra tồn kho, dự báo doanh số và đưa ra chiến lược kinh doanh.
Khi người dùng hỏi về: Tồn kho, cháy hàng, doanh số dự báo, hay so sánh các kỳ, hãy bắt buộc sử dụng các Tools (công cụ) được cung cấp để lấy dữ liệu thực tế từ Database.
KHÔNG bao giờ tự bịa ra số liệu. Nếu Tool trả về lỗi hoặc không có dữ liệu, hãy xin lỗi người dùng và nói là hiện không có dữ liệu.
Trả lời ngắn gọn, súc tích bằng tiếng Việt, trình bày số liệu rõ ràng.
"""

def run_agent(user_query: str, chat_history: list = None) -> str:
    """
    Chạy luồng Agent: User Query -> LLM -> Tool Call -> Tool Result -> LLM -> Final Answer.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    
    # Thêm lịch sử chat nếu có
    if chat_history:
        messages.extend(chat_history)
        
    messages.append({"role": "user", "content": user_query})

    # Bước 1: Gọi API lần 1 để xem LLM có muốn dùng Tool không
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=TOOLS_SCHEMA,
        tool_choice="auto",
        temperature=0.6,
        max_tokens=2048,
    )

    response_message = response.choices[0].message
    
    # Thêm câu trả lời của LLM vào messages để duy trì ngữ cảnh
    messages.append(response_message)

    # Bước 2: Kiểm tra xem LLM có yêu cầu gọi Tool không
    if response_message.tool_calls:
        logger.info(f"LLM yêu cầu gọi {len(response_message.tool_calls)} tool(s).")
        
        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            logger.info(f"Đang thực thi Tool: {function_name} | Args: {function_args}")
            
            # Tìm và thực thi hàm Python tương ứng trong AVAILABLE_TOOLS
            if function_name in AVAILABLE_TOOLS:
                try:
                    # Gọi hàm Python và truyền tham số giải nén (**function_args)
                    result = AVAILABLE_TOOLS[function_name](**function_args)
                except Exception as e:
                    result = {"error": f"Lỗi khi thực thi tool: {str(e)}"}
            else:
                result = {"error": "Tool không tồn tại."}
                
            logger.info(f"Kết quả Tool: {result}")

            # Gửi kết quả của Tool ngược lại cho LLM
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": json.dumps(result, ensure_ascii=False),
            })
            
        # Bước 3: Gọi API lần 2 để LLM đọc kết quả Tool và sinh câu trả lời tự nhiên
        second_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.6,
        )
        
        return second_response.choices[0].message.content

    else:
        # Nếu LLM không gọi Tool mà trả lời trực tiếp
        return response_message.content

# ====================== TEST RUN ======================
if __name__ == "__main__":
    # Tạo câu hỏi giả lập để test
    test_query = "Cửa hàng 25 tuần tới có rủi ro cháy hàng gì không?"
    print(f"\nUSER: {test_query}")
    
    # Chạy Agent
    answer = run_agent(test_query)
    print(f"\nASSISTANT: {answer}")