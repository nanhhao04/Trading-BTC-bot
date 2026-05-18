import os
import openai
from dotenv import load_dotenv

# Load biến môi trường
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_strategy_insight(logs: str) -> str:
    """Gọi OpenAI API để đúc kết nhận định từ log hệ thống."""
    if not openai.api_key:
        return "⚠️ Chưa cấu hình OPENAI_API_KEY trong file .env."
        
    if not logs or logs.strip() == "":
        return "Chưa có đủ dữ liệu log để phân tích."

    prompt = f"""
Bạn là một trợ lý AI chuyên nghiệp về Giao dịch Tiền điện tử (AI Trading Assistant).
Dưới đây là các log gần đây nhất từ hệ thống giao dịch tự động của tôi.
Hãy phân tích và đưa ra 1-2 câu nhận định ngắn gọn, súc tích (Strategy Insight) bằng tiếng Anh hoặc tiếng Việt.
Giọng điệu: Chuyên nghiệp, giống như một chuyên gia định lượng (Quant).
Không giải thích dài dòng, chỉ đưa ra kết luận chiến lược.

LOGS:
{logs}
"""
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional quant trading AI assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Lỗi khi gọi OpenAI API: {str(e)}"

def chat_with_logs(logs: str, user_message: str, chat_history: list) -> str:
    """Chat trực tiếp với AI về log."""
    if not openai.api_key:
        return "⚠️ Chưa cấu hình OPENAI_API_KEY trong file .env."

    messages = [
        {"role": "system", "content": "You are a crypto trading assistant helping the user monitor their bot. Use the provided logs as context to answer their questions."},
        {"role": "system", "content": f"CURRENT LOGS:\n{logs}"}
    ]
    
    # Thêm lịch sử chat
    for msg in chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
        
    messages.append({"role": "user", "content": user_message})

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.5,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Lỗi chat API: {str(e)}"
