import os
import pandas as pd
from datetime import datetime
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Import Binance & AI Agent
from binance.um_futures import UMFutures
from dashboard.ai_agent import get_strategy_insight, chat_with_logs

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
API_KEY = os.getenv("API_KEY") or os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("API_SECRET") or os.getenv("BINANCE_API_SECRET")

LOG_FILE = os.path.join(os.path.dirname(__file__), "..", "logs", "trading.log")
HISTORY_FILE = os.path.join(os.path.dirname(__file__), "..", "balance_history.csv")

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

def init_client():
    if not API_KEY or not API_SECRET:
        return None
    try:
        client = UMFutures(key=API_KEY, secret=API_SECRET, base_url='https://testnet.binancefuture.com')
        client.time()
        return client
    except Exception:
        return None

client = init_client()

def read_logs(lines=100):
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                content = f.readlines()
                return content[-lines:][::-1]
        except Exception:
            return []
    return []

@app.get("/")
def read_root():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))

@app.get("/api/metrics")
def get_metrics():
    if not client:
        return {"net_worth": 0.0, "unrealized_pnl": 0.0, "side": "API ERROR", "size": 0.0, "leverage": 1}
    
    try:
        account_info = client.account()
        total_wallet_balance = float(account_info.get('totalWalletBalance', 0.0))
        total_unrealized_pnl = float(account_info.get('totalUnrealizedProfit', 0.0))
        net_worth = total_wallet_balance + total_unrealized_pnl
        
        positions = account_info.get('positions', [])
        btc_pos = next((p for p in positions if p['symbol'] == 'BTCUSDT'), None)
        
        side = "NEUTRAL"
        amt = 0.0
        lev = 1
        if btc_pos:
            amt = float(btc_pos.get('positionAmt', 0.0))
            if amt > 0: side = "LONG"
            elif amt < 0: side = "SHORT"
            lev = btc_pos.get('leverage', 1)

        # Cập nhật lịch sử
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_data = pd.DataFrame([[timestamp, net_worth]], columns=['Time', 'Net_Worth'])
        if not os.path.exists(HISTORY_FILE):
            new_data.to_csv(HISTORY_FILE, index=False)
        else:
            df_old = pd.read_csv(HISTORY_FILE)
            if not df_old.empty:
                last_time = pd.to_datetime(df_old.iloc[-1]['Time'])
                if (datetime.now() - last_time).total_seconds() > 60:
                    new_data.to_csv(HISTORY_FILE, mode='a', header=False, index=False)

        return {
            "net_worth": net_worth,
            "unrealized_pnl": total_unrealized_pnl,
            "side": side,
            "size": amt,
            "leverage": lev
        }
    except Exception as e:
        return {"net_worth": 0.0, "unrealized_pnl": 0.0, "side": "ERROR", "size": 0.0, "leverage": 1}

@app.get("/api/logs")
def get_logs():
    logs = read_logs(100)
    return {"logs": logs}

@app.get("/api/history")
def get_history():
    if os.path.exists(HISTORY_FILE):
        try:
            df = pd.read_csv(HISTORY_FILE)
            if not df.empty:
                return {"times": df['Time'].tolist(), "net_worths": df['Net_Worth'].tolist()}
        except:
            pass
    return {"times": [], "net_worths": []}

@app.get("/api/insight")
def get_insight():
    logs = "".join(read_logs(100))
    insight = get_strategy_insight(logs)
    return {"insight": insight}

class ChatRequest(BaseModel):
    message: str
    history: list

@app.post("/api/chat")
def post_chat(req: ChatRequest):
    logs = "".join(read_logs(100))
    reply = chat_with_logs(logs, req.message, req.history)
    return {"reply": reply}
