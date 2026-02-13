import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
import os
from datetime import datetime
from dotenv import load_dotenv

# 🔥 THƯ VIỆN CHÍNH CHỦ BINANCE
from binance.um_futures import UMFutures
from binance.error import ClientError

# --- 1. CẤU HÌNH & KẾT NỐI ---
load_dotenv()

# Lấy API Key
API_KEY = os.getenv("API_KEY") or os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("API_SECRET") or os.getenv("BINANCE_API_SECRET")

LOG_FILE = "../logs/trading.log"
HISTORY_FILE = "balance_history.csv"

# Cấu hình trang
st.set_page_config(
    page_title="🤖 Bot Monitor (Official Lib)",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ✅ KHỞI TẠO CLIENT (OFFICIAL LIB)
@st.cache_resource
def init_client():
    if not API_KEY or not API_SECRET:
        st.error("❌ Không tìm thấy API Key trong file .env")
        return None

    try:
        # Kết nối Testnet rất đơn giản: chỉ cần truyền base_url
        client = UMFutures(
            key=API_KEY,
            secret=API_SECRET,
            base_url='https://testnet.binancefuture.com'  # 🔥 URL Testnet chuẩn
        )

        # Test kết nối bằng cách lấy giờ server
        client.time()
        return client

    except ClientError as error:
        st.error(f"❌ Lỗi kết nối Binance: {error.error_message}")
        return None
    except Exception as e:
        st.error(f"❌ Lỗi không xác định: {e}")
        return None


client = init_client()


# --- 2. HÀM XỬ LÝ DỮ LIỆU (VIẾT LẠI CHO LIB MỚI) ---

def fetch_data():
    """Lấy số dư và vị thế dùng binance.um_futures"""
    if not client:
        return None, None, None

    try:
        # 1. Lấy thông tin tài khoản (V2) - Chứa cả số dư và PnL
        # Official lib trả về raw dictionary
        account_info = client.account()

        # Lấy số dư ví (Total Wallet Balance)
        total_wallet_balance = float(account_info.get('totalWalletBalance', 0.0))

        # Lấy PnL chưa chốt (Total Unrealized Profit)
        total_unrealized_pnl = float(account_info.get('totalUnrealizedProfit', 0.0))

        # Tính tổng tài sản ròng
        net_worth = total_wallet_balance + total_unrealized_pnl

        # 2. Lấy vị thế BTC/USDT
        # positions trả về một list tất cả các cặp
        positions = account_info.get('positions', [])
        btc_pos = None

        for p in positions:
            if p['symbol'] == 'BTCUSDT':  # Lưu ý: Official lib dùng 'BTCUSDT' (không có /)
                # Chỉ lấy nếu volume khác 0 để hiển thị cho gọn
                # Nhưng logic ở đây ta cứ lấy object về để Dashboard xử lý
                btc_pos = p
                break

        return net_worth, total_unrealized_pnl, btc_pos

    except ClientError as error:
        print(f"⚠️ Binance Error: {error.error_message}")
        return None, None, None
    except Exception as e:
        print(f"⚠️ Python Error: {e}")
        return None, None, None


def update_history_csv(net_worth):
    """Lưu lịch sử tài sản"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_data = pd.DataFrame([[timestamp, net_worth]], columns=['Time', 'Net_Worth'])

    if not os.path.exists(HISTORY_FILE):
        new_data.to_csv(HISTORY_FILE, index=False)
    else:
        try:
            df_old = pd.read_csv(HISTORY_FILE)
            if not df_old.empty:
                last_time = pd.to_datetime(df_old.iloc[-1]['Time'])
                if (datetime.now() - last_time).total_seconds() > 60:
                    new_data.to_csv(HISTORY_FILE, mode='a', header=False, index=False)
            else:
                new_data.to_csv(HISTORY_FILE, mode='a', header=False, index=False)
        except Exception:
            pass


def load_logs():
    """Đọc file log"""
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                lines = f.readlines()
                return "".join(lines[-50:][::-1])
        except Exception:
            return "Đang đọc log..."
    return "Chưa có file log."


# --- 3. GIAO DIỆN DASHBOARD ---

st.title("⚡ AI Trading Bot Monitor (Official Lib)")

# A. Gọi hàm lấy dữ liệu
net_worth, unrealized_pnl, position = fetch_data()

main_container = st.container()

with main_container:
    if net_worth is not None:
        update_history_csv(net_worth)

        # B. Hiển thị Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("💰 Net Worth", f"${net_worth:,.2f}", f"${unrealized_pnl:,.2f}")
        with col2:
            # Logic xử lý vị thế cho Official Lib
            side = "NEUTRAL"
            amt = 0.0
            if position:
                amt = float(position.get('positionAmt', 0.0))  # Key khác ccxt
                if amt > 0:
                    side = "LONG"
                elif amt < 0:
                    side = "SHORT"

            st.metric("Vị thế", side)
        with col3:
            st.metric("Size (BTC)", f"{amt}")
        with col4:
            # Official lib: leverage, entryPrice
            lev = position.get('leverage', 1) if position else 1
            entry = float(position.get('entryPrice', 0.0)) if position else 0.0
            st.metric("Lev / Entry", f"x{lev}", f"{entry:,.1f}")

        # C. Biểu đồ
        st.divider()
        st.subheader("📈 Tài sản ròng")
        if os.path.exists(HISTORY_FILE):
            try:
                df = pd.read_csv(HISTORY_FILE)
                if not df.empty:
                    df['Time'] = pd.to_datetime(df['Time'])
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df['Time'], y=df['Net_Worth'],
                        mode='lines+markers',
                        name='Net Worth',
                        line=dict(color='#00CC96', width=2),
                        fill='tozeroy'
                    ))
                    fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10), template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.caption("Chờ dữ liệu...")

    # D. Logs
    st.divider()
    st.subheader("📜 Live Logs")
    unique_key = f"log_{datetime.now().timestamp()}"
    st.text_area("Log Output", load_logs(), height=300, key=unique_key)
    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

# Auto Refresh
time.sleep(60)
st.rerun()