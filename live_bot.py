from binance.um_futures import UMFutures  # <--- THAY ĐỔI: Dùng UMFutures thay vì CMFutures
from binance.error import ClientError
import time
from dotenv import load_dotenv
import os


load_dotenv()
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

SYMBOL = "BTCUSDT"       # <--- THAY ĐỔI: Cặp USDT
QUANTITY = 0.002          # <--- THAY ĐỔI: Số lượng tính bằng BTC (0.02 BTC), KHÔNG phải số contract
SLEEP_SECONDS = 100       # Thời gian giữ lệnh demo
LEVERAGE = 20
# MAINNET:
# BASE_URL = "https://fapi.binance.com"

# TESTNET (khuyến nghị):
BASE_URL = "https://testnet.binancefuture.com"

# =========================
# KHỞI TẠO CLIENT
# =========================
client = UMFutures(      # <--- THAY ĐỔI: Khởi tạo UMFutures
    key=API_KEY,
    secret=API_SECRET,
    base_url=BASE_URL
)
def set_leverage():
    print(f"Setting leverage to x{LEVERAGE}...")
    try:
        response = client.change_leverage(
            symbol=SYMBOL,
            leverage=LEVERAGE
        )
        print("Leverage updated success:", response)
    except ClientError as e:
        print("Error setting leverage:", e.error_message)

def place_buy():
    print(f"Placing BUY order for {QUANTITY} {SYMBOL}...")
    try:
        order = client.new_order(
            symbol=SYMBOL,
            side="BUY",
            type="MARKET",
            quantity=QUANTITY
        )
        print("BUY order success:")
        print(f"Order ID: {order['orderId']}")
        print(f"Status: {order['status']}")
    except ClientError as e:
        print("BUY order error:", e.error_message)

def place_sell_reduce_only():
    print("Placing SELL (reduce-only) order...")
    try:
        order = client.new_order(
            symbol=SYMBOL,
            side="SELL",
            type="MARKET",
            quantity=QUANTITY,
            reduceOnly=True
        )
        print("SELL order success:")
        print(f"Order ID: {order['orderId']}")
        print(f"Status: {order['status']}")
    except ClientError as e:
        print("SELL order error:", e.error_message)

if __name__ == "__main__":
    print("=== BINANCE USDT-M FUTURES DEMO ===")

    # Kiểm tra số dư USDT trước khi trade (Optional nhưng nên có)
    try:
        account = client.account()
        usdt_balance = next((item for item in account['assets'] if item['asset'] == 'USDT'), None)
        if usdt_balance:
            print(f"Current USDT Balance: {usdt_balance['walletBalance']}")
    except Exception as e:
        print("Could not check balance:", e)

    print("-" * 30)

    # 1. MỞ VỊ THẾ (LONG)
    set_leverage()
    place_buy()

    print(f"\nHolding position for {SLEEP_SECONDS} seconds...\n")
    time.sleep(SLEEP_SECONDS)

    # 2. ĐÓNG VỊ THẾ (CLOSE)
    place_sell_reduce_only()

    print("\n=== DEMO FINISHED ===")