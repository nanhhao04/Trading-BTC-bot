from binance.um_futures import UMFutures  # <--- THAY ĐỔI: Dùng UMFutures thay vì CMFutures
from binance.error import ClientError
import time
from dotenv import load_dotenv
import os
import yaml
from logging_tool import setup_logging
load_dotenv()
with open("../config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
QUANTITY_DQN = cfg.get("dqn_params", {}).get("fixed_qty", 0.002)
SYMBOL = cfg["symbol"]
LEVERAGE = cfg["leverage"]
WINDOW_SIZE = cfg["env"]["window_size"]
MAX_CAPITAL_USAGE = cfg["max_capital_usage"]

# MAINNET:
# BASE_URL = "https://fapi.binance.com"
# TESTNET (khuyến nghị):
BASE_URL = "https://testnet.binancefuture.com"


logger = setup_logging()


class BinanceExecutor:
    def __init__(self, symbol= SYMBOL , leverage=LEVERAGE, dqn_quantity=QUANTITY_DQN):
        self.symbol = symbol
        self.leverage = leverage
        self.dqn_quantity = dqn_quantity
        self.max_capital_usage = MAX_CAPITAL_USAGE
        self.client = UMFutures(key=API_KEY,
                                secret=API_SECRET,
                                base_url=BASE_URL,)

    def set_leverage(self):
        print(f"Setting leverage to x{LEVERAGE}...")
        try:
            response = self.client.change_leverage(
                symbol=SYMBOL,
                leverage=LEVERAGE
            )
            print("Leverage updated success:", response)
        except ClientError as e:
            print("Error setting leverage:", e.error_message)

    def get_max_qty(self, price):

        account = self.client.account()
        usdt_balance = float(
            next((item for item in account['assets'] if item['asset'] == 'USDT'), None)['walletBalance'])
        print(f"Max USDT balance: {usdt_balance}")

        # Công thức: (Balance * Leverage) / Price
        max_qty = (usdt_balance * self.leverage * 0.95) / price
        return max_qty


    def get_current_state(self):
        positions = self.client.get_position_risk(symbol=self.symbol)
        ticker = self.client.ticker_price(symbol=self.symbol)

        current_pos_amt = 0.0
        for p in positions:
            if p['symbol'] == self.symbol:
                current_pos_amt = float(p['positionAmt'])
                break

        current_price = float(ticker['price'])
        return current_pos_amt, current_price

    def _place_order(self, side, qty, reduce_only=False):
        try:
            qty = round(abs(qty), 3)
            if qty == 0: return False

            params = {
                "symbol": self.symbol,
                "side": side,
                "type": "MARKET",
                "quantity": qty,
            }
            if reduce_only:
                params["reduceOnly"] = "true"

            resp = self.client.new_order(**params)
            print(f" {side} {qty} success! ID: {resp['orderId']}")
            return True
        except ClientError as e:
            print(f" Order Failed: {e}")
            return False

    def close_position(self):
        pos, _ = self.get_current_state()
        if pos != 0:
            side = "SELL" if pos > 0 else "BUY"
            self._place_order(side, abs(pos), reduce_only=True)

    def execute_dqn(self, action_id):
        current_pos, _ = self.get_current_state()
        qty = self.dqn_quantity

        if action_id == 0:
            return

        if action_id == 3: # close
            self.close_position()

        if action_id == 1: #long
            if current_pos < 0:  # Đang Short -> Đóng Short
                self.close_position()
                time.sleep(1)
            if current_pos <= 0:  # Mở Long
                self._place_order("BUY", qty)

        if action_id == 2: #short
            if current_pos > 0:  # Đang long -> đóng
                self.close_position()
                time.sleep(1)
            if current_pos <= 0:  # Mở short
                self._place_order("SELL", qty)

    def execute_ppo(self, action_val):
        current_pos, current_price = self.get_current_state()
        print(f"Current position: {current_pos} | Current price: {current_price}" )

        max_possible_qty = self.get_max_qty(current_price)   # lượng tối đa có thể mua trong 1 lệnh
        target_pct = action_val * max_possible_qty * self.max_capital_usage  # lượng đánh

        delta = (target_pct - current_pos )       # khối lượng thực thi

        logger.info(f" Maxpossibleqty: {max_possible_qty},  Current position: {current_pos}, target position: {target_pct}, delta: {delta}")
        if abs(delta) < (max_possible_qty * 0.01 * 0.1):
            logger.info(f"PPO Delta too small ({delta:.4f})(< {max_possible_qty * 0.01 * 0.1})")
            return

        logger.info(f"PPO Rebalance: Current {current_pos:.3f} -> Target {action_val:.3f} | Delta: {delta:.3f}")

        if delta > 0:
            self._place_order("BUY", delta)
        else:
            self._place_order("SELL", abs(delta))

