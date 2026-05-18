import time
import pandas as pd
import numpy as np
import os
import yaml
import logging_tool
from dotenv import load_dotenv
from datetime import datetime
from stable_baselines3 import PPO, DQN

from binance_api import BinanceExecutor
from data.preprocess_from_raw import calculate_indicators, prepare_state_features
from logging_tool import setup_logging
from metric import PerformanceTracker  # Import Tracker

load_dotenv()
logger = setup_logging()

# --- LOAD CONFIG ---
# Tự động xác định đường dẫn động đến config.yaml
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'config.yaml'))

try:
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
except FileNotFoundError:
    print(f"Error: config.yaml not found at {CFG_PATH}")
    exit()

MODEL_TYPE = cfg["model_type"]
MODEL_PATH = cfg["paths"]["ppo_dir"] if MODEL_TYPE == "PPO" else cfg["paths"]["dqn_dir"]
TIMEFRAME = cfg["timeframes"]
SYMBOL = cfg["symbol"]
LEVERAGE = cfg["leverage"]
WINDOW_SIZE = cfg["env"]["window_size"]
MAX_CAPITAL_USAGE = cfg["max_capital_usage"]


def get_live_klines(client, symbol, interval, limit=100):
    try:
        klines = client.klines(symbol=symbol, interval=interval, limit=limit)
        data = []
        for k in klines:
            data.append({
                "timestamp": k[0],
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5])
            })
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error fetching klines: {e}")
        return None


def construct_observation(df_features, executor):
    # [FIX] Dùng iloc[-2] thay vì iloc[-1]
    # Binance API trả về nến đang hình thành (chưa đóng) ở vị trí cuối cùng.
    # Training luôn dùng nến đã đóng → phải dùng nến kế cuối để tránh distribution shift.
    last_row = df_features.iloc[-2]

    market_state = np.array([
        last_row['price_normalized'],
        last_row['returns_normalized'],
        last_row['rsi_normalized'],
        last_row['macd_hist_normalized'],
        last_row['bb_position'],
        last_row['atr_normalized'],
        last_row['volatility_normalized'],
        last_row['I_trend_float']
    ])

    current_pos_amt, current_price = executor.get_current_state()

    # Lấy Balance để track performance
    try:
        current_balance = executor.get_balance()
    except:
        current_balance = 10000.0  # Default if API fails

    max_qty = executor.get_max_qty(current_price)

    if max_qty > 0:
        current_pos_pct = current_pos_amt / max_qty
    else:
        current_pos_pct = 0.0

    current_pos_pct = np.clip(current_pos_pct, -1.0, 1.0)
    account_pnl_pct = 0.0
    account_state = np.array([current_pos_pct, account_pnl_pct])

    obs = np.concatenate((market_state, account_state)).astype(np.float32)

    return obs, current_price, current_pos_pct, current_balance


def main():
    print(f"STARTING LIVE BOT [{MODEL_TYPE}] - {SYMBOL} ({TIMEFRAME})")

    # Tự động phân giải đường dẫn tương đối (như ../model/...) từ thư mục gốc của project
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
    final_model_path = os.path.normpath(os.path.join(PROJECT_ROOT, MODEL_PATH.lstrip('../')))
    
    if not os.path.exists(final_model_path) and not os.path.exists(final_model_path + ".zip"):
        final_model_path = MODEL_PATH

    if not os.path.exists(final_model_path):
        if os.path.exists(final_model_path + ".zip"):
            final_model_path += ".zip"
        else:
            print(f"Error: Model not found at {MODEL_PATH}\nResolved path: {final_model_path}")
            return

    executor = BinanceExecutor(symbol=SYMBOL)
    executor.set_leverage()

    print(f"Loading model: {final_model_path}")
    if MODEL_TYPE == "PPO":
        model = PPO.load(final_model_path)
    else:
        model = DQN.load(final_model_path)
    print("Model loaded.")

    # --- INIT TRACKER ---
    try:
        initial_bal = executor.get_balance()
    except:
        initial_bal = 10000.0

    tracker = PerformanceTracker(initial_balance=initial_bal)
    print(f"Tracker initialized. Base Balance: ${initial_bal:.2f}")

    while True:
        try:
            # Lấy tối thiểu 250 nến để tính toán SMA200 và các chỉ báo kỹ thuật một cách đầy đủ
            lookback_limit = max(int(WINDOW_SIZE), 250)
            df = get_live_klines(executor.client, SYMBOL, TIMEFRAME, limit=lookback_limit)

            if df is not None and not df.empty:
                df_processed = calculate_indicators(df)
                df_processed = prepare_state_features(df_processed, window=90)

                # Get Obs & State
                obs, current_price, current_pos_pct, current_balance = construct_observation(df_processed, executor)

                # Update Tracker
                tracker.update(current_balance, current_pos_pct, current_price)

                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Price: {current_price:.2f}")
                print(f"State: RSI={obs[1]:.2f} | Trend={obs[5]:.0f} | Pos={obs[6]:.2f}")
                logger.info(f"OBS: {obs[0]:.4f}, {obs[1]:.4f} ...")

                # Predict
                action, _ = model.predict(obs, deterministic=True)

                # Execute
                if MODEL_TYPE == "DQN":
                    if hasattr(action, 'item'):
                        act_int = int(action.item())
                    else:
                        act_int = int(action)
                    print(f"DQN Signal: {act_int}")
                    executor.execute_dqn(act_int)

                elif MODEL_TYPE == "PPO":
                    if isinstance(action, np.ndarray):
                        raw_action = float(action.item())
                    else:
                        raw_action = float(action)

                    target_pct = np.clip(raw_action, -1.0, 1.0)
                    logger.info(f"PPO Target: {target_pct:.2f} ({(target_pct * 100 * MAX_CAPITAL_USAGE):.1f}% Capital)")
                    executor.execute_ppo(target_pct)

                # Print Performance Stats
                if len(tracker.history) > 0:
                    stats = tracker.calculate_metrics()
                    print(
                        f" [PERFORMANCE] Return: {stats.get('Total Return')} | DD: {stats.get('Max Drawdown')} | Winrate: {stats.get('Winrate')}")

                sleep_time = 300 if TIMEFRAME == "5m" else 900 if TIMEFRAME == "15m" else 3600
                print(f"Sleeping {sleep_time}s...")
                time.sleep(sleep_time)


        except KeyboardInterrupt:
            print("\nBot stopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(10)


if __name__ == "__main__":
    main()