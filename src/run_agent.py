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
from data.features_full import add_technical_indicators
from logging_tool import setup_logging

load_dotenv()
logger = setup_logging()

try:
    with open("../config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
except FileNotFoundError:
    try:
        with open('config.yaml', 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file config.yaml (ở ../ hoặc ./)")
        exit()





MODEL_TYPE = cfg["model_type"]

if MODEL_TYPE == "PPO":
    MODEL_PATH = cfg["paths"]["ppo_dir"]
else:
    MODEL_PATH = cfg["paths"]["dqn_dir"]

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
    # 1. Lấy dữ liệu thị trường mới nhất
    last_row = df_features.iloc[-1]

    market_state = np.array([
        last_row['Norm_Close'],
        last_row['RSI14'],
        last_row['Volatility'],
        last_row['MACD'],
        last_row['SMA_Dist'],
        last_row['I_trend']
    ])

    # 2. Lấy dữ liệu tài khoản
    current_pos_amt, current_price = executor.get_current_state()
    max_qty = executor.get_max_qty(current_price)

    if max_qty > 0:
        current_pos_pct = current_pos_amt / max_qty
    else:
        current_pos_pct = 0.0

    # Chuẩn hóa về [-1, 1]
    current_pos_pct = np.clip(current_pos_pct, -1.0, 1.0)

    # Giả lập PnL
    account_pnl_pct = 0.0

    account_state = np.array([current_pos_pct, account_pnl_pct])

    # Gộp lại
    obs = np.concatenate((market_state, account_state)).astype(np.float32)
    return obs, last_row['close']


def main():
    print(f"STARTING LIVE BOT [{MODEL_TYPE}] - {SYMBOL} ({TIMEFRAME})")
    print(f"Model Path: {MODEL_PATH}")

    final_model_path = MODEL_PATH
    if not os.path.exists(final_model_path):
        if os.path.exists(final_model_path + ".zip"):
            final_model_path += ".zip"
        else:
            print(f"Error: Model not found at {MODEL_PATH}")
            return

    executor = BinanceExecutor(symbol=SYMBOL)

    print(f"Loading model: {final_model_path}")
    if MODEL_TYPE == "PPO":
        model = PPO.load(final_model_path)
    else:
        model = DQN.load(final_model_path)
        print("Model loaded successfully!")


    print("Waiting for next candle check...")

    while True:
        try:
            # 1. Lấy dữ liệu
            df = get_live_klines(executor.client, SYMBOL, TIMEFRAME, limit=WINDOW_SIZE)

            if df is not None and not df.empty:
                # 2. Tính Feature
                df_processed = add_technical_indicators(df)
                # 3. Tạo State
                obs, current_price = construct_observation(df_processed, executor)

                # Log
                print(f"\n{datetime.now().strftime('%H:%M:%S')} | Price: {current_price:.2f}")
                print(f"State: RSI={obs[1]:.2f} | Trend={obs[5]:.0f} | Pos={obs[6]:.2f}")

                # 4. Dự đoán
                action, _ = model.predict(obs, deterministic=True)     # deterministic true là dùng mean của gaus

                # 5. Thực thi
                if MODEL_TYPE == "DQN":
                    act_int = int(action)
                    print(f"DQN Signal: {act_int}")
                    executor.execute_dqn(act_int)

                elif MODEL_TYPE == "PPO":
                    target_pct = action[0]
                    logger.info(f"PPO Target: {target_pct:.2f} ({(target_pct * 100 * MAX_CAPITAL_USAGE):.1f}% Vốn)")
                    executor.execute_ppo(target_pct)


            print(" Sleeping 60s...")
            time.sleep(60)

        except KeyboardInterrupt:
            print("\nBot stopped by user.")
            break
        except Exception as e:
            print(f"Critical Error: {e}")
            time.sleep(10)


if __name__ == "__main__":
    main()