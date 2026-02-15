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
from metric import PerformanceTracker  # Import Tracker

load_dotenv()
logger = setup_logging()

# --- LOAD CONFIG ---
try:
    with open("../config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
except FileNotFoundError:
    try:
        with open('config.yaml', 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
    except FileNotFoundError:
        print("Error: config.yaml not found.")
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
    last_row = df_features.iloc[-1]

    market_state = np.array([
        last_row['Norm_Close'],
        last_row['RSI14'],
        last_row['Volatility'],
        last_row['MACD'],
        last_row['SMA_Dist'],
        last_row['I_trend']
    ])

    current_pos_amt, current_price = executor.get_current_state()

    # Lấy Balance để track performance
    try:
        current_balance = executor.get_balance()
    except:
        current_balance = 5000.0  # Default if API fails

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
    print("Model loaded.")

    # --- INIT TRACKER ---
    try:
        initial_bal = executor.get_balance()
    except:
        initial_bal = 5000.0

    tracker = PerformanceTracker(initial_balance=initial_bal)
    print(f"Tracker initialized. Base Balance: ${initial_bal:.2f}")

    while True:
        try:
            df = get_live_klines(executor.client, SYMBOL, TIMEFRAME, limit=WINDOW_SIZE)

            if df is not None and not df.empty:
                df_processed = add_technical_indicators(df)

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
                    act_int = int(action)
                    print(f"DQN Signal: {act_int}")
                    executor.execute_dqn(act_int)

                elif MODEL_TYPE == "PPO":
                    if isinstance(action, np.ndarray):
                        raw_action = float(action.item())
                    else:
                        raw_action = float(action)

                    target_pct = np.tanh(raw_action)
                    logger.info(f"PPO Target: {target_pct:.2f} ({(target_pct * 100 * MAX_CAPITAL_USAGE):.1f}% Capital)")
                    executor.execute_ppo(target_pct)

                # Print Performance Stats
                if len(tracker.history) > 0:
                    stats = tracker.calculate_metrics()
                    print(
                        f" [PERFORMANCE] Return: {stats.get('Total Return')} | DD: {stats.get('Max Drawdown')} | Winrate: {stats.get('Winrate')}")

            print("Sleeping 60s...")
            time.sleep(60)

        except KeyboardInterrupt:
            print("\nBot stopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(10)


if __name__ == "__main__":
    main()