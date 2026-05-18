"""
Preprocess raw 2026 4-month data -> features_full + state CSVs
"""
import pandas as pd
import numpy as np
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
RAW_DIR      = os.path.join(PROJECT_ROOT, 'data', 'raw')
PROCESSED    = os.path.join(PROJECT_ROOT, 'data', 'processed')

# Add src/data to path for features_full import
sys.path.insert(0, SCRIPT_DIR)
from features_full import add_technical_indicators

WARMUP_BARS  = 250   # SMA200 cần 200 nến warmup


def preprocess_2026_4m(interval: str):
    # 1. Load full historical raw (để lấy warmup bars trước 01/01)
    hist_path = f"{RAW_DIR}/BTCUSDT_{interval}.csv"
    data_path  = f"{RAW_DIR}/BTCUSDT_{interval}_2026_4m.csv"

    if not os.path.exists(hist_path):
        # Fallback: if historical file doesn't exist, just use the 4m file with internal warmup
        print(f"Warning: Missing historical file {hist_path}. Indicators will have NaNs at start.")
        df_combined = pd.read_csv(data_path)
        df_combined["date"] = pd.to_datetime(df_combined["timestamp"], unit="ms")
    else:
        df_hist = pd.read_csv(hist_path)
        df_data = pd.read_csv(data_path)

        # Ensure date column
        if "date" not in df_hist.columns:
            df_hist["date"] = pd.to_datetime(df_hist["timestamp"], unit="ms")
        else:
            df_hist["date"] = pd.to_datetime(df_hist["date"])
        
        df_data["date"] = pd.to_datetime(df_data["timestamp"], unit="ms")

        # 2. Lấy WARMUP_BARS nến cuối của historical (trước tháng 1)
        warmup = df_hist.tail(WARMUP_BARS).copy()
        warmup = warmup[["timestamp", "open", "high", "low", "close", "volume", "date"]]

        # 3. Ghép warmup + 2026 data
        df_combined = pd.concat([warmup, df_data], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    print(f"[{interval}] Combined shape: {df_combined.shape}")
    print(f"[{interval}] Date range: {df_combined['date'].min()} -> {df_combined['date'].max()}")

    # 4. Tính technical indicators
    df_processed = add_technical_indicators(df_combined)

    # 5. Chỉ giữ lại phần năm 2026 (bỏ warmup)
    df_final = df_processed[df_processed["date"] >= "2026-01-01"].copy()
    df_final = df_final.reset_index(drop=True)

    print(f"[{interval}] 2026-only rows after processing: {len(df_final)}")

    # 6. Lưu files
    os.makedirs(PROCESSED, exist_ok=True)

    full_out  = f"{PROCESSED}/BTCUSDT_{interval}_2026_4m_features_full.csv"
    state_out = f"{PROCESSED}/BTCUSDT_{interval}_2026_4m_state.csv"

    df_final.to_csv(full_out, index=False, float_format="%.5f")

    state_cols = ["Norm_Close", "RSI14", "Volatility", "MACD", "SMA_Dist", "I_trend"]
    df_final[state_cols].to_csv(state_out, index=False, float_format="%.5f")

    print(f"[{interval}] Saved: {full_out}")
    print(f"[{interval}] Saved: {state_out}")
    print()

    return df_final


if __name__ == "__main__":
    print("=" * 50)
    print("Preprocessing 2026 Q1-Q2 (4 months) data")
    print("=" * 50)
    preprocess_2026_4m("1h")
    preprocess_2026_4m("15m")
    preprocess_2026_4m("5m")
    print("All done.")
