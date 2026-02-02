import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import AverageTrueRange
import os


def add_technical_indicators(df):
    df = df.copy()
    # 1. Ép kiểu dữ liệu
    for col in ['close', 'high', 'low', 'volume']:
        df[col] = df[col].astype(float)

    # RSI (14)
    df["raw_RSI"] = RSIIndicator(close=df["close"], window=14).rsi()
    # MACD Histogram
    macd = MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["raw_MACD"] = macd.macd_diff()
    # ATR (Volatility)
    atr = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14)
    df["raw_ATR"] = atr.average_true_range()
    # SMA 50 & 200
    df["SMA50"] = SMAIndicator(close=df["close"], window=50).sma_indicator()
    df["SMA200"] = SMAIndicator(close=df["close"], window=200).sma_indicator()

    # FEATURE ENGINEERING & NORMALIZATION
    # 1. Norm_Close (Rolling Z-score)
    window_z = 50
    roll_mean = df["close"].rolling(window=window_z).mean()
    roll_std = df["close"].rolling(window=window_z).std()
    df["Norm_Close"] = (df["close"] - roll_mean) / (roll_std + 1e-8)

    # 2. RSI14: Chuẩn hóa về [-1, 1] thay vì [0, 100]
    df["RSI14"] = (df["raw_RSI"] / 50.0) - 1.0

    # 3. Volatility: Dùng Rolling Z-score thay vì Global Mean
    # Lý do: Biến động năm 2024 khác năm 2018. Dùng Rolling giúp Model thích nghi.
    # Bước 1: Tính ATR % (để loại bỏ yếu tố giá tuyệt đối)
    atr_pct = df["raw_ATR"] / df["close"]
    # Bước 2: Z-score hóa cái ATR % đó
    atr_mean = atr_pct.rolling(window=window_z).mean()
    atr_std = atr_pct.rolling(window=window_z).std()
    df["Volatility"] = (atr_pct - atr_mean) / (atr_std + 1e-8)

    # 4. MACD: Cũng dùng Rolling Z-score
    # Lý do: MACD khi giá 60k sẽ to hơn khi giá 10k. Phải Z-score nó.
    macd_mean = df["raw_MACD"].rolling(window=window_z).mean()
    macd_std = df["raw_MACD"].rolling(window=window_z).std()
    df["MACD"] = (df["raw_MACD"] - macd_mean) / (macd_std + 1e-8)

    # 5. SMA Distance (Mean Reversion Signal)
    df["SMA_Dist"] = (df["close"] - df["SMA50"]) / df["SMA50"]

    # 6. Trend Flag (Regime Detection)
    df["I_trend"] = np.where(df["close"] > df["SMA200"], 1.0, 0.0)

    # Loại bỏ NaN sinh ra do các cửa sổ trượt (Rolling windows)
    # Cần drop khoảng 200 dòng đầu (do SMA200)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Xử lý vô cùng (inf) nếu có lỗi chia 0
    df.replace([np.inf, -np.inf], 0, inplace=True)

    return df


if __name__ == "__main__":
    # Đường dẫn file (sửa lại cho đúng máy của bạn)
    input_path = "../../data/raw/BTCUSDT_1h.csv"
    output_dir = "../../data/processed"

    if os.path.exists(input_path):
        df = pd.read_csv(input_path)
        print(f"Raw data shape: {df.shape}")

        df_processed = add_technical_indicators(df)
        print(f"Processed data shape: {df_processed.shape}")

        # Kiểm tra nhanh thống kê (Nên nằm trong khoảng -5 đến 5)
        cols_check = ["Norm_Close", "RSI14", "Volatility", "MACD"]
        print("\n--- Statistics Check (Should be standardized) ---")
        print(df_processed[cols_check].describe().loc[['mean', 'std', 'min', 'max']])

        os.makedirs(output_dir, exist_ok=True)

        # 1. Lưu file FULL (để debug)
        df_processed.to_csv(f"{output_dir}/BTCUSDT_1h_features_full.csv", index=False, float_format="%.5f")

        # 2. Lưu file STATE (chỉ dùng để train)
        state_cols = ["Norm_Close", "RSI14", "Volatility", "MACD", "SMA_Dist", "I_trend"]
        df_state = df_processed[state_cols]
        df_state.to_csv(f"{output_dir}/BTCUSDT_1h_state.csv", index=False, float_format="%.5f")

        print(f"\nSaved files to: {output_dir}")
    else:
        print(f"Error: Input file not found at {input_path}")