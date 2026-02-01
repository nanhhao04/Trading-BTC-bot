import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import AverageTrueRange


def add_technical_indicators(df):
    """
    Hàm này nhận vào DataFrame (OHLCV) và thêm các cột chỉ báo kỹ thuật
    Sử dụng thư viện 'ta' thay vì 'pandas_ta'.
    """
    # Tạo bản sao để không ảnh hưởng dữ liệu gốc
    df = df.copy()

    # Đảm bảo dữ liệu giá là kiểu số (float) để tránh lỗi tính toán
    df['close'] = df['close'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)

    # 1. RSI (Relative Strength Index) - Chu kỳ 14
    rsi_indicator = RSIIndicator(close=df["close"], window=14)
    df["rsi"] = rsi_indicator.rsi()

    # 2. MACD (Moving Average Convergence Divergence)
    macd = MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # 3. ATR (Average True Range) - Đo biến động
    atr_indicator = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14)
    df["atr"] = atr_indicator.average_true_range()

    # 4. SMA (Simple Moving Average)
    df["sma_50"] = SMAIndicator(close=df["close"], window=50).sma_indicator()
    df["sma_200"] = SMAIndicator(close=df["close"], window=200).sma_indicator()

    # 5. Log Return (Lợi nhuận logarit) - Quan trọng cho AI học
    # Thay vì dùng giá tuyệt đối, AI học tốt hơn với % thay đổi
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # 6. Loại bỏ các dòng NaN (Do chỉ báo cần thời gian khởi tạo)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


# --- CHẠY THỬ ĐỂ KIỂM TRA ---
if __name__ == "__main__":
    try:
        # Đọc file CSV bạn đã tải ở bước trước
        df = pd.read_csv("../data/raw/BTCUSDT_1h.csv")
        print(f"📊 Dữ liệu gốc: {df.shape}")

        df_processed = add_technical_indicators(df)
        print(f"✅ Dữ liệu sau xử lý: {df_processed.shape}")

        # In thử 5 dòng đầu
        print("\n=== 5 Dòng đầu tiên ===")
        print(df_processed[['date', 'close', 'rsi', 'macd', 'atr']].head())

        # Lưu file
        import os

        os.makedirs("../data/processed", exist_ok=True)
        df_processed.to_csv("../data/processed/BTCUSDT_1h_features.csv", index=False)
        print("\n💾 Đã lưu file processed thành công.")

    except FileNotFoundError:
        print("❌ Lỗi: Không tìm thấy file data raw. Hãy chạy 'data_loader.py' trước!")
    except Exception as e:
        print(f"❌ Lỗi khác: {e}")