import requests
import pandas as pd
import time
from datetime import datetime
import os

BASE_URL = "https://fapi.binance.com"
KLINES_ENDPOINT = "/fapi/v1/klines"


def get_binance_data(symbol, interval, start_str, end_str=None):

    # Chuyển đổi ngày tháng sang mili-giây (Timestamp)
    start_ts = int(datetime.strptime(start_str, "%d/%m/%Y").timestamp() * 1000)
    if end_str:
        end_ts = int(datetime.strptime(end_str, "%d/%m/%Y").timestamp() * 1000)
    else:
        end_ts = int(time.time() * 1000)

    all_data = []
    limit = 1500

    print(f" Đang tải dữ liệu {symbol} khung {interval} từ {start_str}...")

    while start_ts < end_ts:
        # Gọi API
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ts,
            "endTime": end_ts,
            "limit": limit
        }

        try:
            response = requests.get(BASE_URL + KLINES_ENDPOINT, params=params)
            data = response.json()

            if not data:
                break

            # Lưu dữ liệu vào list
            # Cấu trúc trả về của Binance: [Open Time, Open, High, Low, Close, Volume, ...]
            for candle in data:
                all_data.append([
                    candle[0],  # Timestamp
                    float(candle[1]),  # Open
                    float(candle[2]),  # High
                    float(candle[3]),  # Low
                    float(candle[4]),  # Close
                    float(candle[5]),  # Volume
                ])

            # Cập nhật thời gian bắt đầu cho vòng lặp tiếp theo
            start_ts = data[-1][0] + 1

            temp_date = datetime.fromtimestamp(start_ts / 1000).strftime('%d/%m/%Y')
            print(f"   -> Đã tải đến: {temp_date}")

            # để không bị Binance chặn IP
            time.sleep(0.5)

        except Exception as e:
            print(f"Lỗi kết nối: {e}")
            time.sleep(5)
            continue

    # Chuyển sang DataFrame và lưu file
    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    output_path = f"../../data/raw/{symbol}_{interval}.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Hoàn tất! Đã lưu {len(df)} dòng dữ liệu tại: {output_path}")
    return df


if __name__ == "__main__":
    SYMBOL = "BTCUSDT"
    TIMEFRAME = "1h"
    START_DATE = "01/01/2023"

    get_binance_data(SYMBOL, TIMEFRAME, START_DATE)