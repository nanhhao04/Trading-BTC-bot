import requests
import pandas as pd
import time
import os
from datetime import datetime

BASE_URL = "https://fapi.binance.com"
ENDPOINT = "/fapi/v1/klines"


def fetch(symbol, interval, start_str, end_str):
    start_ts = int(datetime.strptime(start_str, "%d/%m/%Y").timestamp() * 1000)
    end_ts   = int(datetime.strptime(end_str,   "%d/%m/%Y").timestamp() * 1000)
    all_data = []
    
    while start_ts < end_ts:
        r = requests.get(
            BASE_URL + ENDPOINT,
            params={"symbol": symbol, "interval": interval,
                    "startTime": start_ts, "endTime": end_ts, "limit": 1500}
        )
        data = r.json()
        if not data:
            break
        for c in data:
            all_data.append([c[0], float(c[1]), float(c[2]),
                              float(c[3]), float(c[4]), float(c[5])])
        start_ts = data[-1][0] + 1
        time.sleep(0.3)
        
        # In log tiến độ
        current_date = datetime.fromtimestamp(start_ts / 1000).strftime('%Y-%m-%d %H:%M')
        print(f"\rFetched up to {current_date}...", end="", flush=True)

    print()
    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


if __name__ == "__main__":
    OUT = "../../data/raw"
    os.makedirs(OUT, exist_ok=True)

    print("Fetching 5m Historical Data (01/01/2023 -> 01/03/2026)...")
    df5m_hist = fetch("BTCUSDT", "5m", "01/01/2023", "01/03/2026")
    print(f"  5m Hist: {len(df5m_hist)} rows | {df5m_hist['date'].min()} -> {df5m_hist['date'].max()}")
    df5m_hist.to_csv(f"{OUT}/BTCUSDT_5m.csv", index=False)
    print("  Saved: BTCUSDT_5m.csv")
    print("\nDone. 5m Historical data collected.")
