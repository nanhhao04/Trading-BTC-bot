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

    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    OUT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../data/raw"))
    os.makedirs(OUT, exist_ok=True)

    # 4 tháng đầu năm 2026: 01/01/2026 -> 01/05/2026
    START = "01/01/2026"
    END = "01/05/2026"

    print(f"Fetching 1h {START} to {END}...")
    df1h = fetch("BTCUSDT", "1h", START, END)
    print(f"  1h: {len(df1h)} rows | {df1h['date'].min()} -> {df1h['date'].max()}")
    df1h.to_csv(f"{OUT}/BTCUSDT_1h_2026_4m.csv", index=False)
    print("  Saved: BTCUSDT_1h_2026_4m.csv")

    print(f"Fetching 15m {START} to {END}...")
    df15m = fetch("BTCUSDT", "15m", START, END)
    print(f"  15m: {len(df15m)} rows | {df15m['date'].min()} -> {df15m['date'].max()}")
    df15m.to_csv(f"{OUT}/BTCUSDT_15m_2026_4m.csv", index=False)
    print("  Saved: BTCUSDT_15m_2026_4m.csv")

    print(f"Fetching 5m {START} to {END}...")
    df5m = fetch("BTCUSDT", "5m", START, END)
    print(f"  5m: {len(df5m)} rows | {df5m['date'].min()} -> {df5m['date'].max()}")
    df5m.to_csv(f"{OUT}/BTCUSDT_5m_2026_4m.csv", index=False)
    print("  Saved: BTCUSDT_5m_2026_4m.csv")

    print("\nDone. All March 2026 data collected.")
