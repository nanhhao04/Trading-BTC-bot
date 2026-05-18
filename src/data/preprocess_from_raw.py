"""
preprocess_from_raw.py — Tao features_full.csv va state.csv tu raw CSV co san.

Dung:
    cd src/data
    python preprocess_from_raw.py                    # 1h, toan bo raw data
    python preprocess_from_raw.py --timeframe 15m    # 15m
    python preprocess_from_raw.py --timeframe 5m     # 5m
    python preprocess_from_raw.py --start_date 2023-01-01  # loc tu ngay

Output (ghi de):
    data/processed/BTCUSDT_1h_features_full.csv
    data/processed/BTCUSDT_1h_state.csv
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
RAW_DIR      = os.path.join(SCRIPT_DIR, '..', '..', 'data', 'raw')
PROCESSED_DIR= os.path.join(SCRIPT_DIR, '..', '..', 'data', 'processed')


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    # 1. Returns
    df['returns'] = np.log(df['close'] / df['close'].shift(1))

    # 2. RSI (14)
    delta = df['close'].diff()
    gain  = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs    = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))

    # 3. MACD (12, 26, 9)
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd']        = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist']   = df['macd'] - df['macd_signal']

    # 4. ATR (14)
    hl  = df['high'] - df['low']
    hc  = (df['high'] - df['close'].shift()).abs()
    lc  = (df['low']  - df['close'].shift()).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()

    # 5. Bollinger Bands (20, 2)
    sma_20     = df['close'].rolling(20).mean()
    std_20     = df['close'].rolling(20).std()
    df['bb_upper'] = sma_20 + std_20 * 2
    df['bb_lower'] = sma_20 - std_20 * 2
    df['bb_mid']   = sma_20

    # 6. SMAs
    df['sma_9']  = df['close'].rolling(9).mean()
    df['sma_21'] = df['close'].rolling(21).mean()
    df['sma_50'] = df['close'].rolling(50).mean()

    # 7. Trend (sma_9 > sma_21)
    df['I_trend'] = (df['sma_9'] > df['sma_21']).astype(int)

    # 8. Volatility (20-period std of returns)
    df['volatility'] = df['returns'].rolling(20).std()

    df = df.ffill().bfill()
    return df


def prepare_state_features(df: pd.DataFrame, window: int = 90) -> pd.DataFrame:
    df = df.copy()

    df['price_normalized'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)

    df['returns_normalized'] = (
        (df['returns'] - df['returns'].rolling(window).mean())
        / (df['returns'].rolling(window).std() + 1e-8)
    )

    df['rsi_normalized'] = df['rsi'] / 100.0

    df['macd_hist_normalized'] = df['macd_hist'] / (df['macd_hist'].rolling(window).std() + 1e-8)

    df['bb_position'] = (
        (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
    ).clip(0, 1)

    df['atr_normalized']       = df['atr'] / df['close'] * 100
    df['volatility_normalized'] = (
        (df['volatility'] - df['volatility'].rolling(window).mean())
        / (df['volatility'].rolling(window).std() + 1e-8)
    )

    df['I_trend_float'] = df['I_trend'].astype(float)

    df = df.ffill().bfill()
    return df


def process_timeframe(timeframe: str, start_date: str = None, end_date: str = None):
    raw_path = os.path.join(RAW_DIR, f'BTCUSDT_{timeframe}.csv')
    if not os.path.exists(raw_path):
        print(f"[ERROR] Khong tim thay raw file: {raw_path}")
        return

    print(f"\n[{timeframe}] Doc raw data: {raw_path}")
    df = pd.read_csv(raw_path)
    print(f"  Raw rows: {len(df)}")

    # --- Chuan hoa ten cot ---
    # Raw file co the co cot 'timestamp' (ms) hoac 'open_time' (datetime string)
    if 'timestamp' in df.columns and 'open_time' not in df.columns:
        # Convert ms timestamp -> datetime string cho open_time
        df['open_time'] = pd.to_datetime(df['timestamp'], unit='ms')
    elif 'open_time' not in df.columns and 'date' in df.columns:
        df['open_time'] = pd.to_datetime(df['date'])
    elif 'open_time' in df.columns:
        df['open_time'] = pd.to_datetime(df['open_time'])

    df = df.sort_values('open_time').reset_index(drop=True)

    # --- Filter theo date range ---
    if start_date:
        df = df[df['open_time'] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df['open_time'] <= pd.Timestamp(end_date)]
    df = df.reset_index(drop=True)
    print(f"  Sau filter: {len(df)} rows | {df['open_time'].iloc[0]} -> {df['open_time'].iloc[-1]}")

    # --- Tinh indicators ---
    print(f"  Tinh technical indicators...")
    df = calculate_indicators(df)

    # --- Tinh state features ---
    print(f"  Chuan hoa state features...")
    df = prepare_state_features(df, window=90)

    # --- Luu features_full ---
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    feature_cols = [
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'returns', 'rsi', 'macd', 'macd_signal', 'macd_hist',
        'atr', 'bb_upper', 'bb_lower', 'bb_mid',
        'sma_9', 'sma_21', 'sma_50',
        'I_trend', 'volatility'
    ]
    # Them cot timestamp (ms) neu chua co
    df['timestamp_ms'] = df['open_time'].astype('int64') // 10**6
    feature_cols_out = ['open_time'] + [c for c in feature_cols if c != 'open_time'] + ['timestamp_ms']

    df_out = df[feature_cols_out].copy()
    # Doi ten timestamp_ms -> timestamp de backtest.py doc duoc
    df_out = df_out.rename(columns={'timestamp_ms': 'timestamp'})

    full_path = os.path.join(PROCESSED_DIR, f'BTCUSDT_{timeframe}_features_full.csv')
    df_out.to_csv(full_path, index=False, float_format='%.6f')
    print(f"  [OK] features_full saved: {full_path} ({len(df_out)} rows)")

    # --- Luu state ---
    state_cols = [
        'price_normalized', 'returns_normalized', 'rsi_normalized',
        'macd_hist_normalized', 'bb_position', 'atr_normalized',
        'volatility_normalized', 'I_trend_float'
    ]
    df_state = df[state_cols].copy()
    state_path = os.path.join(PROCESSED_DIR, f'BTCUSDT_{timeframe}_state.csv')
    df_state.to_csv(state_path, index=False, float_format='%.6f')
    print(f"  [OK] state saved:         {state_path} ({len(df_state)} rows)")

    print(f"\n  === Tong ket [{timeframe}] ===")
    print(f"  Rows     : {len(df_out)}")
    print(f"  Tu       : {df['open_time'].iloc[0]}")
    print(f"  Den      : {df['open_time'].iloc[-1]}")
    print(f"  Price    : {df['close'].min():.2f} - {df['close'].max():.2f} USDT")


def main():
    parser = argparse.ArgumentParser(description="Preprocess raw BTCUSDT CSV -> features_full + state")
    parser.add_argument('--timeframe',  type=str, default='1h',
                        help="1h, 15m, 5m (mac dinh: 1h). Dung 'all' de xu ly ca 3.")
    parser.add_argument('--start_date', type=str, default=None,
                        help="Loc data tu ngay nay (YYYY-MM-DD)")
    parser.add_argument('--end_date',   type=str, default=None,
                        help="Loc data den ngay nay (YYYY-MM-DD)")
    args = parser.parse_args()

    if args.timeframe == 'all':
        for tf in ['1h', '15m', '5m']:
            process_timeframe(tf, args.start_date, args.end_date)
    else:
        process_timeframe(args.timeframe, args.start_date, args.end_date)

    print("\nHoan thanh!")


if __name__ == '__main__':
    main()
