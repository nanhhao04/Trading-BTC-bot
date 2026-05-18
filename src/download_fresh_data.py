"""
Download BTCUSDT data từ Binance và chuẩn bị cho training.
Script này sẽ:
1. Tải dữ liệu 1h từ tháng 1-5/2026
2. Tính technical indicators (RSI, MACD, ATR...)
3. Lưu ra BTCUSDT_1h_features_full.csv
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Thử import binance, nếu không có thì yêu cầu cài
try:
    from binance.spot import Spot
    from binance.error import ClientError
except ImportError:
    print("Cài binance-connector: pip install binance-connector")
    sys.exit(1)

def download_klines(symbol: str, interval: str, start_date: str, end_date: str, limit=1000):
    """
    Download OHLCV từ Binance.
    
    symbol: BTCUSDT
    interval: 1h, 4h, 1d...
    start_date: '2026-01-01'
    end_date: '2026-05-31'
    """
    client = Spot()  # Public API, không cần credentials
    
    # Convert dates to milliseconds
    start_ms = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    end_ms = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
    
    all_klines = []
    current_ms = start_ms
    
    print(f"Downloading {symbol} {interval} from {start_date} to {end_date}...")
    
    while current_ms < end_ms:
        try:
            klines = client.klines(
                symbol=symbol,
                interval=interval,
                startTime=current_ms,
                endTime=end_ms,
                limit=limit
            )
            
            if not klines:
                break
            
            all_klines.extend(klines)
            # Lấy thời gian từ kline cuối cùng để tiếp tục
            current_ms = int(klines[-1][0]) + 1  # +1ms để không lặp
            
            print(f"  Downloaded {len(all_klines)} candles...")
            
        except ClientError as e:
            print(f"Error downloading data: {e}")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
            break
    
    # Convert to DataFrame
    df = pd.DataFrame(all_klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'n_trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    # Chuyển đổi kiểu dữ liệu
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    
    # Sắp xếp theo thời gian
    df = df.sort_values('open_time').reset_index(drop=True)
    
    print(f"\n✓ Downloaded {len(df)} candles")
    return df[['open_time', 'open', 'high', 'low', 'close', 'volume']]


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính technical indicators từ OHLCV.
    Hàm này tính RSI, MACD, ATR, Bollinger Bands...
    """
    df = df.copy()
    
    # 1. Returns
    df['returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # 2. RSI (14)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 3. MACD (12, 26, 9)
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # 4. ATR (14)
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    
    # 5. Bollinger Bands (20, 2)
    sma_20 = df['close'].rolling(20).mean()
    std_20 = df['close'].rolling(20).std()
    df['bb_upper'] = sma_20 + (std_20 * 2)
    df['bb_lower'] = sma_20 - (std_20 * 2)
    df['bb_mid'] = sma_20
    
    # 6. Simple Moving Averages
    df['sma_9'] = df['close'].rolling(9).mean()
    df['sma_21'] = df['close'].rolling(21).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    
    # 7. Trend indicator (0=downtrend, 1=uptrend)
    df['I_trend'] = (df['sma_9'] > df['sma_21']).astype(int)
    
    # 8. Volatility (20-period)
    df['volatility'] = df['returns'].rolling(20).std()
    
    # Fill NaN từ technical indicators
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    return df


def prepare_state_features(df: pd.DataFrame, window: int = 90) -> pd.DataFrame:
    """
    Chuẩn bị state features (normalized indicators) cho agent observation.
    """
    df = df.copy()
    
    # Normalize price: [close - low] / [high - low]
    df['price_normalized'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    
    # Normalize returns (z-score)
    df['returns_normalized'] = (df['returns'] - df['returns'].rolling(window).mean()) / (df['returns'].rolling(window).std() + 1e-8)
    
    # Normalize RSI (0-1)
    df['rsi_normalized'] = df['rsi'] / 100.0
    
    # MACD histogram normalized
    df['macd_hist_normalized'] = df['macd_hist'] / (df['macd_hist'].rolling(window).std() + 1e-8)
    
    # Bollinger Band position: (close - bb_lower) / (bb_upper - bb_lower)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
    df['bb_position'] = df['bb_position'].clip(0, 1)
    
    # ATR normalized
    df['atr_normalized'] = df['atr'] / df['close'] * 100  # % of price
    
    # Volatility normalized
    df['volatility_normalized'] = (df['volatility'] - df['volatility'].rolling(window).mean()) / (df['volatility'].rolling(window).std() + 1e-8)
    
    # Trend as float (0.0 or 1.0)
    df['I_trend_float'] = df['I_trend'].astype(float)
    
    return df


def main():
    # ============ THAY ĐỔI NGÀY TẠI ĐÂY ============
    # Nếu bạn muốn data từ tháng 1-5/2026, dùng:
    start_date = '2026-01-01'
    end_date = '2026-05-31'
    
    # Hoặc nếu bạn muốn data gần đây hơn (ví dụ: last 3 months):
    # end_date = datetime.now().strftime('%Y-%m-%d')
    # start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    
    interval = '1h'
    symbol = 'BTCUSDT'
    
    # 1. Download data
    df = download_klines(symbol, interval, start_date, end_date)
    
    print("\n✓ Data downloaded. Calculating indicators...")
    
    # 2. Tính indicators
    df = calculate_indicators(df)
    
    print("✓ Indicators calculated. Preparing state features...")
    
    # 3. Chuẩn bị state features
    df = prepare_state_features(df, window=90)
    
    # 4. Lưu full features (dùng cho training)
    output_dir = '../data/processed'
    os.makedirs(output_dir, exist_ok=True)
    
    # Lưu feature set đầy đủ
    feature_cols = [
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'returns', 'rsi', 'macd', 'macd_signal', 'macd_hist',
        'atr', 'bb_upper', 'bb_lower', 'bb_mid',
        'sma_9', 'sma_21', 'sma_50',
        'I_trend', 'volatility'
    ]
    df_features = df[feature_cols].copy()
    output_file_full = os.path.join(output_dir, f'{symbol}_{interval}_features_full.csv')
    df_features.to_csv(output_file_full, index=False)
    print(f"✓ Saved features to: {output_file_full}")
    
    # Lưu state features (normalized indicators cho agent observation)
    state_cols = [
        'price_normalized', 'returns_normalized', 'rsi_normalized',
        'macd_hist_normalized', 'bb_position', 'atr_normalized',
        'volatility_normalized', 'I_trend_float'
    ]
    df_state = df[state_cols].copy()
    output_file_state = os.path.join(output_dir, f'{symbol}_{interval}_state.csv')
    df_state.to_csv(output_file_state, index=False)
    print(f"✓ Saved state features to: {output_file_state}")
    
    print(f"\n✓✓✓ Data preparation complete!")
    print(f"  - Total candles: {len(df)}")
    print(f"  - Date range: {df['open_time'].min()} to {df['open_time'].max()}")
    print(f"  - Price range: {df['close'].min():.2f} - {df['close'].max():.2f} USDT")
    print(f"\nNow update config.yaml paths:")
    print(f"  data_full: \"../data/processed/{symbol}_{interval}_features_full.csv\"")
    print(f"  data_state: \"../data/processed/{symbol}_{interval}_state.csv\"")


if __name__ == '__main__':
    main()
