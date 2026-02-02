# 1. Chuẩn bị API key cho sàn Binance
Cho tiền tươi thóc thật: 
BASE_URL = "https://fapi.binance.com"

Cho tiền demo: 
BASE_URL = "https://testnet.binancefuture.com"

# 2. Load Data và xử lý dữ liệu.
## Load Data
* Dùng api của sàn binance: BASE_URL = "https://fapi.binance.com"
* Thu các chỉ số của sàn. [Open Time, Open, High, Low, Close, Volume, ...] của BTC/USDT khung 1H từ 1/1/2023
## Process data
* Dùng thư viện ta (technical analysis dành riêng cho finance).
- Tính toan các trường dữ liệu RSI14, MACD, ATR, SMA50-200, logReturn.
- Các trường dữ liệu nâng cao hơn: 
  - Z-score, 
  - TrendFlag(1: Uptrend, 0: Downtrend), 
  - Distance to SMA (Đo xem giá đã đi xa đường trung bình chưa 
  - vol_change: volume hiện tại so với trung bình volume 20 phiên.
