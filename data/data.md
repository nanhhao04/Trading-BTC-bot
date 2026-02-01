# Các chỉ số quan trọng trong Trading Crypto (BTC, Altcoins)

Trong giao dịch Crypto, nhà đầu tư và các hệ thống AI/RL thường sử dụng nhiều chỉ số để đánh giá:

---

## 1. Chỉ số giá cơ bản (Price Metrics)

### 1.1 OHLCV
Dữ liệu nền tảng nhất trong trading:

- **Open**: giá mở cửa
- **High**: giá cao nhất
- **Low**: giá thấp nhất
- **Close**: giá đóng cửa
- **Volume**: khối lượng giao dịch

### 1.2 Return (Lợi nhuận)

#### Simple Return:
$$
R_t = \frac{P_t - P_{t-1}}{P_{t-1}}
$$

#### Log Return (chuẩn trong finance):
$$
r_t = \log\left(\frac{P_t}{P_{t-1}}\right)
$$

---

## 2. Chỉ số xu hướng (Trend Indicators)

### 2.1 Moving Average (MA)

- **SMA**: trung bình giá đơn giản
- **EMA**: trung bình giá lũy thừa (nhạy hơn)

$$
SMA_n = \frac{1}{n}\sum_{i=1}^{n}P_i
$$

### 2.2 MACD (Moving Average Convergence Divergence)

Đo động lượng xu hướng:

$$
MACD = EMA_{12} - EMA_{26}
$$

- MACD > 0: xu hướng tăng
- MACD < 0: xu hướng giảm

---

## 3. Chỉ số động lượng (Momentum Indicators)

### 3.1 RSI (Relative Strength Index)

Đo mức quá mua/quá bán:

$$
RSI = 100 - \frac{100}{1+RS}
$$

- RSI > 70: Overbought (quá mua)
- RSI < 30: Oversold (quá bán)

### 3.2 Stochastic Oscillator

So sánh giá đóng cửa với vùng giá gần đây:

- Trên 80: quá mua
- Dưới 20: quá bán

---

## 4. Chỉ số biến động (Volatility Indicators)

### 4.1 Volatility (Độ biến động)

Crypto có volatility rất cao:

$$
\sigma = std(r_t)
$$

### 4.2 Bollinger Bands

Dải biến động quanh MA:

$$
Upper = MA + k\sigma
$$

$$
Lower = MA - k\sigma
$$

- Giá chạm band trên: có thể đảo chiều giảm
- Giá chạm band dưới: có thể đảo chiều tăng

### 4.3 ATR (Average True Range)

Đo biên độ dao động trung bình:

- ATR cao: thị trường biến động mạnh
- ATR thấp: thị trường ổn định

---

## 5. Chỉ số thanh khoản (Liquidity Metrics)

### 5.1 Volume

Khối lượng giao dịch lớn thể hiện thị trường mạnh và ổn định hơn.

### 5.2 Bid-Ask Spread

Chênh lệch giá mua/bán:

$$
Spread = Ask - Bid
$$

Spread lớn thể hiện thanh khoản kém, dễ trượt giá (slippage).

### 5.3 Order Book Imbalance

Mất cân bằng giữa lệnh mua và bán:

$$
Imbalance = \frac{BidVolume}{BidVolume + AskVolume}
$$

---

## 6. Chỉ số tâm lý thị trường (Sentiment Indicators)

### 6.1 Fear & Greed Index

- Fear: thị trường hoảng loạn
- Greed: thị trường hưng phấn

### 6.2 Funding Rate (Futures Market)

- Funding > 0: Long quá nhiều
- Funding < 0: Short quá nhiều

### 6.3 Open Interest

Tổng số hợp đồng futures đang mở:

- Tăng mạnh: dòng tiền vào thị trường
- Giảm mạnh: dòng tiền rút ra

---

## 7. Chỉ số đánh giá chiến lược (Performance Metrics)

### 7.1 Total Profit / Return

$$
Return = \frac{V_{final}-V_{initial}}{V_{initial}}
$$

### 7.2 Sharpe Ratio (Quan trọng nhất)

Đo lợi nhuận trên rủi ro:

$$
Sharpe = \frac{E[R]-R_f}{\sigma_R}
$$

Sharpe cao thể hiện chiến lược tốt và ổn định.

### 7.3 Maximum Drawdown (MDD)

Đo mức sụt giảm lớn nhất:

$$
MDD = \max\left(\frac{Peak - Trough}{Peak}\right)
$$

MDD cao thể hiện rủi ro phá sản lớn.

### 7.4 Win Rate

$$
WinRate = \frac{\#Trades_{win}}{\#Trades_{total}}
$$

### 7.5 Profit Factor

$$
PF = \frac{GrossProfit}{GrossLoss}
$$

PF > 1.5 thể hiện chiến lược tốt.

---

## 8. Chỉ số quan trọng trong RL Trading

Trong Reinforcement Learning, reward thường dùng:

### Reward có phí giao dịch

$$
r_t = \Delta V_t - c|a_t-a_{t-1}|
$$

### Reward risk-adjusted

$$
r_t = Return - \lambda \cdot Drawdown
$$

---

## Tổng kết: Các chỉ số bạn bắt buộc phải biết

| Nhóm | Chỉ số quan trọng |
|------|------------------|
| Giá | OHLCV, Return, Log-return |
| Trend | MA, EMA, MACD |
| Momentum | RSI, Stochastic |
| Volatility | Volatility, ATR, Bollinger |
| Liquidity | Volume, Spread, Orderbook |
| Sentiment | Fear&Greed, Funding, Open Interest |
| Performance | Sharpe, MDD, Profit Factor |
| RL Reward | Transaction cost, Risk-adjusted reward |

---

Nếu bạn làm dự án **RL trading BTC**, nên tập trung nhất vào:

- Log-return
- Volatility
- Transaction cost
- Sharpe ratio reward
- Max Drawdown penalty