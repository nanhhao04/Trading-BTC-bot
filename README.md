# 📉 Deep Reinforcement Learning for Bitcoin Trading: Mathematical Foundation & Implementation

**Technical Reference for AI Engineers** **Based on:** *Prasetyo et al., 2025* **Topic:** Optimal Control in Non-Stationary Financial Markets (High-Frequency Crypto Trading).

---

## 📑 Table of Contents
1. [Vấn đề Cốt lõi: Tính Không Dừng (Non-Stationarity)](#1-vấn-đề-cốt-lõi-tính-không-dừng-non-stationarity)
2. [Feature Engineering: Xử lý Tín hiệu & Chuẩn hóa](#2-feature-engineering-xử-lý-tín-hiệu--chuẩn-hóa)
3. [Reward Shaping: Hàm Mục Tiêu Đa Biến](#3-reward-shaping-hàm-mục-tiêu-đa-biến)
4. [Algorithm Dynamics: Toán học của DQN vs PPO](#4-algorithm-dynamics-toán-học-của-dqn-vs-ppo)
5. [Implementation Checklist](#5-implementation-checklist)

---

## 1. Vấn đề Cốt lõi: Tính Không Dừng (Non-Stationarity)

Thị trường tiền điện tử (Bitcoin) thường tuân theo mô hình **Geometric Brownian Motion** với các tham số thay đổi theo thời gian (Time-varying parameters):

$$dP_t = \mu_t P_t dt + \sigma_t P_t dW_t$$

* $\mu_t$: Drift (Xu hướng) thay đổi liên tục.
* $\sigma_t$: Volatility (Độ biến động) không ổn định (Heteroscedasticity).
* **Hệ quả:** Hiện tượng **Covariate Shift**. Phân phối dữ liệu đầu vào $P(X)$ thay đổi liên tục, khiến mạng Nơ-ron bị "lạc trôi" (Gradient trôi dạt).

---

## 2. Feature Engineering: Xử lý Tín hiệu & Chuẩn hóa

Để giải quyết Covariate Shift, ta không đưa giá thô ($P_t$) vào mạng. Ta cần biến đổi không gian trạng thái sang dạng ổn định (Stationary).

### 2.1. Rolling Z-Score Normalization
Kỹ thuật này đưa phân phối cục bộ tại mọi thời điểm về chuẩn tắc $\mathcal{N}(0, 1)$.

$$z_t = \frac{x_t - \mu_{rolling}}{\sigma_{rolling}}$$

* **Tại sao cần thiết?**
    * Tránh bão hòa hàm kích hoạt (Activation Saturation). Nếu $x_t$ quá lớn (giá BTC 100k), tích vô hướng $W \cdot x + b$ sẽ đẩy đầu ra của Sigmoid/Tanh về $\pm 1$, nơi đạo hàm $\approx 0$ (Vanishing Gradient).
    * $z_t$ giữ cho đầu vào luôn nằm trong vùng tuyến tính của hàm kích hoạt.

### 2.2. Regime Filter (Bộ lọc Xu hướng)
Sử dụng kiến thức chuyên gia (Expert Knowledge) để tách không gian trạng thái thành 2 chế độ (Regimes):

$$I_{trend} = \mathbb{I}(P_t > \text{SMA}_{200})$$

* $I_{trend} = 1$: Bull Market (Uptrend).
* $I_{trend} = 0$: Bear Market (Downtrend).
* **Input Vector $S_t$:**
    $$S_t = [\text{Norm\_Close}, \text{RSI}_{14}, \text{Volatility}, \text{MACD}, \text{SMA\_Dist}, I_{trend}]$$

---

## 3. Reward Shaping: Hàm Mục Tiêu Đa Biến

Thay vì dùng Lợi nhuận đơn thuần ($P_t - P_{t-1}$), ta xây dựng một hàm mục tiêu vô hướng hóa (Scalarized Objective) để giải quyết vấn đề **Temporal Credit Assignment** và **Risk Management**.

**Công thức Tổng quát:**
$$R_t = \phi_{regime} \times \left( \underbrace{R_{unrealized}}_{\text{Dẫn hướng}} + \alpha \underbrace{R_{realized}}_{\text{Chốt đơn}} - \beta \underbrace{DD_t}_{\text{Rủi ro}} \right) - \text{Cost}$$

### 3.1. Thành phần Dẫn hướng (Dense Reward Manifold)
$$R_{unrealized} = \text{Position}_t \times \frac{P_t - P_{t-1}}{P_{t-1}}$$
* Tạo ra một bề mặt Gradient liên tục, giúp Agent học được hướng đi đúng ngay cả khi chưa đóng lệnh.

### 3.2. Thành phần Rủi ro (Non-linear Penalty)
$$DD_t = \frac{\text{Peak}_t - \text{Equity}_t}{\text{Peak}_t}$$
* Phạt dựa trên **Max Drawdown**.
* **Cơ chế Toán học:** Tạo ra một "vách núi" (Cliff) trong không gian $Q$-value. Nếu một hành động dẫn đến trạng thái có Drawdown cao, giá trị $Q(s,a)$ tại đó sẽ sụt giảm nghiêm trọng $\rightarrow$ Agent học cách tránh xa các chuỗi hành động rủi ro cao.

### 3.3. Hệ số Môi trường (Regime Factor)
$$\phi_{regime} = \begin{cases} 1.0 & \text{if } P_t > \text{SMA}_{200} \\ 0.8 & \text{if } P_t \le \text{SMA}_{200} \end{cases}$$
* Giảm kỳ vọng phần thưởng trong Downtrend. Làm giảm $Q$-value của hành động MUA, khiến Agent trở nên "thận trọng" (Conservative) hơn.

---

## 4. Algorithm Dynamics: Toán học của DQN vs PPO

### 4.1. DQN (Deep Q-Network) - Robustness against Noise
DQN chiến thắng trong thị trường nhiễu (Sideways) nhờ **Huber Loss**.

$$\mathcal{L}_{Huber}(\delta) = \begin{cases} \frac{1}{2}\delta^2 & \text{if } |\delta| \le \delta_{thresh} \\ \delta_{thresh} (|\delta| - \frac{1}{2}\delta_{thresh}) & \text{otherwise} \end{cases}$$

* **Phân tích:**
    * Với nhiễu nhỏ (Normal market): Học nhanh như hàm bậc 2 (MSE).
    * Với nhiễu lớn/Outliers (Flash crash): Học tuyến tính (Linear), không bị quá nhạy cảm.
* **Hành vi:** DQN lọc bỏ nhiễu, chỉ vào lệnh khi tín hiệu cực rõ ($Q$-value vượt trội). **Phù hợp cho Swing Trading.**

### 4.2. PPO (Proximal Policy Optimization) - Vulnerability in High Variance
PPO gặp khó khăn do ước lượng Advantage ($A_t$) có phương sai cao.

$$L^{CLIP} = \mathbb{E} [ \min(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t) ]$$

* **Vấn đề:**
    * Trong Crypto, $A_t(s,a) = Q(s,a) - V(s)$ biến động cực mạnh.
    * PPO là thuật toán ngẫu nhiên (Stochastic Policy). Việc "thử sai" trong môi trường biến động lớn dẫn đến chuỗi thua lỗ liên tiếp (Whipsaw effect).
* **Hành vi:** PPO trade rất nhiều, thắng lớn khi có Trend mạnh, nhưng thua lỗ nặng khi thị trường đảo chiều (Mean Reversion). **Phù hợp cho Momentum Trading.**

---

## 5. Implementation Checklist

Để tái hiện (reproduce) thành công bài báo này, Codebase cần đảm bảo:

- [ ] **Preprocessing:**
    - [ ] Cài đặt `RollingWindow` để tính Z-Score (không dùng Global Mean).
    - [ ] Tính `SMA200` để tạo feature `trend_flag`.
- [ ] **Environment (Gym):**
    - [ ] `step()` function phải trả về `info` chứa Drawdown hiện tại.
    - [ ] Implement hàm reward tổng hợp (Unrealized + Realized - Penalty).
- [ ] **Agent Config (Stable-Baselines3):**
    - [ ] **DQN:** Sử dụng `exploration_fraction=0.1` (giảm explore khi deploy), `huber_loss`.
    - [ ] **PPO:** Tăng `ent_coef` (Entropy coefficient) nếu Agent bị kẹt (không chịu trade), hoặc giảm nếu trade quá nhiều.
- [ ] **Safety Layer:**
    - [ ] Hard-code quy tắc: Nếu `Close < SMA200`, giảm `max_position_size` xuống 50%.

---
*Document prepared by Gemini specifically for AI Engineering usage.*