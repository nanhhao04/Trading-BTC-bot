# ĐỊNH NGHĨA HÀM THƯỞNG CHUNG

Thay vì dùng Lợi nhuận đơn thuần ($P_t - P_{t-1}$), ta xây dựng một hàm mục tiêu vô hướng hóa (Scalarized Objective) để giải quyết vấn đề **Temporal Credit Assignment** và **Risk Management**.

**Công thức Tổng quát:**
$$R_t = \phi_{regime} \times \left( \underbrace{R_{unrealized}}_{\text{Dẫn hướng}} + \alpha \underbrace{R_{realized}}_{\text{Chốt đơn}} - \beta \underbrace{DD_t}_{\text{Rủi ro}} \right) - \text{Cost}$$

### 3.1. Thành phần Dẫn hướng (Dense Reward Manifold)
$$R_{unrealized} = \text{Position}_t \times \frac{P_t - P_{t-1}}{P_{t-1}}$$
* Tạo ra một bề mặt Gradient liên tục, giúp Agent học được hướng đi đúng ngay cả khi chưa đóng lệnh.
* 
* log_return = np.log(current_price / past_price)
        
* step_reward = position * log_return

*  Tuy nhiên, để đơn giản cho RL, ta coi sự thay đổi ròng của Net Worth là tổng hợp của cả Unrealized và Realized. Ta nhấn mạnh vào Realized bằng cách thêm trọng số alpha nếu có giao dịch đóng.

### 3.2. Thành phần Rủi ro (Non-linear Penalty)
$$DD_t = \frac{\text{Peak}_t - \text{Equity}_t}{\text{Peak}_t}$$
* Phạt dựa trên **Max Drawdown**.
* **Cơ chế Toán học:** Tạo ra một "vách núi" (Cliff) trong không gian $Q$-value. Nếu một hành động dẫn đến trạng thái có Drawdown cao, giá trị $Q(s,a)$ tại đó sẽ sụt giảm nghiêm trọng $\rightarrow$ Agent học cách tránh xa các chuỗi hành động rủi ro cao.
* current_dd = (self.max_net_worth - net_worth) / self.max_net_worth
* dd_penalty = self.beta * current_dd

### 3.3. Hệ số Môi trường (Regime Factor)
$$\phi_{regime} = \begin{cases} 1.0 & \text{if } P_t > \text{SMA}_{200} \\ 0.5 & \text{if } P_t \le \text{SMA}_{200} \end{cases}$$
* Giảm kỳ vọng phần thưởng trong Downtrend. Làm giảm $Q$-value của hành động MUA, khiến Agent trở nên "thận trọng" (Conservative) hơn.
* Long trong down trend thì sẽ nhận 50% phần reward (trend_flag = 0 và pos > 0)


### Clipping reward trong khoảng [-10, 10]