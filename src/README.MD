# Optimized Reward Engineering for PPO Trading Agent

**Model Architecture:** Proximal Policy Optimization (PPO) with Tanh Squashing
**Optimization Goal:** Risk-Adjusted Return Maximization (Sharpe Ratio Proxy)

---

## 1. Hàm Mục Tiêu Tổng Quát (The Scalarized Objective Function)

Thay vì tối ưu hóa lợi nhuận PnL đơn thuần, Agent tối ưu hóa một hàm mục tiêu đa biến, cân bằng giữa lợi nhuận, rủi ro nắm giữ vị thế và sụt giảm tài khoản.

**Công thức Reward tại bước thời gian $t$:**

$$
R_t = \sigma \times \left[ \underbrace{\Phi_{trend} \cdot (a_t \cdot r_{log})}_{\text{Profit Component}} - \underbrace{\beta \cdot DD_t}_{\text{Drawdown Penalty}} - \underbrace{\lambda \cdot a_t^2}_{\text{Holding Risk}} \right]
$$

**Trong đó:**
* $a_t \in [-1, 1]$: Vị thế (Action) sau khi qua hàm kích hoạt `tanh`.
* $r_{log}$: Logarithmic Return của giá.
* $\sigma, \Phi, \beta, \lambda$: Các siêu tham số (Hyperparameters).

---

## 2. Chi Tiết Các Thành Phần & Tham Số

### 2.1. Lợi nhuận Logarit (Logarithmic Return)
Sử dụng Log Return thay vì Simple Return để đảm bảo tính cộng (additivity) theo thời gian và đối xứng toán học.

$$
r_{log} = \ln \left( \frac{P_t}{P_{t-1}} \right)
$$

### 2.2. Phạt Rủi ro Vị thế (Quadratic Holding Penalty) - Quan trọng nhất
Đây là cơ chế "Soft Constraint" giúp Agent không "All-in" (Action = 1.0) một cách mù quáng.

$$
Cost_{risk} = \lambda \times (a_t)^2
$$

* **Tham số thực tế:** $\lambda = \mathbf{0.002}$
* **Cơ chế:**
    * Hàm phạt là hàm bậc 2 (Quadratic).
    * Nếu $a_t = 1.0$ (Full Margin) $\rightarrow$ Phạt $1^2 \times 0.002 = 0.002$ (**0.2%**).
    * Nếu $a_t = 0.5$ (Half Margin) $\rightarrow$ Phạt $0.5^2 \times 0.002 = 0.0005$ (**0.05%**).
* **Ý nghĩa:** Agent chỉ dám All-in khi Lợi nhuận kỳ vọng ($E[r_{log}]$) lớn hơn chi phí rủi ro 0.2%. Nếu tín hiệu yếu, nó tự động giảm Volume xuống.

### 2.3. Phạt Sụt giảm Tài khoản (Drawdown Penalty)
Cơ chế bảo vệ vốn, ép Agent phải "nhát tay" hơn khi tài khoản đang lỗ sâu.

$$
DD_t = \frac{\text{MaxNetWorth} - \text{CurrentNetWorth}}{\text{MaxNetWorth}}
$$

$$
Penalty_{DD} = \beta \times DD_t
$$

* **Tham số thực tế:** $\beta = \mathbf{0.05}$
* **Ý nghĩa:** Giảm hệ số này xuống 0.05 giúp Bot không bị "tê liệt" (paralysis) khi lỗ. Nó vẫn dám giao dịch để gỡ lỗ (Recovery mode) thay vì sợ phạt mà đứng im.

### 2.4. Hệ số Xu hướng (Trend Alignment Factor)
Sử dụng tham số `alpha` để phạt hành vi đi ngược xu hướng dài hạn.

$$
\Phi_{trend} = \begin{cases} 
1.0 & \text{if } (a_t > 0 \land I_{trend}=1) \lor (a_t < 0 \land I_{trend}=0) \\ 
1.0 - \alpha & \text{otherwise (Counter-trend)} 
\end{cases}
$$

* **Tham số thực tế:** $\alpha = \mathbf{0.1}$
* **Ý nghĩa:** Nếu đi ngược Trend (ví dụ Long trong Downtrend), phần thưởng bị giảm đi 10% ($\Phi = 0.9$). Không cấm đoán hoàn toàn, nhưng khuyến khích đánh thuận xu hướng.

### 2.5. Hệ số Phóng đại (Reward Scaling)
Chuẩn hóa giá trị Reward về vùng hoạt động tốt nhất của Gradient Descent.

* **Tham số thực tế:** $\sigma = \mathbf{10.0}$
* **Ý nghĩa:** Biến đổi Reward thô (thường rất nhỏ, vd: 0.001) thành con số lớn hơn (vd: 0.01 - 0.1) để mạng Neuron cập nhật trọng số nhanh và hiệu quả hơn.

---

## 3. Không Gian Hành Động (Action Space Dynamics)

Để đảm bảo Agent luôn hoạt động trong giới hạn cho phép và có độ mượt (differentiable):

$$
a_{raw} \sim \pi_\theta(s_t) \quad (\text{Gaussian Distribution})
$$

$$
a_t = \tanh(a_{raw})
$$

* **Input:** $a_{raw} \in (-\infty, +\infty)$ (Đầu ra thô từ mạng Neuron).
* **Output:** $a_t \in (-1, 1)$ (Tỷ trọng vốn thực tế).
* **Lợi ích:** Giải quyết vấn đề bão hòa Gradient và loại bỏ biên cứng (Hard clipping), cho phép Bot học được các giá trị tiệm cận biên (như 0.9, 0.95).

---

## 4. Bảng Tổng Hợp Tham Số (Configuration Cheat Sheet)

| Tham số | Ký hiệu | Giá trị | Vai trò | Tác động Hành vi |
| :--- | :---: | :---: | :--- | :--- |
| **Scaling** | $\sigma$ | **10.0** | Khuếch đại tín hiệu | Giúp Bot học nhanh hơn, tín hiệu rõ ràng hơn. |
| **Trend Alpha** | $\alpha$ | **0.1** | Phạt ngược Trend | Giảm 10% lợi nhuận nếu trade ngược dòng. |
| **DD Beta** | $\beta$ | **0.05** | Phạt sụt giảm | Phạt nhẹ khi lỗ, khuyến khích gỡ lỗ (Recovery). |
| **Risk Lambda** | $\lambda$ | **0.002** | Phạt lệnh lớn | **Quan trọng nhất:** Chống All-in mù quáng. |