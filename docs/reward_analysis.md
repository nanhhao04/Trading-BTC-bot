# Phân Tích Chuyên Sâu Reward Function v2 & Optimal Parameters

Tài liệu này giải thích chi tiết cấu trúc của **RewardHandler v2**, lý do đằng sau việc lựa chọn các tham số tối ưu (Optimal Parameters) cho dữ liệu nến 1H, và cách các hàm số tương tác với nhau để định hình hành vi của AI Agent (PPO/DQN).

---

## 1. Cơ Chế Hoạt Động Của Reward Function v2

Hàm Reward được thiết kế để giải quyết bài toán cốt lõi của RL trong Trading: Sự kiện thưa thớt (Sparse Rewards) và Cân bằng rủi ro (Risk Management). Tổng Reward mỗi step được tính bằng công thức:

```math
Raw\_Reward = (Step\_Reward + Realized\_Reward - DD\_Penalty - Risk\_Cost) \times Trend\_Factor
```
```math
Final\_Reward = Clip(Raw\_Reward \times Scaling)
```

### Các Thành Phần Cốt Lõi:
1. **Step Reward (`position * log_return`)**: Thưởng/phạt liên tục theo từng biến động giá của nến. Giúp bot học được sự tương quan giữa vị thế và giá trị tài sản thay vì phải đợi chốt lệnh.
2. **Realized Reward (`alpha * trade_return`)**: Kích hoạt ĐỘC NHẤT khi lệnh được ĐÓNG (CLOSE). Cung cấp tín hiệu cực mạnh để bot học được thời điểm cắt lỗ / chốt lời.
3. **DD Penalty (`beta * effective_dd^2`)**: Phạt theo hàm bậc 2 nếu tài khoản rơi vào vùng Drawdown lớn hơn ngưỡng cho phép. Ngăn bot hold những lệnh lỗ quá sâu.
4. **Risk Cost (`position^2 * holding_penalty`)**: Chi phí cơ hội khi nắm giữ vị thế. Ép bot về Flat (đứng ngoài) nếu thị trường sideway không rõ xu hướng.
5. **Trend Factor**: Hệ số khuếch đại tự nhiên. Đánh thuận trend được cộng thêm reward, đánh ngược trend bị giảm trừ.

---

## 2. Bảng Tham Số Tối Ưu (Optimal Params - Khung 1H)

Dựa trên kết quả benchmark bằng cấu hình mô phỏng *Realistic Imperfect Policy* (60% theo trend, 40% ngược trend, 0.04% fee), đây là bộ thông số tối ưu nhất được áp dụng:

```yaml
reward:
  scaling: 15.0          
  alpha: 1.5             
  beta: 0.12            
  holding_penalty: 0.0003 
  dd_threshold: 0.05    
  clip_low: -1.5         
  clip_high: 1.5
```

### Lý do lựa chọn (Định lượng):

#### A. `scaling = 15.0` (Chuẩn hoá Gradient Variance)
- Dữ liệu 1H có độ lệch chuẩn biến động (standard deviation) lớn gấp ~2 lần so với khung 15m (từ ~0.002 lên ~0.005).
- Nếu giữ `scaling = 50.0` như trước đây, Standard Deviation của Reward lên tới `0.45` -> Quá nhiễu (Noisy).
- Thuận theo PPO, lý tưởng nhất là biên độ Reward nằm trong khoảng `[-1, 1]`. Việc chọn `scaling = 15.0` kéo Standard Deviation về mức lý tưởng `~0.13`, giúp mạng nơ-ron hội tụ (converge) nhanh và ổn định hơn rất nhiều.

#### B. `alpha = 1.5` (Duy trì sức mạnh của Realized Signal)
- Do `scaling` giảm từ 50 xuống 15, `Step Reward` trung bình bị giảm mạnh (về mức `0.043`).
- Giữ `alpha = 1.5` đảm bảo `Realized Reward` (khi ĐÓNG lệnh) giữ được cường độ trung bình `0.33` — Lớn gấp **~7.5 lần** so với `Step Reward`.
- **Hệ quả:** Bot nhận biết rõ ràng: "Việc chốt lệnh đúng lúc là hành vi quan trọng và được chú ý nhiều nhất".

#### C. `beta = 0.12` & `dd_threshold = 0.05` (Drawdown Penalty cân bằng)
- Drawdown (DD) cực kỳ phổ biến trong trading thực tế (thường dao động 5-20% khi bot dự đoán sai). 
- Thuật toán phạt DD bắt đầu từ mức rủi ro thực sự (`dd_threshold = 5%`).
- Với `beta = 0.12`, tỷ lệ `DD Penalty / Step Reward` rơi vào khoảng **~39%**. Hình phạt này đủ để răn đe bot sớm cắt lỗ, nhưng không lớn đến mức tiêu diệt hoàn toàn Step Reward (như lỗi `beta = 0.5` làm bot sợ hãi và chỉ ưu tiên đứng ngoài thị trường).

#### D. `clip_low = -1.5` & `clip_high = 1.5` (Bảo vệ PPO)
- Cắt bỏ hoàn toàn các outliers (chẳng hạn khi BTC sập/pump 15-20% trong 1 giờ). 
- Các phần thưởng nằm ngoài biên độ `±1.5` sẽ bị bẻ gãy. Ở benchmark thực tế, `clip=1.5` đã cắt 29 nến dị biệt, giúp model không bị "tẩy não" bởi một sự kiện flash crash.

---

## 3. Các Ví Dụ Cụ Thể (Simulated Steps)

Dưới đây là một số kịch bản để thấy sức mạnh của Reward v2. (Giả định giá BTC thay đổi `+0.5%` hoặc `-0.5%` ở mỗi bước)

> [!TIP]
> Tất cả các phép tính dưới đây đã được nhân với `scaling = 15.0`.

### Kịch bản 1: Mở và Giữ Lệnh (Lãi đang chạy)
- Giá tăng `+0.5%` (0.005). Bot đang giữ `Long (pos = 1.0)`.
- **Step Reward:** $1.0 \times 0.005 \times 15.0 = +0.075$
- **Holding Penalty:** $-0.0003 \times 15.0 = -0.0045$
- **DD Penalty:** Không có (vì Net worth đang tăng).
- **Tổng Reward:** $\approx +0.07$
- *Hành vi bot học được:* Hold lệnh đúng hướng sẽ liên tục được cộng điểm nhỏ.

### Kịch bản 2: Đi sai hướng & Chạm Drawdown lớn
- Giá giảm `-1.0%`. Bot ngoan cố giữ `Long (pos = 1.0)`. Net worth sụt giảm tạo ra Max Drawdown lên tới **15%**.
- **Step Reward:** $1.0 \times (-0.01) \times 15.0 = -0.15$
- **DD Penalty:** (Tài khoản đang âm 15%, vượt threshold 5% -> Effective DD = 10%)
  $\beta \times (0.10^2) \times 15.0 = 0.12 \times 0.01 \times 15.0 = -0.018$
- **Tổng Reward:** $-0.15 - 0.018 = -0.168$
- *Hành vi bot học được:* Không những mất điểm do giá giảm, bot còn bị trừ thêm điểm phạt kép do thả trôi tài khoản vào vùng âm sâu.

### Kịch bản 3: Hành Động CẮT LỖ (Realized Signal)
- Tiếp nối Kịch bản 2, bot quyết định sửa sai bằng hành động `CLOSE (pos = 0.0)`.
- Tài khoản đang lỗ tổng cộng `10%` từ lúc mở lệnh.
- **Step Reward:** Bot đứng ngoài -> $0.0$
- **Realized Reward:** Thường/phạt chốt lệnh (Lỗ 10% * chiều Long)
  $1.5 \times (-0.10) \times 15.0 = -2.25$
- **Clip Mềm:** Output $-2.25$ bị clip lại ở mức giới hạn dưới là `-1.5`.
- **Tổng Reward Trả Về:** **-1.5**
- *Hành vi bot học được:* Đây là điểm đau nhất. Bot nhận được điểm phạt CỰC LỚN (-1.5) so với khi chỉ hold (-0.168). Thuật toán RL sẽ update ngược Weight để lần sau bot không phạm vào chuỗi lệnh dẫn đến kết quả cắt lỗ tồi tệ này, hoặc ép nó phải cắt lỗ từ khi mới âm 2% (Realized = -0.45).

### Kịch bản 4: Chốt Lời Thành Công
- Bot vào `Short (-1.0)`, giá sập `+5%`. Bot chốt lệnh `CLOSE (pos = 0.0)`.
- **Realized Reward:** $1.5 \times (+0.05) \times 15.0 = +1.125$
- *Hành vi bot học được:* Một phần thưởng "Jackpot" rất lớn (+1.125) sẽ củng cố mạng nơ ron tin rằng hành động MỞ SHORT + HOLD + CLOSE tại thời điểm đó là một kiệt tác, gia tăng xác suất lặp lại hành vi này trong tương lai.

---

## Tổng Kết

Bộ thông số mới `scaling=15, beta=0.12, clip=1.5` hoàn toàn không phải là "magic numbers" đoán mò. Chúng được thiết kế tinh xảo để:
1. Đảm bảo Gradient ổn định (nhờ Standard Deviation nhỏ, có ngắt Clip).
2. Xử lý triệt để việc Hold lệnh gồng lỗ (nhờ DD Penalty ở ngưỡng 5%).
3. Khuyến khích chốt lời chủ động (Signal Realized Reward cực mạnh).

Kết hợp cùng dữ liệu chất lượng, bộ Reward v2 này tạo ra một vòng lặp 피드백 (feedback loop) toán học khép kín hoàn hảo.
