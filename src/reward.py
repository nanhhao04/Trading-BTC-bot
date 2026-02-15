import numpy as np


class RewardHandler:
    """
    RewardHandler v2 — Đã fix 5 lỗi phân tích định lượng:
      A) DD penalty chỉ kích hoạt khi DD vượt ngưỡng dd_threshold (mặc định 5%)
      B) DD penalty bậc 2 thay vì tuyến tính → nhẹ khi DD nhỏ, nặng dần khi DD lớn
      C) Flat position (|pos| < 0.05) được miễn DD penalty — flat là trung tính
      D) alpha = 1.5 (thay vì 0.3) → realized_reward đủ mạnh để bot học chốt lệnh
      E) scaling và clip được truyền vào từ config, không hardcode
    """

    def __init__(self,
                 scaling: float = 50.0,
                 alpha: float = 1.5,
                 beta: float = 0.15,
                 holding_penalty: float = 0.0003,
                 dd_threshold: float = 0.05,
                 clip_low: float = -5.0,
                 clip_high: float = 5.0):
        self.scaling         = scaling          # Nhân toàn bộ reward cuối
        self.alpha           = alpha            # Hệ số khuếch đại realized reward
        self.beta            = beta             # Hệ số phạt DD (sau khi vượt ngưỡng)
        self.holding_penalty = holding_penalty  # Chi phí giữ lệnh mỗi step
        self.dd_threshold    = dd_threshold     # DD tối thiểu trước khi bắt đầu phạt
        self.clip_low        = clip_low
        self.clip_high       = clip_high

        # Biến trạng thái
        self.max_net_worth = 0.0
        self.entry_price   = None
        self.prev_position = 0.0

    def reset(self, initial_net_worth: float):
        self.max_net_worth = initial_net_worth
        self.entry_price   = None
        self.prev_position = 0.0

    def calculate(self, net_worth, current_price, past_price, position, action_type, trend_flag):
        # ------------------------------------------------------------------
        # 1. Cập nhật Max Net Worth
        # ------------------------------------------------------------------
        if net_worth > self.max_net_worth:
            self.max_net_worth = net_worth

        # ------------------------------------------------------------------
        # 2. Step Reward — log return × position
        # ------------------------------------------------------------------
        log_return  = np.log(current_price / past_price)
        step_reward = position * log_return

        # ------------------------------------------------------------------
        # 3. Holding Cost — phạt giữ lệnh (bậc 2 theo size)
        # ------------------------------------------------------------------
        risk_cost = abs(position ** 2) * self.holding_penalty

        # ------------------------------------------------------------------
        # 4. Realized Reward — thưởng/phạt khi đóng lệnh
        realized_reward = 0.0

        # Phát hiện MỞ vị thế (Neutral → Có lệnh)
        if self.prev_position == 0 and position != 0:
            self.entry_price = current_price

        # Phát hiện ĐÓNG vị thế (Có lệnh → Neutral)
        if self.prev_position != 0 and position == 0 and self.entry_price is not None:
            trade_return     = np.log(current_price / self.entry_price)
            direction        = 1 if self.prev_position > 0 else -1
            realized_return  = trade_return * direction
            realized_reward  = self.alpha * realized_return  # FIX D
            self.entry_price = None

        # 5. Drawdown Penalty
        current_dd = (self.max_net_worth - net_worth) / self.max_net_worth

        if abs(position) < 0.05:  # FIX C: flat = trung tính
            dd_penalty = 0.0
        else:
            effective_dd = max(0.0, current_dd - self.dd_threshold)  # FIX A
            dd_penalty   = self.beta * (effective_dd ** 2)            # FIX B

        # ------------------------------------------------------------------
        # 6. Trend Scaling
        # ------------------------------------------------------------------
        trend_factor = 1.0
        if trend_flag == 1.0 and position > 0:    # Uptrend + Long
            trend_factor = 1.2
        elif trend_flag == 0.0 and position < 0:  # Downtrend + Short
            trend_factor = 1.3
        elif trend_flag == 1.0 and position < 0:  # Uptrend + Short
            trend_factor = 0.8
        elif trend_flag == 0.0 and position > 0:  # Downtrend + Long
            trend_factor = 0.8

        # ------------------------------------------------------------------
        # 7. Tổng hợp Reward — FIX E: scaling + clip từ config
        # ------------------------------------------------------------------
        raw_reward   = ((step_reward + realized_reward) * trend_factor) - dd_penalty - risk_cost
        total_reward = raw_reward * self.scaling
        total_reward = float(np.clip(total_reward, self.clip_low, self.clip_high))

        # Cập nhật vị thế cũ cho bước sau
        self.prev_position = position

        return total_reward, {
            'step_reward':     step_reward,
            'realized_reward': realized_reward,
            'dd_penalty':      dd_penalty,
            'trend_factor':    trend_factor,
            'current_dd':      current_dd,
            'risk_cost':       risk_cost,
        }