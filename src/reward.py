import numpy as np


class RewardHandler:
    def __init__(self, scaling, alpha, beta, holding_penalty):
        self.scaling = scaling
        self.alpha = alpha  # Hệ số thưởng cho Realized Reward (Chốt lời)
        self.beta = beta  # Hệ số phạt Drawdown
        self.holding_penalty = holding_penalty

        # Biến trạng thái
        self.max_net_worth = 0.0
        self.entry_price = None
        self.prev_position = 0.0

    def reset(self, initial_net_worth):
        self.max_net_worth = initial_net_worth
        self.entry_price = None
        self.prev_position = 0.0

    def calculate(self, net_worth, current_price, past_price, position, action_type, trend_flag):
        # 1. Cập nhật Max Net Worth (Để tính Drawdown)
        if net_worth > self.max_net_worth:
            self.max_net_worth = net_worth

        # 2. Step Reward (Lợi nhuận tạm tính từng bước)
        # Log return tự nhiên
        log_return = np.log(current_price / past_price)
        step_reward = position * log_return

        # 3. Holding Cost (Phạt giữ lệnh)
        risk_cost = abs(position ** 2) * self.holding_penalty

        # 4. Realized Reward (Thưởng khi Chốt lời / Phạt cắt lỗ)
        realized_reward = 0.0

        # -- Logic phát hiện MỞ vị thế (Từ 0 -> Có lệnh) --
        if self.prev_position == 0 and position != 0:
            self.entry_price = current_price

        if self.prev_position != 0 and position == 0 and self.entry_price is not None:

            trade_return = np.log(current_price / self.entry_price)
            direction = 1 if self.prev_position > 0 else -1
            realized_return = trade_return * direction

            # Nhân với Alpha để khuếch đại phần thưởng chốt lời
            realized_reward = self.alpha * realized_return

            # Reset entry price
            self.entry_price = None

        # 5. Drawdown Penalty (Phạt sụt giảm)
        current_dd = (self.max_net_worth - net_worth) / self.max_net_worth
        dd_penalty = self.beta * current_dd

        # 6. Trend Scaling (Điều chỉnh theo xu hướng)
        trend_factor = 1.0

        # Thưởng khi thuận xu hướng
        if trend_flag == 1.0 and position > 0:  # Uptrend + Long
            trend_factor = 1.2
        elif trend_flag == 0.0 and position < 0:  # Downtrend + Short
            trend_factor = 1.6

        # Phạt nhẹ khi ngược xu hướng (Tùy chọn, để bot không quá lì lợm)
        elif trend_flag == 1.0 and position < 0:  # Uptrend + Short
            trend_factor = 0.8
        elif trend_flag == 0.0 and position > 0:  # Downtrend + Long
            trend_factor = 0.8

        # 7. Tổng hợp Reward
        raw_reward = (step_reward + realized_reward - dd_penalty - risk_cost) * trend_factor
        total_reward = raw_reward * self.scaling
        total_reward = np.clip(total_reward, -10, 10)

        # QUAN TRỌNG: Cập nhật vị thế cũ cho bước sau
        self.prev_position = position

        return total_reward, {
            'step_reward': step_reward,
            'realized_reward': realized_reward,
            'dd_penalty': dd_penalty,
            'trend_factor': trend_factor,
            'max_drawdown': current_dd
        }