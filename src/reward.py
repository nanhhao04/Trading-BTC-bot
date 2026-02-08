import numpy as np


class RewardHandler:
    def __init__(self, scaling, alpha, beta, holding_penalty):

        self.scaling = scaling
        self.alpha = alpha
        self.beta = beta
        self.holding_penalty = holding_penalty

        # Theo dõi đỉnh tài sản để tính Drawdown
        self.max_net_worth = 0.0

    def reset(self, initial_net_worth):
        self.max_net_worth = initial_net_worth

    def calculate(self,
                  net_worth,
                  current_price,
                  past_price,
                  position,
                  action_type,
                  trend_flag):
        # 1. Cập nhật Max Net Worth (Để tính Drawdown)
        if net_worth > self.max_net_worth:
            self.max_net_worth = net_worth

        # Step Reward (Unrealized PnL)
        log_return = np.log(current_price / past_price)
        step_reward = position * log_return

        risk_cost = abs(position) * self.holding_penalty

        # Realized Reward (Thưởng khi Chốt lời/Cắt lỗ)
        realized_reward = 0.0
        if action_type in ['CLOSE', 'SELL'] and position > 0:
            pass

        # Drawdown Penalty (Phạt sụt giảm) - QUAN TRỌNG NHẤT
        current_dd = (self.max_net_worth - net_worth - risk_cost) / self.max_net_worth
        dd_penalty = self.beta * current_dd

        # Trend Scaling (Điều chỉnh theo xu hướng)
        trend_factor = 1.0
        if trend_flag == 0.0 and position > 0:  # Đang Long trong Downtrend
            trend_factor = 0.5


        raw_reward = (step_reward - dd_penalty) * trend_factor
        total_reward = raw_reward * self.scaling
        total_reward = np.clip(total_reward, -10, 10)

        return total_reward, {
            'step_reward': step_reward,
            'dd_penalty': dd_penalty,
            'trend_factor': trend_factor,
            'max_drawdown': current_dd
        }