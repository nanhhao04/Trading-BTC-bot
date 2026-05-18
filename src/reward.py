import numpy as np


class RewardHandler:
    """
    RewardHandler — Theo đúng công thức Paper (Eq. 8)

    Running Reward (per bar):
        R(t) = r(t) * A(t) - |A(t) - A(t-1)| * C

        r(t)          : log-return của asset tại bước t
        A(t)          : vị thế tại t ∈ {-1, 0, +1} (short / flat / long)
        C             : transaction cost (basis points)
        |A(t)-A(t-1)| : penalty khi đổi chiều giao dịch

    Terminal Reward (Paper Section 2.2):
        - Mất >= terminal_loss_threshold vốn → terminal_loss_penalty (large negative)
        - Kết thúc có lãi                  → terminal_profit_multiplier * portfolio_return
        - Kết thúc lỗ (chưa đến ngưỡng)   → terminal_loss_multiplier  * |portfolio_return|

    1h calibration:
        - C (transaction_cost) = 0.00075 (7.5 bps)
        - terminal_loss_threshold = 0.60  (mất 60% vốn → dừng, chặt hơn 70% của daily)
    """

    POS_THRESHOLD = 0.05  # Ngưỡng "có vị thế" (tránh float == 0 bug)

    def __init__(self,
                 transaction_cost: float = 0.00075,    # C trong paper (7.5 bps cho 1h)
                 clip_low: float = -5.0,
                 clip_high: float = 5.0,
                 scaling: float = 1.0,                 # Hệ số khuếch đại reward signal (1h: ~150.0)
                 # Terminal reward params (Paper Section 2.2)
                 terminal_loss_threshold: float = 0.60,    # Mất >= X% vốn → phạt nặng
                 terminal_loss_penalty: float = -15.0,     # Large negative (dùng cho cả liquidation)
                 terminal_profit_multiplier: float = 3.0,  # Hệ số thưởng khi có lãi
                 terminal_loss_multiplier: float = -2.0,   # Hệ số phạt khi lỗ
                 short_bonus_when_longtrend: float = 0.0,  # Thưởng short signal để cân bằng long-bias
                 ):
        self.transaction_cost               = transaction_cost
        self.clip_low                       = clip_low
        self.clip_high                      = clip_high
        self.scaling                        = scaling
        self.terminal_loss_threshold        = terminal_loss_threshold
        self.terminal_loss_penalty          = terminal_loss_penalty
        self.terminal_profit_multiplier     = terminal_profit_multiplier
        self.terminal_loss_multiplier       = terminal_loss_multiplier
        self.short_bonus_when_longtrend     = short_bonus_when_longtrend

        # Trạng thái nội bộ
        self.prev_position = 0.0

    def reset(self, initial_net_worth: float):
        self.prev_position = 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _is_flat(self, pos: float) -> bool:
        return abs(pos) < self.POS_THRESHOLD

    # ------------------------------------------------------------------
    # Main calculate — R(t) = r(t)*A(t) - |A(t)-A(t-1)|*C
    # ------------------------------------------------------------------
    def calculate(self, net_worth, current_price, past_price, position,
                  action_type, abstract_fee=0.0, reward_multiplier=1.0, log_return_trend=None):
        """
        Tính running reward theo công thức paper (Eq. 8) mở rộng:

        R(t) = r(t) * A(t)  -  |A(t) - A(t-1)| * C  +  short_bonus*(if uptrend)

        Đúng Eq.8 paper — không thêm baseline.
        Short-bonus: khi market uptrend (log_return_trend > 0), thưởng SHORT action
        để cân bằng long-bias tự nhiên trong uptrend.
        """

        # r(t): log-return của asset tại bước t (Eq.8)
        # Safety: handle edge cases (price=0, etc)
        if current_price <= 0 or past_price <= 0:
            log_return = 0.0
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                log_return = np.log(current_price / past_price)
            log_return = float(np.nan_to_num(log_return, nan=0.0, posinf=0.0, neginf=0.0))

        # r(t) * A(t) * reward_multiplier: P&L ở account-level
        step_reward = position * log_return * reward_multiplier
        step_reward = float(np.nan_to_num(step_reward, nan=0.0))

        # |A(t) - A(t-1)| * C: transaction cost penalty
        # FIX Scale Mismatch: nhân với reward_multiplier để đưa về account-level
        # Trước đây abstract_fee ở asset-level → bị phóng đại gấp 5x so với step_reward
        # Sau fix: cả hai cùng scale → agent không còn sợ giao dịch
        cost_penalty = abstract_fee * reward_multiplier
        cost_penalty = float(np.nan_to_num(cost_penalty, nan=0.0))

        # Short-bonus: khi market uptrend, thưởng SHORT (-1) signal
        # Giúp agent học Short counter-trend thay vì always Long
        short_bonus = 0.0
        if self.short_bonus_when_longtrend > 0 and log_return_trend is not None:
            log_return_trend = float(np.nan_to_num(log_return_trend, nan=0.0))
            # Nếu uptrend (positive return trend) và position là SHORT (-1.0)
            # → reward = -short_bonus (= large negative), nên ta thêm bonus để SHORT có advantage
            if log_return_trend > 0 and position < -0.5:
                short_bonus = self.short_bonus_when_longtrend
            elif log_return_trend > 0 and position > 0.5:
                # Penalize LONG thêm khi uptrend → avoid pure long-bias
                short_bonus = -self.short_bonus_when_longtrend * 0.5

        # R(t) = r_excess * A(t) - |A(t)-A(t-1)| * C + short_bonus
        # Nhân thêm scaling factor để khuếch đại tín hiệu (giúp PPO Value Network hội tụ nhanh)
        raw_reward = (step_reward - cost_penalty + short_bonus) * self.scaling
        raw_reward = float(np.nan_to_num(raw_reward, nan=0.0))

        # Clip để tránh gradient explosion
        total_reward = float(np.clip(raw_reward, self.clip_low, self.clip_high))

        # Final NaN check
        if np.isnan(total_reward):
            total_reward = 0.0

        # Cập nhật trạng thái
        self.prev_position = position

        return total_reward, {
            'step_reward':  step_reward,
            'cost_penalty': cost_penalty,
            'short_bonus':  short_bonus,
            'raw_reward':   raw_reward,
            'total_reward': total_reward,
            'holding_fee_lump': 0.0,
        }

    # ------------------------------------------------------------------
    # Terminal Reward (Paper Section 2.2)
    # ------------------------------------------------------------------
    def terminal_reward(self, net_worth: float, initial_balance: float) -> float:
        """
        Tính terminal reward khi kết thúc episode (done=True).

        Paper Section 2.2:
          - Mất >= terminal_loss_threshold vốn → terminal_loss_penalty (large negative)
          - Kết thúc có lãi                   → terminal_profit_multiplier * return
          - Kết thúc lỗ (chưa đến ngưỡng)    → terminal_loss_multiplier * |return|
        """
        loss_fraction = (initial_balance - net_worth) / initial_balance  # > 0 khi lỗ

        # Trường hợp 1: Mất >= threshold → phạt nặng
        if loss_fraction >= self.terminal_loss_threshold:
            return self.terminal_loss_penalty

        portfolio_return = (net_worth - initial_balance) / initial_balance

        # Trường hợp 2: Có lãi → thưởng multiplier * return
        if portfolio_return > 0:
            return self.terminal_profit_multiplier * portfolio_return

        # Trường hợp 3: Lỗ nhưng chưa đến ngưỡng → phạt nhẹ
        return self.terminal_loss_multiplier * abs(portfolio_return)