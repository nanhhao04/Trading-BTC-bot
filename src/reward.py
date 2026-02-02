import numpy as np


class RewardHandler:
    """
    Module quản lý hàm thưởng (Reward Shaping) cho Bitcoin Trading.
    Mục tiêu: Tối đa hóa lợi nhuận nhưng phạt nặng rủi ro (Drawdown).
    """

    def __init__(self, scaling, alpha, beta):
        """
        Args:
            scaling (float): Phóng đại Reward để mạng Nơ-ron dễ học (BTC biến động nhỏ 0.01 -> nhân 100 thành 1.0).
            alpha (float): Trọng số thưởng cho việc CHỐT LỜI (Realized PnL).
            beta (float): Trọng số phạt RỦI RO (Drawdown).
        """
        self.scaling = scaling
        self.alpha = alpha
        self.beta = beta

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
        """
        Tính toán Reward tại mỗi bước (Step).

        Args:
            net_worth (float): Tổng tài sản hiện tại (Balance + Unrealized PnL).
            current_price (float): Giá BTC hiện tại.
            past_price (float): Giá BTC bước trước.
            position (float): Vị thế hiện tại (-1.0 đến 1.0).
            action_type (str): 'HOLD', 'BUY', 'SELL', 'CLOSE'.
            trend_flag (float): 1.0 (Uptrend) hoặc 0.0 (Downtrend).

        Returns:
            total_reward (float): Giá trị thưởng cuối cùng.
            info (dict): Các thành phần để log.
        """

        # 1. Cập nhật Max Net Worth (Để tính Drawdown)
        if net_worth > self.max_net_worth:
            self.max_net_worth = net_worth

        # -----------------------------------------------------------
        # THÀNH PHẦN 1: Step Reward (Unrealized PnL - Lợi nhuận từng bước)
        # -----------------------------------------------------------
        # Dùng Log Return để có tính cộng tính và đối xứng tốt hơn % thường
        log_return = np.log(current_price / past_price)

        # Nếu đang Long: Giá tăng -> Thưởng. Nếu đang Short: Giá giảm -> Thưởng.
        # Position từ -1 đến 1.
        step_reward = position * log_return

        # THÀNH PHẦN 2: Realized Reward (Thưởng khi Chốt lời/Cắt lỗ)
        # Khuyến khích Bot hiện thực hóa lợi nhuận thay vì "gồng" mãi
        realized_reward = 0.0
        if action_type in ['CLOSE', 'SELL'] and position > 0:  # Đóng Long
            # Logic này được xử lý kỹ hơn trong Env, ở đây ta nhận tín hiệu từ thay đổi net_worth
            pass

        # Tuy nhiên, để đơn giản cho RL, ta coi sự thay đổi ròng của Net Worth
        # là tổng hợp của cả Unrealized và Realized.
        # Ta nhấn mạnh vào Realized bằng cách thêm trọng số alpha nếu có giao dịch đóng.

        # -----------------------------------------------------------
        # THÀNH PHẦN 3: Drawdown Penalty (Phạt sụt giảm) - QUAN TRỌNG NHẤT
        # -----------------------------------------------------------
        # Drawdown càng sâu, phạt càng nặng
        current_dd = (self.max_net_worth - net_worth) / self.max_net_worth
        dd_penalty = self.beta * current_dd

        # -----------------------------------------------------------
        # THÀNH PHẦN 4: Trend Scaling (Điều chỉnh theo xu hướng)
        # -----------------------------------------------------------
        # Nếu đang Downtrend (trend_flag=0) mà đi Long -> Giảm reward (hoặc phạt nặng hơn)
        # Mục đích: Dạy Bot "Don't fight the trend"
        trend_factor = 1.0
        if trend_flag == 0.0 and position > 0:  # Đang Long trong Downtrend
            trend_factor = 0.5  # Giảm 50% lợi nhuận kiếm được (nếu có) để Bot thấy không đáng

        # -----------------------------------------------------------
        # TỔNG HỢP
        # -----------------------------------------------------------
        # Công thức: (Lợi nhuận - Phạt rủi ro) * Hệ số xu hướng

        raw_reward = (step_reward - dd_penalty) * trend_factor
        total_reward = raw_reward * self.scaling

        # Clipping (Cắt bớt) để tránh Reward quá lớn làm nổ Gradient (Exploding Gradient)
        # Giới hạn trong khoảng [-10, 10]
        total_reward = np.clip(total_reward, -10, 10)

        return total_reward, {
            'step_reward': step_reward,
            'dd_penalty': dd_penalty,
            'trend_factor': trend_factor,
            'max_drawdown': current_dd
        }