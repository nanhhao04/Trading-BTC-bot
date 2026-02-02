import numpy as np


class ActionPPO:
    """
    Xử lý Action Space liên tục (Continuous) cho PPO.
    Space: Box(low=-1, high=1, shape=(1,))

    Ý nghĩa giá trị action (a):
    a > 0: Tỷ trọng LONG (VD: 0.5 là Long 50% vốn)
    a < 0: Tỷ trọng SHORT (VD: -1.0 là Short 100% vốn)
    a ~ 0: Neutral (Cash)
    """

    def __init__(self, fee_rate=0.0004, threshold=0.1):
        self.fee_rate = fee_rate
        self.threshold = threshold  # Vùng đệm để tránh nhiễu quanh 0

    def step(self, action_val, current_pos_pct, current_price):
        """
        Args:
            action_val (float): Giá trị output từ mạng PPO (khoảng -1 đến 1)
            current_pos_pct (float): Tỷ trọng hiện tại (-1.0 đến 1.0).
                                     VD: Đang Long Full = 1.0, Đang Short nửa = -0.5
            current_price (float): Giá

        Returns:
            new_pos_pct (float): Tỷ trọng mới
            fee (float): Phí giao dịch phát sinh
            trade_type (str): 'BUY', 'SELL', hoặc 'HOLD'
        """
        target_pct = np.clip(action_val, -1.0, 1.0)

        if abs(target_pct) < self.threshold:
            target_pct = 0.0

        delta = target_pct - current_pos_pct  # lượng giao dịch
        if abs(delta) < 0.1:
            return current_pos_pct, 0.0, 'HOLD'

        fee_pct = abs(delta) * self.fee_rate # Phí giao dịch

        trade_type = 'BUY' if delta > 0 else 'SELL'

        return target_pct, fee_pct, trade_type

    def normalize_action(self, action):
        # Hàm tiện ích nếu cần unscale (thường PPO tự xử lý Tanh ra -1 đến 1 rồi)
        return np.clip(action, -1, 1)