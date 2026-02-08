import numpy as np


class ActionPPO:


    def __init__(self, fee_rate=0.0004, threshold=0.1):
        self.fee_rate = fee_rate
        self.threshold = threshold  # Vùng đệm để tránh nhiễu quanh 0

    def step(self, action_val, current_pos_pct, current_price):

        target_pct = np.clip(action_val, -1.0, 1.0)
        if abs(target_pct) < self.threshold:
            target_pct = 0.0

        delta = target_pct - current_pos_pct  # lượng giao dịch
        if abs(delta) < 0.1:
            return current_pos_pct, 0.0, 'HOLD'

        fee_pct = abs(delta) * self.fee_rate # Phí giao dịch

        trade_type = 'BUY' if delta > 0 else 'SELL'

        return target_pct, fee_pct, trade_type

