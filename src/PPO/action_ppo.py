import numpy as np


class ActionPPO:
    """
    ActionPPO — Discrete action space theo đúng paper.

    Target Position mapping A(t) ∈ {-1, 0, +1}:
        0 → SHORT  (position = -1.0)
        1 → FLAT   (position =  0.0) ← đóng lệnh, đứng ngoài thị trường
        2 → LONG   (position = +1.0)

    Ưu điểm:
    - HOLD tự nhiên: nếu đang LONG và chọn LONG tiếp → delta ≈ 0 → không tốn phí
    - FLAT cho phép bot đứng ngoài thị trường khi sideways → Flat % > 0%
    - Không bị lock-in bắt buộc phải luôn Long hoặc Short như trước
    - PPO vẫn dùng được với Discrete(3) action space (CategoricalDistribution)
    """

    ACTION_SHORT = 0
    ACTION_FLAT  = 1  # Đóng lệnh về 0.0 (Fix: trước đây là HOLD giữ nguyên)
    ACTION_LONG  = 2

    def __init__(self, fee_rate=0.00075):
        self.fee_rate = fee_rate

    def step(self, action: int, current_pos_pct: float, current_price: float):
        """
        Trả về (target_pct, fee_rate, trade_type)

        action    : int ∈ {0, 1, 2}
        target_pct: float ∈ {-1.0, 0.0, +1.0}  (vị thế đích mong muốn)

        HOLD xảy ra tự nhiên khi target_pct == current_pos_pct (delta ≈ 0)
        """
        action = int(action)

        if action == self.ACTION_LONG:
            target_pct = 1.0
        elif action == self.ACTION_SHORT:
            target_pct = -1.0
        else:  # ACTION_FLAT — đóng lệnh về Flat (0.0)
            target_pct = 0.0

        delta = target_pct - current_pos_pct

        # HOLD tự nhiên: đã ở đúng vị thế đích → không giao dịch, không tốn phí
        if abs(delta) < 1e-6:
            return current_pos_pct, 0.0, 'HOLD'

        trade_type = 'BUY' if delta > 0 else 'SELL'
        return target_pct, self.fee_rate, trade_type
