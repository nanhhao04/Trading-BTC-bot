import numpy as np


class ActionDQN:

    def __init__(self, fee_rate=0.0004):
        self.fee_rate = fee_rate
        self.action_map = {0: 'WAIT', 1: 'LONG', 2: 'SHORT', 3: 'CLOSE'}

    def step(self, action_id, current_pos, current_price):
        """
        Thực hiện hành động và trả về vị thế mới + phí giao dịch.

        Args:
            action_id (int): 0, 1, 2, 3
            current_pos (int): 0 (Neutral), 1 (Long), -1 (Short)
            current_price (float): Giá hiện tại

        Returns:
            new_pos (int): Vị thế mới (0, 1, -1)
            fee (float): Phí giao dịch phát sinh (theo % giá trị lệnh)
            executed (bool): Có giao dịch thực sự diễn ra không?
        """
        new_pos = current_pos
        fee = 0.0
        executed = False

        # --- ACTION 1: LONG ---
        if action_id == 1:
            if current_pos == 0:  # Neutral -> Long
                new_pos = 1
                fee = self.fee_rate  # Phí mở
                executed = True
            elif current_pos == -1:  # Short -> Long (Đảo chiều)
                new_pos = 1
                fee = self.fee_rate * 2  # Phí đóng Short + Phí mở Long
                executed = True
            # Nếu đang Long (1) -> Giữ nguyên, không mất phí

        # --- ACTION 2: SHORT ---
        elif action_id == 2:
            if current_pos == 0:  # Neutral -> Short
                new_pos = -1
                fee = self.fee_rate
                executed = True
            elif current_pos == 1:  # Long -> Short (Đảo chiều)
                new_pos = -1
                fee = self.fee_rate * 2  # Phí đóng Long + Phí mở Short
                executed = True
            # Nếu đang Short (-1) -> Giữ nguyên

        # --- ACTION 3: CLOSE (Take Profit / Cut Loss) ---
        elif action_id == 3:
            if current_pos != 0:  # Đang có lệnh -> Đóng
                new_pos = 0
                fee = self.fee_rate
                executed = True
            # Nếu đang Neutral -> Giữ nguyên (như Wait)

        # --- ACTION 0: WAIT ---
        # Giữ nguyên mọi thứ

        return new_pos, fee, executed

    def get_action_name(self, action_id):
        return self.action_map.get(action_id, "UNKNOWN")