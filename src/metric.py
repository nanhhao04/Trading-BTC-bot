import numpy as np
import pandas as pd


class PerformanceTracker:
    def __init__(self, initial_balance=1000.0):
        self.initial_balance = initial_balance
        self.history = []
        self.trades = []  # Lưu PnL của từng lệnh đã đóng
        self.returns = []  # Lưu % lợi nhuận mỗi bước (để tính Sharpe)

        self.current_entry_price = 0
        self.current_entry_balance = 0
        self.is_in_position = False

    def update(self, current_balance, position, current_price):
        # 1. Ghi nhận lịch sử số dư
        self.history.append({
            'balance': current_balance,
            'price': current_price,
            'position': position
        })

        # 2. Tính Return cho bước này (dùng cho Sharpe/Sortino)
        if len(self.history) > 1:
            prev_balance = self.history[-2]['balance']
            step_return = (current_balance - prev_balance) / prev_balance
            self.returns.append(step_return)

        # 3. Theo dõi Trade (Để tính Winrate, Avg Win/Loss)
        # Phát hiện vào lệnh
        if not self.is_in_position and abs(position) > 0.05:
            self.is_in_position = True
            self.current_entry_balance = current_balance

        # Phát hiện đóng lệnh (hoặc đảo chiều)
        elif self.is_in_position and abs(position) < 0.05:
            self.is_in_position = False
            # Tính PnL của lệnh vừa xong
            pnl_amount = current_balance - self.current_entry_balance
            pnl_pct = pnl_amount / self.current_entry_balance
            self.trades.append(pnl_amount)

    def calculate_metrics(self):
        if not self.history or not self.returns:
            return {}

        # Chuyển returns sang numpy array
        returns_np = np.array(self.returns)
        current_balance = self.history[-1]['balance']

        # 1. Total Return
        total_return = (current_balance - self.initial_balance) / self.initial_balance * 100

        # 2. Max Drawdown
        balances = [h['balance'] for h in self.history]
        max_balance = np.maximum.accumulate(balances)
        drawdowns = (max_balance - balances) / max_balance
        max_drawdown = np.max(drawdowns) * 100

        # 3. Sharpe Ratio (Giả sử dữ liệu 1H -> Annualize factor = sqrt(365*24))
        # Risk free rate = 0
        if np.std(returns_np) > 0:
            sharpe = np.mean(returns_np) / np.std(returns_np) * np.sqrt(365 * 24)
        else:
            sharpe = 0

        # 4. Sortino Ratio (Chỉ tính độ lệch chuẩn của returns âm)
        negative_returns = returns_np[returns_np < 0]
        if len(negative_returns) > 0 and np.std(negative_returns) > 0:
            sortino = np.mean(returns_np) / np.std(negative_returns) * np.sqrt(365 * 24)
        else:
            sortino = 0

        # 5. Trade Metrics (Winrate, Profit Factor)
        wins = [t for t in self.trades if t > 0]
        losses = [t for t in self.trades if t <= 0]

        winrate = (len(wins) / len(self.trades) * 100) if self.trades else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 999.0

        return {
            "Total Return": f"{total_return:.2f}%",
            "Max Drawdown": f"{max_drawdown:.2f}%",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Sortino Ratio": f"{sortino:.2f}",
            "Winrate": f"{winrate:.2f}% ({len(wins)}/{len(self.trades)})",
            "Avg Win": f"${avg_win:.2f}",
            "Avg Loss": f"${avg_loss:.2f}",
            "Profit Factor": f"{profit_factor:.2f}"
        }