import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class TradingEnv(gym.Env):
    """
    Môi trường giao dịch giả lập (Trading Environment)
    Kế thừa từ gymnasium.Env để tương thích với Stable-Baselines3
    """

    def __init__(self, df, initial_balance=1000, commission_fee=0.0004):
        super(TradingEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.commission_fee = commission_fee  # 0.04% (Phí taker sàn Binance)

        # --- 1. ACTION SPACE (Không gian hành động) ---
        # 0: Hold (Giữ nguyên)
        # 1: Buy (Mua toàn bộ tiền đang có)
        # 2: Sell (Bán toàn bộ coin đang có)
        self.action_space = spaces.Discrete(3)

        # --- 2. OBSERVATION SPACE (Không gian quan sát) ---
        # Bot sẽ nhìn thấy các chỉ số kỹ thuật (RSI, MACD...) trừ các cột không cần thiết
        self.ignore_cols = ['date', 'timestamp', 'open', 'high', 'low', 'volume']
        self.feature_cols = [c for c in df.columns if c not in self.ignore_cols]

        # Khai báo kích thước dữ liệu đầu vào cho Bot
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.feature_cols),), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """
        Hàm reset môi trường về trạng thái ban đầu để bắt đầu lượt chơi mới (Episode)
        """
        super().reset(seed=seed)

        self.current_step = 0
        self.balance = self.initial_balance  # Tiền mặt (USDT)
        self.shares_held = 0  # Số lượng Coin nắm giữ
        self.net_worth = self.initial_balance  # Tổng tài sản = Tiền + (Coin * Giá)
        self.max_net_worth = self.initial_balance

        return self._next_observation(), {}

    def _next_observation(self):
        """
        Lấy dữ liệu thị trường tại thời điểm hiện tại để đưa cho Bot xem
        """
        obs = self.df.iloc[self.current_step][self.feature_cols].values
        return obs.astype(np.float32)

    def step(self, action):
        """
        Thực hiện hành động và trả về kết quả (Reward)
        """
        # Lấy giá hiện tại (Close price)
        current_price = self.df.iloc[self.current_step]['close']

        # --- XỬ LÝ HÀNH ĐỘNG ---
        # Action 1: BUY (Chỉ mua nếu đang cầm Tiền và chưa cầm Coin)
        if action == 1 and self.balance > 0:
            amount_to_invest = self.balance
            # Trừ phí giao dịch
            fee = amount_to_invest * self.commission_fee
            # Tính số coin mua được
            self.shares_held = (amount_to_invest - fee) / current_price
            self.balance = 0  # Đã tiêu hết tiền vào coin

        # Action 2: SELL (Chỉ bán nếu đang cầm Coin)
        elif action == 2 and self.shares_held > 0:
            # Quy đổi coin ra tiền
            amount_received = self.shares_held * current_price
            # Trừ phí giao dịch
            fee = amount_received * self.commission_fee
            self.balance += amount_received - fee
            self.shares_held = 0  # Đã bán hết coin

        # Action 0: Hold (Không làm gì, giữ nguyên trạng thái)

        # --- CẬP NHẬT TRẠNG THÁI ---
        self.current_step += 1

        # Kiểm tra xem đã đi hết dữ liệu chưa
        terminated = self.current_step >= len(self.df) - 1
        truncated = False

        # --- TÍNH TOÁN PHẦN THƯỞNG (REWARD) ---
        # Tổng tài sản hiện tại
        new_net_worth = self.balance + (self.shares_held * self.df.iloc[self.current_step]['close'])

        # Reward = Tiền lãi kiếm được trong bước này
        # Nếu lãi dương -> Thưởng, Lãi âm (Lỗ) -> Phạt
        reward = new_net_worth - self.net_worth

        # Cập nhật net_worth cho bước sau
        self.net_worth = new_net_worth

        # Thông tin thêm để debug
        info = {'net_worth': self.net_worth, 'step': self.current_step}

        return self._next_observation(), reward, terminated, truncated, info

    def render(self, mode='human'):
        """
        Hàm in ra màn hình để theo dõi (Optional)
        """
        print(f"Step: {self.current_step}, Net Worth: {self.net_worth:.2f}")


# --- CHẠY THỬ ĐỂ KIỂM TRA (Unit Test) ---
if __name__ == "__main__":
    try:
        # Load dữ liệu đã xử lý
        df = pd.read_csv("../data/processed/BTCUSDT_1h_features.csv")

        # Khởi tạo môi trường
        env = TradingEnv(df)

        # Reset môi trường
        obs, _ = env.reset()
        print("🔍 Môi trường khởi tạo thành công!")
        print(f"   - Action Space: {env.action_space}")
        print(f"   - Observation Shape: {obs.shape}")

        # Thử chạy 10 bước ngẫu nhiên
        print("\n▶️ Chạy thử 10 bước ngẫu nhiên:")
        for _ in range(10):
            action = env.action_space.sample()  # Chọn hành động bừa (0, 1 hoặc 2)
            obs, reward, done, _, info = env.step(action)
            print(f"   Action: {action} | Reward: {reward:.4f} | Net Worth: {info['net_worth']:.2f}")
            if done: break

        print("\n✅ Môi trường hoạt động TỐT! Sẵn sàng để Train.")

    except FileNotFoundError:
        print("❌ Lỗi: Không tìm thấy file data processed. Hãy chạy 'features.py' trước!")