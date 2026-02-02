import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

# Import các module bạn đã viết
from DQN.action_dqn import ActionDQN
from PPO.action_ppo import ActionPPO
from reward import RewardHandler


class BitcoinTradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, df_full, df_state, model_type='DQN', initial_balance=10000, fee_rate=0.0004):
        super(BitcoinTradingEnv, self).__init__()

        self.df_full = df_full.reset_index(drop=True)
        self.df_state = df_state.reset_index(drop=True)
        self.model_type = model_type
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate

        # --- 1. Cấu hình Action Space ---
        if model_type == 'DQN':
            # 0: Wait, 1: Long, 2: Short, 3: Close
            self.action_space = spaces.Discrete(4)
            self.action_handler = ActionDQN(fee_rate=fee_rate)
            # DQN dùng biến rời rạc: 0 (Neutral), 1 (Long), -1 (Short)
            self.pos_tracker = 0

            # Cấu hình Reward cho DQN (Phạt nặng rủi ro)
            self.reward_handler = RewardHandler(scaling=10.0, alpha=1.0, beta=0.5)

        elif model_type == 'PPO':
            # [-1, 1]: Tỷ trọng vốn
            self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
            self.action_handler = ActionPPO(fee_rate=fee_rate)
            # PPO dùng biến liên tục: -1.0 đến 1.0
            self.pos_tracker = 0.0

            # Cấu hình Reward cho PPO (Thưởng lớn để Gradient rõ)
            self.reward_handler = RewardHandler(scaling=100.0, alpha=0.5, beta=0.2)

        # Cấu hình Observation Space
        # Tổng = 8 features ( 6 + 2 )
        self.obs_shape = (self.df_state.shape[1] + 2,)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.obs_shape, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset các biến tài khoản
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.current_step = 0

        # Reset tracker vị thế
        if self.model_type == 'DQN':
            self.pos_tracker = 0
        else:
            self.pos_tracker = 0.0

        # Reset Reward Handler
        self.reward_handler.reset(self.initial_balance)

        return self._get_observation(), {}

    def step(self, action):
        # 1. Lấy dữ liệu thị trường hiện tại
        current_price = self.df_full.loc[self.current_step, 'close']
        # Lấy giá bước trước để tính PnL thay đổi (nếu cần) hoặc dùng logic Reward
        prev_price = self.df_full.loc[self.current_step - 1, 'close'] if self.current_step > 0 else current_price

        trend_flag = self.df_state.loc[self.current_step, 'I_trend']

        # 2. Thực hiện hành động (Qua Action Handler)
        # Handler sẽ tính toán vị thế mới và phí
        if self.model_type == 'DQN':
            new_pos, fee_rate, executed = self.action_handler.step(action, self.pos_tracker, current_price)
            trade_type_str = self.action_handler.get_action_name(action)
        else:  # PPO
            # Action của PPO là array, lấy phần tử đầu tiên
            new_pos, fee_rate, trade_type_str = self.action_handler.step(action[0], self.pos_tracker, current_price)

        # 3. Tính toán tiền nong (Balance & Net Worth)
        fee_cost = self.net_worth * fee_rate
        self.net_worth -= fee_cost

        # PnL = % thay đổi giá * tỷ trọng vị thế * Tổng tài sản
        price_change_pct = (current_price - prev_price) / prev_price
        pnl = self.net_worth * self.pos_tracker * price_change_pct
        self.net_worth += pnl

        # Cập nhật vị thế mới
        self.pos_tracker = new_pos

        # 4. Tính Reward
        reward, reward_info = self.reward_handler.calculate(
            net_worth=self.net_worth,
            current_price=current_price,
            past_price=prev_price,
            position=self.pos_tracker,
            action_type=trade_type_str,
            trend_flag=trend_flag
        )

        # 5. Chuyển bước tiếp theo
        self.current_step += 1
        done = self.current_step >= len(self.df_full) - 1

        # Điều kiện dừng sớm: Cháy tài khoản (còn dưới 50% vốn)
        if self.net_worth < self.initial_balance * 0.5:
            done = True
            reward = -100  # Phạt cực nặng nếu cháy

        obs = self._get_observation()

        info = {
            'net_worth': self.net_worth,
            'step_reward': reward,
            'action': trade_type_str,
            'position': self.pos_tracker
        }

        return obs, reward, done, False, info

    def _get_observation(self):
        # Lấy state từ file csv đã chuẩn hóa
        market_state = self.df_state.iloc[self.current_step].values

        # Thêm thông tin tài khoản vào state để Bot biết mình đang Long hay Short
        account_state = np.array([
            self.pos_tracker,  # Vị thế hiện tại
            (self.net_worth - self.initial_balance) / self.initial_balance  # Lợi nhuận tích lũy (%)
        ])

        return np.concatenate((market_state, account_state)).astype(np.float32)

    def render(self, mode='human'):
        if self.current_step % 100 == 0:
            print(f"Step: {self.current_step}, Net Worth: {self.net_worth:.2f}, Pos: {self.pos_tracker}")