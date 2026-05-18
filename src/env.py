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

    def __init__(self, df_full, df_state, model_type='DQN',
                 initial_balance=10000, fee_rate=0.0006, leverage=1,
                 max_capital_usage=1.0, reward_cfg: dict = None,
                 max_episode_steps: int = 500, is_backtest: bool = False):
        super(BitcoinTradingEnv, self).__init__()

        self.df_full = df_full.reset_index(drop=True)
        self.df_state = df_state.reset_index(drop=True)
        self.model_type = model_type
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.leverage = leverage
        self.max_capital_usage  = max_capital_usage
        self.max_episode_steps  = max_episode_steps
        self.episode_step       = 0
        self.long_count         = 0
        self.is_backtest        = is_backtest

        # Funding fee params (luôn âm, áp dụng mỗi funding_period bars)
        rw = reward_cfg or {}
        self.funding_rate    = rw.get('funding_rate',   0.0002)  # 2 bps / 8h
        self.funding_period  = rw.get('funding_period', 8)       # 8 bars = 8h (1h timeframe)
        self.short_count = 0
        self.flat_count = 0

        # FIX #5: transaction_cost (reward signal) ≠ fee_rate (sàn thực tế)
        # abstract_fee dùng transaction_cost từ config (nhỏ hơn) → không overwhelm step_reward
        # fee_cost thực dùng fee_rate → vẫn trừ đúng vào net_worth
        self.transaction_cost = rw.get('transaction_cost', fee_rate)

        # Reward Multiplier: Hệ số đưa biến động Asset về biến động Account
        self.reward_multiplier = self.leverage * self.max_capital_usage

        # Đọc reward params từ config (dùng lại rw đã khai báo trên, không gán lại)
        reward_handler = RewardHandler(
            transaction_cost           = rw.get('transaction_cost',        fee_rate),
            clip_low                   = rw.get('clip_low',                -5.0),
            clip_high                  = rw.get('clip_high',                 5.0),
            scaling                    = rw.get('scaling',                   1.0),  # Khuếch đại reward signal
            # Terminal reward params (Paper Section 2.2)
            terminal_loss_threshold    = rw.get('terminal_loss_threshold',  0.60),
            terminal_loss_penalty      = rw.get('terminal_loss_penalty',  -15.0),
            terminal_profit_multiplier = rw.get('terminal_profit_multiplier', 3.0),
            terminal_loss_multiplier   = rw.get('terminal_loss_multiplier',  -2.0),
            # Short-bonus: cân bằng long-bias khi uptrend
            short_bonus_when_longtrend = rw.get('short_bonus_when_longtrend', 0.0),
        )

        # --- 1. Cấu hình Action Space ---

        if model_type == 'DQN':
            self.action_space   = spaces.Discrete(4)
            self.action_handler = ActionDQN(fee_rate=fee_rate)
            self.pos_tracker    = 0
            self.reward_handler = reward_handler

        elif model_type == 'PPO':
            # Discrete(3): 0=Short, 1=Hold, 2=Long — đúng paper
            self.action_space   = spaces.Discrete(3)
            self.action_handler = ActionPPO(fee_rate=fee_rate)
            self.pos_tracker    = 0.0
            self.reward_handler = reward_handler

        self.obs_shape = (self.df_state.shape[1] + 2,)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.obs_shape, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance        = self.initial_balance
        self.net_worth      = self.initial_balance
        self.episode_step   = 0

        if self.is_backtest:
            self.current_step = 0
        else:
            n         = len(self.df_full)
            max_start = int(n * 0.75)   # giữ 25% cuối cho episode chạy
            closes    = self.df_full['close'].values
            roll_val  = self.np_random.random()

            if roll_val < 0.33:
                # 33% episodes: bắt đầu từ local PEAK → agent thấy downtrend/sideways sau đó
                peaks = np.where(
                    (closes[1:-1] > closes[:-2]) &
                    (closes[1:-1] > closes[2:])
                )[0] + 1
                peaks = peaks[peaks < max_start]
                if len(peaks) > 0:
                    self.current_step = int(self.np_random.choice(peaks))
                else:
                    self.current_step = int(self.np_random.integers(0, max_start))

            elif roll_val < 0.66:
                # 33% episodes: bắt đầu từ local TROUGH → agent thấy uptrend/recovery sau đó
                troughs = np.where(
                    (closes[1:-1] < closes[:-2]) &
                    (closes[1:-1] < closes[2:])
                )[0] + 1
                troughs = troughs[troughs < max_start]
                if len(troughs) > 0:
                    self.current_step = int(self.np_random.choice(troughs))
                else:
                    self.current_step = int(self.np_random.integers(0, max_start))

            else:
                # 34% episodes: random uniform — thấy cả uptrend lẫn sideways
                self.current_step = int(self.np_random.integers(0, max_start))

        if self.model_type == 'DQN':
            self.pos_tracker = 0
        else:
            self.pos_tracker = 0.0

        self.reward_handler.reset(self.initial_balance)
        return self._get_observation(), {}

    def step(self, action):
        price_at_t = self.df_full.loc[self.current_step, 'close']
        
        # Chuyển action thành int thuần túy (numpy array/scalar -> python int) để tránh lỗi unhashable type
        if hasattr(action, 'item'):
            action = action.item()
        action = int(action)

        if self.model_type == 'DQN':
            new_pos, fee_rate, executed = self.action_handler.step(action, self.pos_tracker, price_at_t)
            trade_type_str = self.action_handler.get_action_name(action)
        else:  # PPO discrete
            new_pos, fee_rate, trade_type_str = self.action_handler.step(action, self.pos_tracker, price_at_t)

        # 3. Tính Phí (Thực tế trừ vào Net Worth)
        delta_pos = abs(new_pos - self.pos_tracker)
        fee_cost = (self.net_worth * self.max_capital_usage * delta_pos * fee_rate * self.leverage)
        self.net_worth -= fee_cost
        
        # abstract_fee: dùng transaction_cost (từ config reward.1h, thường nhỏ hơn fee_rate)
        # ✔️ Phân tách có chủ đích:
        #   fee_cost    trừ thật vào net_worth (dùng fee_rate = 7.5 bps thực của sàn)
        #   abstract_fee dùng transaction_cost (1 bps) → penalty reward không overwhelm step_reward
        abstract_fee = delta_pos * self.transaction_cost

        # FIX Bug 2: gán pos_tracker TRƯỚC khi tính holding fee
        # → holding fee tính trên vị thế MỚI (new_pos), không phải vị thế cũ
        self.pos_tracker = new_pos

        # FIX Bug #3: tăng episode_step TRƯỚC khi kiểm tra funding
        # → episode_step=0 không bao giờ bị tính holding_fee (vừa mở lệnh xong)
        self.current_step  += 1
        self.episode_step  += 1

        # Funding fee: cấu hình bi quan (luôn trừ phí cho cả hai chiều Long/Short để đảm bảo an toàn tuyệt đối khi chạy thực tế)
        # Điều kiện: episode_step >= 1 (đã qua bar đầu) VÀ chia hết funding_period
        holding_fee     = 0.0   # asset-level penalty (cho reward/abstract_fee - luôn dương để làm penalty)
        holding_fee_usd = 0.0   # USD (cho net_worth thực tế - luôn trừ)
        if (self.episode_step % self.funding_period == 0) and (abs(self.pos_tracker) > 0.05):
            # 1. Chi phí phạt ảo (abstract penalty) để huấn luyện Bot - luôn phạt giữ lệnh lâu để tránh bias
            holding_fee     = abs(self.pos_tracker) * self.funding_rate
            abstract_fee   += holding_fee

            # 2. Chi phí thực tế (real USD) trừ vào net_worth: luôn trừ cho cả hai vị thế để an toàn nhất
            holding_fee_usd = self.net_worth * self.max_capital_usage * holding_fee * self.leverage
            self.net_worth -= holding_fee_usd

        # Episode kết thúc: hết data HOẶC đã chạy đủ max_episode_steps
        done = (self.current_step >= len(self.df_full) - 1) or \
               (self.episode_step >= self.max_episode_steps)

        price_at_t1 = self.df_full.loc[self.current_step, 'close']

        # 6. Tính PnL (Account-level)
        price_change_pct = (price_at_t1 - price_at_t) / price_at_t
        pnl = self.net_worth * self.max_capital_usage * self.pos_tracker * price_change_pct * self.leverage
        self.net_worth += pnl

        # 9. Liquidation Check (Paper Section 2.2 — "agent uses up 70% of capital")
        # liquidation_floor = initial * (1 - terminal_loss_threshold)
        # Ví dụ 1h: threshold=0.60 → floor = 10000 * 0.40 = 4000 USD
        liquidation_floor = self.initial_balance * (1.0 - self.reward_handler.terminal_loss_threshold)
        if self.net_worth <= liquidation_floor:
            done   = True
            # Dùng terminal_loss_penalty (− 15.0) — đây là "large negative" theo paper.
            # Không dùng liquidation_penalty riêng biệt để thống nhất với terminal reward.
            reward = self.reward_handler.terminal_loss_penalty
            reward_info = {}
        else:
            # 7. Tính Reward v5 (Gửi thêm reward_multiplier và log_return_trend)
            # Trend detection: tính 5-bar MA của returns để detect uptrend
            window = 5
            start_idx = max(0, self.current_step - window)
            recent_closes = self.df_full.loc[start_idx:self.current_step, 'close'].values
            
            log_return_trend = 0.0
            if len(recent_closes) > 1:
                # Compute log-returns safely: handle inf/nan
                with np.errstate(divide='ignore', invalid='ignore'):
                    recent_returns = np.diff(np.log(recent_closes))
                # Replace inf/nan with 0
                recent_returns = np.nan_to_num(recent_returns, nan=0.0, posinf=0.0, neginf=0.0)
                log_return_trend = np.mean(recent_returns)
                # Extra safety: clip to reasonable range
                log_return_trend = np.clip(float(log_return_trend), -0.1, 0.1)
            
            reward, reward_info = self.reward_handler.calculate(
                net_worth=self.net_worth,
                current_price=price_at_t1,
                past_price=price_at_t,
                position=self.pos_tracker,
                action_type=trade_type_str,
                abstract_fee=abstract_fee,
                reward_multiplier=self.reward_multiplier,
                log_return_trend=log_return_trend
            )

            # 7b. Terminal Reward: cộng thêm khi episode kết thúc bình thường (hết data)
            if done:
                term_r = self.reward_handler.terminal_reward(
                    net_worth=self.net_worth,
                    initial_balance=self.initial_balance
                )
                reward += term_r

        # 8. Cập nhật thống kê
        if self.pos_tracker > 0.05:
            self.long_count += 1
        elif self.pos_tracker < -0.05:
            self.short_count += 1
        else:
            self.flat_count += 1

        obs = self._get_observation()
        info = {
            'net_worth':   self.net_worth,
            'step_reward': reward,
            'action':      trade_type_str,
            'position':    self.pos_tracker,
            'fee':         fee_cost,         # Phí mở/đóng lệnh (USD)
            'holding_fee': holding_fee_usd,  # Funding fee (USD) — 0 nếu không phải kỳ funding
            'pnl':         pnl,
        }

        if done:
            total = self.long_count + self.short_count + self.flat_count
            if total > 0:
                print("\n===== POSITION STATS =====")
                print(f"Long  : {self.long_count / total:.2%}")
                print(f"Short : {self.short_count / total:.2%}")
                print(f"Flat  : {self.flat_count / total:.2%}")
                print("==========================\n")
            self.long_count = 0
            self.short_count = 0
            self.flat_count = 0

        return obs, reward, done, False, info

    def _get_observation(self):
        market_state = self.df_state.iloc[self.current_step].values
        account_state = np.array([
            self.pos_tracker,
            (self.net_worth - self.initial_balance) / self.initial_balance
        ])
        obs = np.concatenate((market_state, account_state)).astype(np.float32)
        # Safety: replace any NaN/inf with 0
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs

    def render(self, mode='human'):
        if self.current_step % 100 == 0:
            print(f"Step: {self.current_step}, Net Worth: {self.net_worth:.2f}, Pos: {self.pos_tracker}")