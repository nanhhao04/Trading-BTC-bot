import yaml
import pandas as pd
import os
import torch
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
import gymnasium as gym

# Import môi trường
from env import BitcoinTradingEnv


class TradeStatsCallback(BaseCallback):
    """
    Callback: in số lượng lệnh BUY/SELL/HOLD và position ratio mỗi rollout.
    Giúp monitor Long-bias trực tiếp trong quá trình train.
    """
    def __init__(self, n_steps: int, n_envs: int, verbose=0):
        super().__init__(verbose)
        self.n_steps    = n_steps
        self.n_envs     = n_envs
        self.buy_count  = 0
        self.sell_count = 0
        self.hold_count = 0
        self.long_bars  = 0
        self.short_bars = 0
        self.flat_bars  = 0
        self.last_print = 0
        self.print_every = n_steps * n_envs  # in sau mỗi 1 rollout

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        for info in infos:
            action = info.get('action', '')
            pos    = info.get('position', 0.0)
            if action == 'BUY':   self.buy_count  += 1
            elif action == 'SELL': self.sell_count += 1
            else:                  self.hold_count += 1
            if   pos >  0.05: self.long_bars  += 1
            elif pos < -0.05: self.short_bars += 1
            else:              self.flat_bars  += 1

        if self.num_timesteps - self.last_print >= self.print_every:
            total_trades = self.buy_count + self.sell_count + self.hold_count
            total_bars   = self.long_bars + self.short_bars + self.flat_bars
            print(f"\n{'='*55}")
            print(f"  TRADE STATS @ step {self.num_timesteps:,}")
            print(f"{'='*55}")
            print(f"  Orders : BUY={self.buy_count:,}  SELL={self.sell_count:,}  HOLD={self.hold_count:,}")
            if total_bars > 0:
                print(f"  Position: Long={self.long_bars/total_bars:.1%}  "
                      f"Short={self.short_bars/total_bars:.1%}  "
                      f"Flat={self.flat_bars/total_bars:.1%}")
            print(f"{'='*55}\n")
            # Reset counters sau mỗi lần in
            self.buy_count = self.sell_count = self.hold_count = 0
            self.long_bars = self.short_bars = self.flat_bars  = 0
            self.last_print = self.num_timesteps
        return True


def load_config():
    # Lấy đường dẫn tuyệt đối đến thư mục chứa file train.py (là thư mục src)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # config.yaml nằm ở thư mục cha của src
    config_path = os.path.join(script_dir, '..', 'config.yaml')
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# Hàm tạo môi trường (Bắt buộc phải tách ra hàm riêng để chạy song song)
def make_env(rank, df_full, df_state, cfg, seed=0):
    def _init():
        env = BitcoinTradingEnv(
            df_full=df_full,
            df_state=df_state,
            model_type=cfg['model_type'],
            initial_balance=cfg['env']['initial_balance'],
            fee_rate=cfg['env']['fee_rate'],
            leverage=cfg['leverage'],
            max_capital_usage=cfg.get('max_capital_usage', 1.0),
            reward_cfg=cfg.get('reward', {}).get(cfg.get('timeframes', '5m'), {}),
            max_episode_steps=cfg['env'].get('max_episode_steps', 500),
        )
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    # 1. Load Config
    cfg = load_config()
    
    # Lấy project root (thư mục cha của src)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    device = cfg['system'].get('device', 'auto')
    print(f"Training on DEVICE: {device.upper()}")

    # 2. Load Dữ liệu - Chuyển sang đường dẫn tuyệt đối
    print("Loading data...")
    data_full_path = os.path.normpath(os.path.join(project_root, 'src', cfg['paths']['data_full']))
    data_state_path = os.path.normpath(os.path.join(project_root, 'src', cfg['paths']['data_state']))
    
    df_full = pd.read_csv(data_full_path)
    df_state = pd.read_csv(data_state_path)

    min_len = min(len(df_full), len(df_state))
    df_full = df_full.iloc[:min_len]
    df_state = df_state.iloc[:min_len]

    n_envs = cfg['system'].get('n_envs', 1)
    print(f"Creating {n_envs} parallel environments...")

    if n_envs > 1:
        # SubprocVecEnv: Chạy trên nhiều core CPU (Đa luồng thực sự)
        env = SubprocVecEnv([make_env(i, df_full, df_state, cfg) for i in range(n_envs)])
    else:
        # DummyVecEnv: Chạy trên 1 luồng (Dành cho debug hoặc máy yếu)
        env = DummyVecEnv([make_env(0, df_full, df_state, cfg)])

    # 4. Khởi tạo Model với tham số 'device' - Chuyển sang đường dẫn tuyệt đối
    model_type = cfg['model_type'].upper()
    # Đảm bảo lưu đúng vào thư mục model bên trong project
    abs_models_dir = os.path.normpath(os.path.join(project_root, 'model'))
    save_dir = os.path.join(abs_models_dir, f"{model_type}_{cfg['timeframes']}_{cfg['project_name']}")

    # 5. Callback & Train
    abs_logs_dir = os.path.normpath(os.path.join(project_root, 'tensorboard_logs'))
    
    if model_type == "PPO":
        model = PPO(
            env=env,
            device=device,
            tensorboard_log=abs_logs_dir,
            seed=cfg['seed'],
            **cfg['ppo_params']
        )
    elif model_type == "DQN":
        model = DQN(
            env=env,
            device=device,
            tensorboard_log=abs_logs_dir,
            seed=cfg['seed'],
            **cfg['dqn_params']
        )

    # 5. Callback & Train
    checkpoint_callback = CheckpointCallback(
        save_freq=cfg['training']['save_interval'] // n_envs,  # Điều chỉnh freq theo số env
        save_path=save_dir,
        name_prefix=f"{model_type}_model"
    )

    print(f"Start training...")
    n_steps = cfg['ppo_params'].get('n_steps', 4096) if model_type == "PPO" else 1000
    trade_cb = TradeStatsCallback(n_steps=n_steps, n_envs=n_envs)
    model.learn(
        total_timesteps=cfg['training']['total_timesteps'],
        callback=[checkpoint_callback, trade_cb],
        tb_log_name=model_type
    )

    model.save(os.path.join(save_dir, "final_model"))
    print("Training finished & Saved.")


if __name__ == "__main__":
    main()