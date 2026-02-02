import yaml
import pandas as pd
import os
import torch
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback
import gymnasium as gym

# Import môi trường
from env import BitcoinTradingEnv




def load_config():
    with open('../config.yaml', 'r', encoding='utf-8') as f: # Thêm encoding='utf-8'
        return yaml.safe_load(f)

# Hàm tạo môi trường (Bắt buộc phải tách ra hàm riêng để chạy song song)
def make_env(rank, df_full, df_state, cfg, seed=0):
    def _init():
        env = BitcoinTradingEnv(
            df_full=df_full,
            df_state=df_state,
            model_type=cfg['model_type'],
            initial_balance=cfg['env']['initial_balance'],
            fee_rate=cfg['env']['fee_rate']
        )
        env.reset(seed=seed + rank)
        return env

    return _init


def main():
    # 1. Load Config
    cfg = load_config()
    device = cfg['system'].get('device', 'auto')

    # Kiểm tra xem có GPU thật không
    #if device == "cuda" and not torch.cuda.is_available():
     #   print("CẢNH BÁO: Bạn chọn 'cuda' nhưng không tìm thấy GPU. Chuyển về 'cpu'.")
      #  device = "cpu"
    print(f"Training on DEVICE: {device.upper()}")

    # 2. Load Dữ liệu
    print("Loading data...")
    df_full = pd.read_csv(cfg['paths']['data_full'])
    df_state = pd.read_csv(cfg['paths']['data_state'])

    min_len = min(len(df_full), len(df_state))
    df_full = df_full.iloc[:min_len]
    df_state = df_state.iloc[:min_len]

    # 3. Khởi tạo ĐA MÔI TRƯỜNG (Vectorized Environment)
    # Đây là chìa khóa để GPU chạy nhanh: Chạy 4-8 env cùng lúc
    n_envs = cfg['system'].get('n_envs', 1)
    print(f"Creating {n_envs} parallel environments...")

    if n_envs > 1:
        # SubprocVecEnv: Chạy trên nhiều core CPU (Đa luồng thực sự)
        env = SubprocVecEnv([make_env(i, df_full, df_state, cfg) for i in range(n_envs)])
    else:
        # DummyVecEnv: Chạy trên 1 luồng (Dành cho debug hoặc máy yếu)
        env = DummyVecEnv([make_env(0, df_full, df_state, cfg)])

    # 4. Khởi tạo Model với tham số 'device'
    model_type = cfg['model_type'].upper()
    save_dir = os.path.join(cfg['paths']['models_dir'], f"{model_type}_{cfg['project_name']}")

    if model_type == "PPO":
        model = PPO(
            env=env,
            device=device,
            tensorboard_log=cfg['paths']['logs_dir'],
            seed=cfg['seed'],
            **cfg['ppo_params']
        )
    elif model_type == "DQN":
        model = DQN(
            env=env,
            device=device,
            tensorboard_log=cfg['paths']['logs_dir'],
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
    model.learn(
        total_timesteps=cfg['training']['total_timesteps'],
        callback=checkpoint_callback,
        tb_log_name=model_type
    )

    model.save(os.path.join(save_dir, "final_model"))
    print("Training finished & Saved.")


if __name__ == "__main__":
    main()