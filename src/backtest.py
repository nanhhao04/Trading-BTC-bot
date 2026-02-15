"""
backtest.py — Backtest so sanh 2 model tren data thang 3/2026

Su dung:
    cd src
    python backtest.py --model_type PPO         # Chi backtest model hien tai
    python backtest.py --model_type PPO --compare  # So sanh old vs new reward

Dau ra:
    backtest_results/
        backtest_YYYY-MM-DD_HH-MM.csv     <- Log tung buoc
        trade_log_YYYY-MM-DD_HH-MM.csv    <- Tung giao dich
        summary_YYYY-MM-DD_HH-MM.csv      <- Metrics tong hop
"""

import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
from stable_baselines3 import PPO, DQN

# ---- Path setup ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from env import BitcoinTradingEnv

CFG_PATH         = os.path.join(SCRIPT_DIR, '..', 'config.yaml')
BACKTEST_DATA_1H = os.path.join(SCRIPT_DIR, '..', 'data', 'processed', 'BTCUSDT_1h_mar2026_features_full.csv')
BACKTEST_DATA_15M= os.path.join(SCRIPT_DIR, '..', 'data', 'processed', 'BTCUSDT_15m_mar2026_features_full.csv')
BACKTEST_DATA_5M = os.path.join(SCRIPT_DIR, '..', 'data', 'processed', 'BTCUSDT_5m_mar2026_features_full.csv')
OUT_DIR          = os.path.join(SCRIPT_DIR, '..', 'backtest_results')
STATE_COLS       = ['Norm_Close', 'RSI14', 'Volatility', 'MACD', 'SMA_Dist', 'I_trend']

# Reward params cu (truoc khi fix) de so sanh
OLD_REWARD_CFG = {
    'scaling':         8.0,
    'alpha':           0.3,
    'beta':            0.6,
    'holding_penalty': 0.003,
    'dd_threshold':    0.0,    # khong co nguong -> phat ngay tu dau
    'clip_low':       -10.0,
    'clip_high':       10.0,
}


def load_config() -> dict:
    with open(CFG_PATH, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_model(cfg: dict, model_type: str):
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
    raw_path = cfg['paths']['ppo_dir'] if model_type == 'PPO' else cfg['paths']['dqn_dir']

    # Resolve relative path from project root (not from src/)
    path = os.path.normpath(os.path.join(PROJECT_ROOT, raw_path.lstrip('../')))
    # Fallback: try as-is if above doesn't work
    if not os.path.exists(path) and not os.path.exists(path + '.zip'):
        path = os.path.normpath(os.path.join(SCRIPT_DIR, raw_path))

    if not os.path.exists(path) and os.path.exists(path + '.zip'):
        path += '.zip'
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}\n  Config path: {raw_path}")

    print(f"  Loading {model_type} model: {path}")
    return PPO.load(path) if model_type == 'PPO' else DQN.load(path)



def load_backtest_data(timeframe: str) -> tuple:
    """Tra ve (df_full, df_state)"""
    if timeframe == '1h':
        full_path = BACKTEST_DATA_1H
    elif timeframe == '15m':
        full_path = BACKTEST_DATA_15M
    elif timeframe == '5m':
        full_path = BACKTEST_DATA_5M
    else:
        raise ValueError(f"Timeframe khong hop le: {timeframe}. Chon '1h', '15m' hoac '5m'.")

    state_path = full_path.replace('features_full', 'state')

    if not os.path.exists(full_path):
        raise FileNotFoundError(
            f"Khong tim thay backtest data: {full_path}\n"
            f"Hay chay: python src/data/preprocess_mar2026.py"
        )

    df_full  = pd.read_csv(full_path)
    df_state = pd.read_csv(state_path)

    # Dong bo do dai
    min_len  = min(len(df_full), len(df_state))
    df_full  = df_full.iloc[:min_len].reset_index(drop=True)
    df_state = df_state.iloc[:min_len].reset_index(drop=True)

    print(f"  Backtest data ({timeframe}): {len(df_full)} nen | "
          f"{pd.to_datetime(df_full['timestamp'].iloc[0], unit='ms')} -> "
          f"{pd.to_datetime(df_full['timestamp'].iloc[-1], unit='ms')}")
    return df_full, df_state


# ===================================================================
#  CORE BACKTEST RUNNER
# ===================================================================
def run_backtest(model, cfg: dict, df_full: pd.DataFrame, df_state: pd.DataFrame,
                 label: str, reward_cfg: dict = None) -> tuple:
    """
    Chay 1 episode backtest.
    Tra ve:
        step_log : DataFrame log tung buoc
        trade_log: DataFrame log tung giao dich da dong
    """
    env = BitcoinTradingEnv(
        df_full        = df_full,
        df_state       = df_state,
        model_type     = cfg['model_type'],
        initial_balance= cfg['env']['initial_balance'],
        fee_rate       = cfg['env']['fee_rate'],
        reward_cfg     = reward_cfg,
    )

    obs, _ = env.reset(seed=42)

    step_rows  = []
    trade_rows = []

    entry_price  = None
    entry_step   = None
    entry_pos    = None
    prev_pos     = 0.0

    done = False
    step = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)

        current_price = df_full.loc[step, 'close'] if step < len(df_full) else df_full.loc[len(df_full)-1, 'close']
        current_pos   = info['position']
        net_worth     = info['net_worth']
        action_name   = info['action']

        step_rows.append({
            'step':        step,
            'price':       current_price,
            'position':    current_pos,
            'net_worth':   net_worth,
            'reward':      reward,
            'action':      action_name,
        })

        # ---- Phat hien giao dich da dong ----
        if abs(prev_pos) > 0.05 and abs(current_pos) < 0.05 and entry_price is not None:
            direction = 'LONG' if entry_pos > 0 else 'SHORT'
            pnl_pct   = (current_price - entry_price) / entry_price * (1 if entry_pos > 0 else -1)
            pnl_usd   = cfg['env']['initial_balance'] * abs(entry_pos) * pnl_pct
            hold_bars = step - entry_step

            trade_rows.append({
                'label':       label,
                'entry_step':  entry_step,
                'exit_step':   step,
                'hold_bars':   hold_bars,
                'direction':   direction,
                'entry_price': entry_price,
                'exit_price':  current_price,
                'pnl_pct':     pnl_pct * 100,
                'pnl_usd':     pnl_usd,
                'win':         pnl_usd > 0,
            })
            entry_price = None
            entry_step  = None

        if abs(prev_pos) < 0.05 and abs(current_pos) > 0.05:
            entry_price = current_price
            entry_step  = step
            entry_pos   = current_pos

        prev_pos = current_pos
        step    += 1

    return pd.DataFrame(step_rows), pd.DataFrame(trade_rows)


# ===================================================================
#  METRICS CALCULATION
# ===================================================================
def compute_metrics(step_log: pd.DataFrame, trade_log: pd.DataFrame,
                    label: str, initial_balance: float) -> dict:
    balances   = step_log['net_worth'].values
    final_bal  = balances[-1]
    returns_pct= np.diff(balances) / balances[:-1]

    # Total Return
    total_ret = (final_bal - initial_balance) / initial_balance * 100

    # Max Drawdown
    peak  = np.maximum.accumulate(balances)
    dd    = (peak - balances) / peak
    max_dd= dd.max() * 100

    # Sharpe (annualized, 1h = sqrt(24*365))
    if returns_pct.std() > 0:
        sharpe = returns_pct.mean() / returns_pct.std() * np.sqrt(24 * 365)
    else:
        sharpe = 0.0

    # Sortino
    neg_ret = returns_pct[returns_pct < 0]
    if len(neg_ret) > 0 and neg_ret.std() > 0:
        sortino = returns_pct.mean() / neg_ret.std() * np.sqrt(24 * 365)
    else:
        sortino = 0.0

    # Calmar
    calmar = (total_ret / max_dd) if max_dd > 0 else 0.0

    # Position stats
    positions = step_log['position'].values
    long_pct  = (positions > 0.05).mean() * 100
    short_pct = (positions < -0.05).mean() * 100
    flat_pct  = (np.abs(positions) <= 0.05).mean() * 100

    # Trade metrics
    if len(trade_log) > 0:
        wins        = trade_log[trade_log['win']]
        losses      = trade_log[~trade_log['win']]
        winrate     = len(wins) / len(trade_log) * 100
        avg_win     = wins['pnl_usd'].mean() if len(wins) > 0 else 0.0
        avg_loss    = losses['pnl_usd'].mean() if len(losses) > 0 else 0.0
        avg_hold    = trade_log['hold_bars'].mean()
        gross_profit= wins['pnl_usd'].sum() if len(wins) > 0 else 0.0
        gross_loss  = abs(losses['pnl_usd'].sum()) if len(losses) > 0 else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 999.0
        n_trades    = len(trade_log)
    else:
        winrate = avg_win = avg_loss = avg_hold = profit_factor = n_trades = 0.0

    return {
        'Label':          label,
        'Total Return %': round(total_ret, 3),
        'Final Balance':  round(final_bal, 2),
        'Max Drawdown %': round(max_dd, 3),
        'Sharpe':         round(sharpe, 3),
        'Sortino':        round(sortino, 3),
        'Calmar':         round(calmar, 3),
        'N Trades':       int(n_trades),
        'Win Rate %':     round(winrate, 2),
        'Avg Win $':      round(avg_win, 2),
        'Avg Loss $':     round(avg_loss, 2),
        'Avg Hold (bars)':round(avg_hold, 1) if n_trades > 0 else 0,
        'Profit Factor':  round(profit_factor, 3),
        'Long %':         round(long_pct, 1),
        'Short %':        round(short_pct, 1),
        'Flat %':         round(flat_pct, 1),
    }


# ===================================================================
#  REPORT PRINTER
# ===================================================================
def print_report(metrics_list: list):
    print("\n" + "=" * 70)
    print(" BACKTEST REPORT — BTCUSDT March 2026")
    print("=" * 70)

    keys = list(metrics_list[0].keys())
    labels = [m['Label'] for m in metrics_list]

    # Header
    col_w = 24
    print(f"{'Metric':<28}", end="")
    for lbl in labels:
        print(f"{lbl:>{col_w}}", end="")
    print()
    print("-" * (28 + col_w * len(labels)))

    for key in keys:
        if key == 'Label':
            continue
        print(f"  {key:<26}", end="")
        for m in metrics_list:
            val = m.get(key, '-')
            print(f"{str(val):>{col_w}}", end="")
        print()

    print("=" * 70)


# ===================================================================
#  SAVE OUTPUTS
# ===================================================================
def save_outputs(step_logs: list, trade_logs: list, metrics_list: list, ts: str):
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Step logs
    for i, (sl, label) in enumerate(zip(step_logs, [m['Label'] for m in metrics_list])):
        sl['label'] = label
    combined_steps = pd.concat(step_logs, ignore_index=True)
    step_path = os.path.join(OUT_DIR, f"step_log_{ts}.csv")
    combined_steps.to_csv(step_path, index=False, float_format="%.5f")
    print(f"\nStep log saved: {step_path}")

    # 2. Trade logs
    if any(len(tl) > 0 for tl in trade_logs):
        combined_trades = pd.concat([tl for tl in trade_logs if len(tl) > 0], ignore_index=True)
        trade_path = os.path.join(OUT_DIR, f"trade_log_{ts}.csv")
        combined_trades.to_csv(trade_path, index=False, float_format="%.5f")
        print(f"Trade log saved: {trade_path}")

    # 3. Summary
    summary_df = pd.DataFrame(metrics_list)
    summary_path = os.path.join(OUT_DIR, f"summary_{ts}.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved:  {summary_path}")


# ===================================================================
#  MAIN
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="Backtest BTC RL Bot tren data thang 3/2026")
    parser.add_argument('--model_type', type=str, default=None,
                        help="PPO hoac DQN (mac dinh: doc tu config.yaml)")
    parser.add_argument('--timeframe', type=str, default=None,
                        help="1h, 15m hoac 5m (mac dinh: doc tu config.yaml)")
    parser.add_argument('--compare', action='store_true',
                        help="So sanh OLD reward vs NEW reward tren cung 1 model")
    args = parser.parse_args()

    # Load config
    cfg = load_config()
    model_type = (args.model_type or cfg['model_type']).upper()
    timeframe  = args.timeframe or cfg.get('timeframes', '5m')
    new_reward_cfg = cfg.get('reward', {}).get(timeframe, {})  # Reward params moi tu config.yaml

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")

    print(f"\n{'='*60}")
    print(f" BACKTEST — {model_type} | {timeframe} | March 2026")
    print(f"{'='*60}")

    # Load data
    print("\n[1] Loading backtest data...")
    df_full, df_state = load_backtest_data(timeframe)

    # Load model
    print(f"\n[2] Loading model ({model_type})...")
    model = load_model(cfg, model_type)

    metrics_list = []
    step_logs    = []
    trade_logs   = []

    if args.compare:
        # ---- Chay 2 lan: OLD reward vs NEW reward ----
        print(f"\n[3a] Running backtest with OLD reward params...")
        sl_old, tl_old = run_backtest(model, cfg, df_full, df_state,
                                       label="OLD Reward", reward_cfg=OLD_REWARD_CFG)
        m_old = compute_metrics(sl_old, tl_old, "OLD Reward", cfg['env']['initial_balance'])
        metrics_list.append(m_old)
        step_logs.append(sl_old)
        trade_logs.append(tl_old)

        print(f"\n[3b] Running backtest with NEW reward params (v2)...")
        sl_new, tl_new = run_backtest(model, cfg, df_full, df_state,
                                       label="NEW Reward v2", reward_cfg=new_reward_cfg)
        m_new = compute_metrics(sl_new, tl_new, "NEW Reward v2", cfg['env']['initial_balance'])
        metrics_list.append(m_new)
        step_logs.append(sl_new)
        trade_logs.append(tl_new)

    else:
        # ---- Chay 1 lan voi reward tu config ----
        label = f"{model_type} (NEW Reward v2)"
        print(f"\n[3] Running backtest [{label}]...")
        sl, tl = run_backtest(model, cfg, df_full, df_state,
                               label=label, reward_cfg=new_reward_cfg)
        m = compute_metrics(sl, tl, label, cfg['env']['initial_balance'])
        metrics_list.append(m)
        step_logs.append(sl)
        trade_logs.append(tl)

    # ---- In report ----
    print_report(metrics_list)

    # ---- Luu CSV ----
    print("\n[4] Saving results...")
    save_outputs(step_logs, trade_logs, metrics_list, ts)

    print("\nBacktest hoan thanh.")


if __name__ == "__main__":
    main()
