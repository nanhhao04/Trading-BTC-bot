"""
backtest.py — Backtest RL Bot tren toan bo du lieu lich su

Su dung:
    cd src
    python backtest.py --model_type PPO              # Chay tren toan bo data
    python backtest.py --model_type PPO --compare    # So sanh old vs new reward
    python backtest.py --data_path <path/to/file>    # Dung file data tuy chinh
    python backtest.py --start_date 2025-01-01       # Loc data tu ngay nay
    python backtest.py --end_date 2026-03-31         # Loc data den ngay nay

Dau ra:
    backtest_results/
        step_log_YYYY-MM-DD_HH-MM.csv     <- Log tung buoc
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
BACKTEST_DATA_1H = os.path.join(SCRIPT_DIR, '..', 'data', 'processed', 'BTCUSDT_1h_features_full.csv')
BACKTEST_DATA_15M= os.path.join(SCRIPT_DIR, '..', 'data', 'processed', 'BTCUSDT_15m_features_full.csv')
BACKTEST_DATA_5M = os.path.join(SCRIPT_DIR, '..', 'data', 'processed', 'BTCUSDT_5m_features_full.csv')
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
    try:
        return PPO.load(path) if model_type == 'PPO' else DQN.load(path)
    except Exception as e:
        # Provide clearer error and fallback to a dummy policy for debugging/backtest
        print(f"  Warning: failed to load model due to: {e}")
        print("  Falling back to a dummy HOLD policy for backtest (not a trained model).")

        class DummyModel:
            def __init__(self, model_type):
                self.model_type = model_type

            def predict(self, obs, deterministic=True):
                # For PPO (3 actions) default to HOLD(1); for DQN fallback to 1
                return 1, None

        return DummyModel(model_type)



def load_backtest_data(timeframe: str, data_path: str = None) -> tuple:
    """Tra ve (df_full, df_state). If `data_path` is provided it is used instead of the default path for the timeframe."""
    if timeframe == '1h':
        full_path = BACKTEST_DATA_1H
    elif timeframe == '15m':
        full_path = BACKTEST_DATA_15M
    elif timeframe == '5m':
        full_path = BACKTEST_DATA_5M
    else:
        raise ValueError(f"Timeframe khong hop le: {timeframe}. Chon '1h', '15m' hoac '5m'.")

    # Allow overriding the default full_path with a user-provided data_path
    if data_path is not None:
        full_path = data_path

    state_path = full_path.replace('features_full', 'state')

    if not os.path.exists(full_path):
        raise FileNotFoundError(
            f"Khong tim thay backtest data: {full_path}\n"
            f"Hay chay script preprocess tuong ung de tao file features_full.csv"
        )

    df_full  = pd.read_csv(full_path)
    df_state = pd.read_csv(state_path)

    # Dong bo do dai
    min_len  = min(len(df_full), len(df_state))
    df_full  = df_full.iloc[:min_len].reset_index(drop=True)
    df_state = df_state.iloc[:min_len].reset_index(drop=True)
    # Ensure there is a 'timestamp' column in milliseconds
    if 'timestamp' not in df_full.columns:
        if 'open_time' in df_full.columns:
            # open_time may be datetime strings — convert to ms
            try:
                dt = pd.to_datetime(df_full['open_time'])
                df_full['timestamp'] = (dt.view('int64') // 10**6).astype('int64')
            except Exception:
                # Fallback: use pandas to_datetime then epoch
                dt = pd.to_datetime(df_full['open_time'], errors='coerce')
                df_full['timestamp'] = (dt.astype('int64') // 10**6).fillna(0).astype('int64')
        else:
            # If no time column, create synthetic timestamps as sequential ms
            df_full['timestamp'] = (pd.RangeIndex(start=0, stop=len(df_full)) * 3600 * 1000).astype('int64')

    # Ensure df_state numeric (sanitize any non-numeric columns)
    for col in df_state.columns:
        if df_state[col].dtype == object:
            df_state[col] = pd.to_numeric(df_state[col], errors='coerce')
    df_state = df_state.fillna(method='bfill').fillna(method='ffill').fillna(0.0)

    # Print range using timestamp (ms -> datetime)
    try:
        start_dt = pd.to_datetime(int(df_full['timestamp'].iloc[0]), unit='ms')
        end_dt = pd.to_datetime(int(df_full['timestamp'].iloc[-1]), unit='ms')
    except Exception:
        start_dt = df_full['timestamp'].iloc[0]
        end_dt = df_full['timestamp'].iloc[-1]

    print(f"  Backtest data ({timeframe}): {len(df_full)} nen | {start_dt} -> {end_dt}")
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
        leverage       = cfg['leverage'],
        max_capital_usage = cfg.get('max_capital_usage', 1.0),
        reward_cfg     = reward_cfg,
        max_episode_steps = len(df_full) + 10,  # Backtest: chạy hết toàn bộ data, không giới hạn
        is_backtest    = True
    )

    obs, _ = env.reset(seed=42)

    step_rows  = []
    trade_rows = []

    entry_price     = None
    entry_step      = None
    entry_pos       = None
    prev_pos        = 0.0
    prev_net_worth  = cfg['env']['initial_balance']
    prev_price      = df_full.loc[0, 'close']
    total_fee_usd   = 0.0
    n_fee_events    = 0

    done = False
    step = 0

    while not done:
        # Adapt observation to model's expected input shape (pad/truncate) to avoid
        # ValueError from stable-baselines when env obs dim != trained policy obs dim.
        obs_for_pred = obs
        expected_dim = None
        # Try common attributes to find expected observation dimension
        if hasattr(model, 'policy') and hasattr(model.policy, 'observation_space'):
            shp = getattr(model.policy.observation_space, 'shape', None)
            if shp is not None and len(shp) > 0:
                expected_dim = int(shp[0])
        elif hasattr(model, 'observation_space'):
            shp = getattr(model.observation_space, 'shape', None)
            if shp is not None and len(shp) > 0:
                expected_dim = int(shp[0])

        if expected_dim is not None:
            arr = np.asarray(obs)
            # handle 1D obs
            if arr.ndim == 1:
                if arr.shape[0] < expected_dim:
                    pad = np.zeros(expected_dim - arr.shape[0], dtype=arr.dtype)
                    arr = np.concatenate([arr, pad])
                elif arr.shape[0] > expected_dim:
                    arr = arr[:expected_dim]
                obs_for_pred = arr
            # handle batched obs (n_envs, dim)
            elif arr.ndim == 2:
                n, d = arr.shape
                if d < expected_dim:
                    pad = np.zeros((n, expected_dim - d), dtype=arr.dtype)
                    arr = np.concatenate([arr, pad], axis=1)
                elif d > expected_dim:
                    arr = arr[:, :expected_dim]
                obs_for_pred = arr

        action, _ = model.predict(obs_for_pred, deterministic=True)
        obs, reward, done, _, info = env.step(action)

        current_price = df_full.loc[step, 'close'] if step < len(df_full) else df_full.loc[len(df_full)-1, 'close']
        current_pos   = info['position']
        net_worth     = info['net_worth']
        action_name   = info['action']

        # --- Phí được lấy trực tiếp từ info['fee'] và info['holding_fee'] ở phía dưới

        nw_diff    = net_worth - prev_net_worth
        price_diff = current_price - prev_price
        step_fee   = info.get('fee', 0.0)
        h_fee      = info.get('holding_fee', 0.0)   # v9
        step_pnl   = info.get('pnl', 0.0)

        # Cộng vào tổng phí report
        total_fee_usd += (step_fee + h_fee)
        # Count fee events (trade fees or holding fees)
        if step_fee > 0:
            n_fee_events += 1
        if h_fee > 0:
            n_fee_events += 1

        step_rows.append({
            'step':        step,
            'timestamp':   df_full.loc[step, 'timestamp'] if step < len(df_full) else df_full.loc[len(df_full)-1, 'timestamp'],
            'price':       current_price,
            'price_diff':  price_diff,
            'position':    current_pos,
            'net_worth':   net_worth,
            'nw_diff':     nw_diff,
            'fee':         step_fee,
            'holding_fee': h_fee,          # v9
            'pnl':         step_pnl,
            'reward':      reward,
            'action':      action_name,
        })

        prev_net_worth = net_worth
        prev_price     = current_price

        # ---- Phat hien giao dich da dong ----
        # ---- Phat hien giao dich da dong hoac dao chieu ----
        if abs(prev_pos) > 0.05 and entry_price is not None:
            # Dong lenh neu vi the moi tro ve Flat, hoac vi the moi dao chieu (Long -> Short hoac Short -> Long)
            if abs(current_pos) < 0.05 or np.sign(prev_pos) != np.sign(current_pos):
                direction = 'LONG' if entry_pos > 0 else 'SHORT'
                pnl_pct   = (current_price - entry_price) / entry_price * (1 if entry_pos > 0 else -1) * cfg['leverage']
                pnl_usd   = cfg['env']['initial_balance'] * cfg.get('max_capital_usage', 1.0) * abs(entry_pos) * pnl_pct
                hold_bars = step - entry_step

                trade_rows.append({
                    'label':       label,
                    'entry_step':  entry_step,
                    'exit_step':   step,
                    'entry_time':  df_full.loc[entry_step, 'timestamp'],
                    'exit_time':   df_full.loc[step, 'timestamp'],
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

        # ---- Phat hien mo giao dich moi (Tu Flat, hoac sau khi vua dao chieu) ----
        if abs(current_pos) > 0.05 and entry_price is None:
            entry_price = current_price
            entry_step  = step
            entry_pos   = current_pos

        prev_pos = current_pos
        step    += 1

    # ---- Dong vi the cuoi cung neu con dang mo ----
    if abs(prev_pos) > 0.05 and entry_price is not None:
        # Neu bi dung som (done=True), step hien tai la step cuoi cung thuc thi
        last_idx = min(step, len(df_full) - 1)
        current_price = df_full.loc[last_idx, 'close']
        direction = 'LONG' if entry_pos > 0 else 'SHORT'
        pnl_pct   = (current_price - entry_price) / entry_price * (1 if entry_pos > 0 else -1) * cfg['leverage']
        pnl_usd   = cfg['env']['initial_balance'] * cfg.get('max_capital_usage', 1.0) * abs(entry_pos) * pnl_pct
        hold_bars = last_idx - entry_step

        trade_rows.append({
            'label':       label,
            'entry_step':  entry_step,
            'exit_step':   last_idx,
            'entry_time':  df_full.loc[entry_step, 'timestamp'],
            'exit_time':   df_full.loc[last_idx, 'timestamp'],
            'hold_bars':   hold_bars,
            'direction':   direction,
            'entry_price': entry_price,
            'exit_price':  current_price,
            'pnl_pct':     pnl_pct * 100,
            'pnl_usd':     pnl_usd,
            'win':         pnl_usd > 0,
        })

    # Compute avg fee per closed trade (prefer trade-based avg if trades exist)
    n_trades = len(trade_rows)
    avg_fee_per_trade = 0.0
    if n_trades > 0:
        avg_fee_per_trade = round(total_fee_usd / n_trades, 4)
    else:
        avg_fee_per_trade = round(total_fee_usd / n_fee_events, 4) if n_fee_events > 0 else 0.0

    return pd.DataFrame(step_rows), pd.DataFrame(trade_rows), {
        'total_fee_usd':     round(total_fee_usd, 4),
        'n_fee_events':      n_fee_events,
        'n_trades':          n_trades,
        'avg_fee_per_trade': avg_fee_per_trade,
    }


# ===================================================================
#  METRICS CALCULATION
# ===================================================================
def compute_metrics(step_log: pd.DataFrame, trade_log: pd.DataFrame,
                    label: str, initial_balance: float = None,
                    fee_info: dict = None) -> dict:
    if len(step_log) == 0:
        return {'Label': label}

    # Neu khong truyen initial_balance, lay net_worth dau tien cua step_log
    if initial_balance is None:
        initial_balance = step_log['net_worth'].iloc[0]
    balances   = step_log['net_worth'].values
    final_bal  = balances[-1]
    returns_pct= np.diff(balances) / balances[:-1]

    # Total Return
    total_ret = (final_bal - initial_balance) / initial_balance * 100

    # Max Drawdown
    peak  = np.maximum.accumulate(balances)
    dd    = (peak - balances) / peak
    max_dd= dd.max() * 100

    # Sharpe (annualized) — tự động theo timeframe
    # 1h: 24*365=8760 | 15m: 4*24*365=35040 | 5m: 12*24*365=105120
    TF_BARS = {'1h': 24*365, '15m': 4*24*365, '5m': 12*24*365}
    BARS_PER_YEAR = TF_BARS.get(step_log.attrs.get('timeframe', '5m'), 24*365) \
        if hasattr(step_log, 'attrs') else 24*365  # default 1h nếu không có attrs
    if returns_pct.std() > 0:
        sharpe = returns_pct.mean() / returns_pct.std() * np.sqrt(BARS_PER_YEAR)
    else:
        sharpe = 0.0

    # Sortino
    neg_ret = returns_pct[returns_pct < 0]
    if len(neg_ret) > 0 and neg_ret.std() > 0:
        sortino = returns_pct.mean() / neg_ret.std() * np.sqrt(BARS_PER_YEAR)
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

    # Balance fluctuation
    balance_std   = balances.std()
    balance_swing = balances.max() - balances.min()   # peak-to-trough swing

    # Fee info
    fi = fee_info or {}
    total_fee   = fi.get('total_fee_usd', 0.0)
    avg_fee     = fi.get('avg_fee_per_trade', 0.0)

    return {
        'Label':              label,
        'Total Return %':     round(total_ret, 3),
        'Final Balance':      round(final_bal, 2),
        'Max Drawdown %':     round(max_dd, 3),
        'Sharpe':             round(sharpe, 3),
        'Sortino':            round(sortino, 3),
        'Calmar':             round(calmar, 3),
        'N Trades':           int(n_trades),
        'Win Rate %':         round(winrate, 2),
        'Avg Win $':          round(avg_win, 2),
        'Avg Loss $':         round(avg_loss, 2),
        'Avg Hold (bars)':    round(avg_hold, 1) if n_trades > 0 else 0,
        'Profit Factor':      round(profit_factor, 3),
        '--- Balance ---':    '',
        'Balance Std $':      round(balance_std, 2),
        'Balance Swing $':    round(balance_swing, 2),
        '--- Fee ---':        '',
        'Total Fee $':        round(total_fee, 4),
        'Avg Fee/Trade $':    round(avg_fee, 4),
        '--- Position ---':   '',
        'Long %':             round(long_pct, 1),
        'Short %':            round(short_pct, 1),
        'Flat %':             round(flat_pct, 1),
    }


# ===================================================================
#  REPORT PRINTER
# ===================================================================
def print_report(metrics_list: list, date_range: str = ""):
    print("\n" + "=" * 70)
    header = f" BACKTEST REPORT — BTCUSDT"
    if date_range:
        header += f" | {date_range}"
    print(header)
    print("=" * 70)

    keys = list(metrics_list[0].keys())
    labels = [m['Label'] for m in metrics_list]

    # Header
    col_w = max(24, max([len(l) for l in labels] + [24]) + 2)
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
    parser = argparse.ArgumentParser(description="Backtest BTC RL Bot tren toan bo du lieu lich su")
    parser.add_argument('--model_type', type=str, default=None,
                        help="PPO hoac DQN (mac dinh: doc tu config.yaml)")
    parser.add_argument('--timeframe', type=str, default=None,
                        help="1h, 15m hoac 5m (mac dinh: doc tu config.yaml)")
    parser.add_argument('--compare', action='store_true',
                        help="So sanh OLD reward vs NEW reward tren cung 1 model")
    parser.add_argument('--data_path', type=str, default=None,
                        help="Duong dan toi file features_full CSV (thay the default).")
    parser.add_argument('--start_date', type=str, default=None,
                        help="Loc data tu ngay nay (format: YYYY-MM-DD). Vi du: 2025-01-01")
    parser.add_argument('--end_date', type=str, default=None,
                        help="Loc data den ngay nay (format: YYYY-MM-DD). Vi du: 2026-03-31")
    args = parser.parse_args()

    # Load config
    cfg = load_config()
    model_type = (args.model_type or cfg['model_type']).upper()
    timeframe  = args.timeframe or cfg.get('timeframes', '5m')
    new_reward_cfg = cfg.get('reward', {}).get(timeframe, {})  # Reward params moi tu config.yaml

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")

    # Gan timeframe vao step_log de tinh Sharpe dung
    _timeframe_ref = timeframe  # luu lai de dung trong lambda sau

    print(f"\n{'='*60}")
    print(f" BACKTEST — {model_type} | {timeframe} | ALL DATA")
    print(f"{'='*60}")

    # Load data
    print("\n[1] Loading backtest data...")
    df_full, df_state = load_backtest_data(timeframe, data_path=args.data_path)

    # ---- Filter by date range if provided ----
    if args.start_date or args.end_date:
        df_full['_dt'] = pd.to_datetime(df_full['timestamp'], unit='ms')
        mask = pd.Series([True] * len(df_full), index=df_full.index)
        if args.start_date:
            mask &= df_full['_dt'] >= pd.Timestamp(args.start_date)
        if args.end_date:
            mask &= df_full['_dt'] <= pd.Timestamp(args.end_date)
        orig_idx = list(df_full.index[mask])        # original integer positions
        df_state = df_state.iloc[orig_idx].reset_index(drop=True)
        df_full  = df_full.loc[mask].reset_index(drop=True)
        df_full  = df_full.drop(columns=['_dt'])
        print(f"  Filtered to: {len(df_full)} bars | {args.start_date or 'start'} -> {args.end_date or 'end'}")

    # Load model
    print(f"\n[2] Loading model ({model_type})...")
    model = load_model(cfg, model_type)

    metrics_list = []
    step_logs    = []
    trade_logs   = []

    if args.compare:
        # ---- Chay 2 lan: OLD reward vs NEW reward ----
        print(f"\n[3a] Running backtest with OLD reward params...")
        sl_old, tl_old, fi_old = run_backtest(model, cfg, df_full, df_state,
                                       label="OLD Reward", reward_cfg=OLD_REWARD_CFG)
        m_old = compute_metrics(sl_old, tl_old, "OLD Reward", cfg['env']['initial_balance'], fee_info=fi_old)
        metrics_list.append(m_old)
        step_logs.append(sl_old)
        trade_logs.append(tl_old)

        print(f"\n[3b] Running backtest with NEW reward params (v2)...")
        sl_new, tl_new, fi_new = run_backtest(model, cfg, df_full, df_state,
                                       label="NEW Reward v2", reward_cfg=new_reward_cfg)
        m_new = compute_metrics(sl_new, tl_new, "NEW Reward v2", cfg['env']['initial_balance'], fee_info=fi_new)
        metrics_list.append(m_new)
        step_logs.append(sl_new)
        trade_logs.append(tl_new)

    else:
        # ---- Chay 1 lan voi reward tu config ----
        label = f"{model_type} (NEW Reward v2)"
        print(f"\n[3] Running backtest [{label}]...")
        sl, tl, fi = run_backtest(model, cfg, df_full, df_state,
                               label=label, reward_cfg=new_reward_cfg)

        # 1. Total metrics
        m_total = compute_metrics(sl, tl, f"{label} - TOTAL", cfg['env']['initial_balance'], fee_info=fi)
        metrics_list.append(m_total)
        step_logs.append(sl)
        trade_logs.append(tl)

        # 2. Monthly metrics
        sl['date'] = pd.to_datetime(sl['timestamp'], unit='ms')
        if len(tl) > 0:
            tl['date'] = pd.to_datetime(tl['exit_time'], unit='ms')
        else:
            tl['date'] = pd.Series(dtype='datetime64[ns]')

        months = sl['date'].dt.to_period('M').unique()
        for month in sorted(months):
            month_label = f"{label} - {month}"
            sl_month = sl[sl['date'].dt.to_period('M') == month].copy()
            tl_month = tl[tl['date'].dt.to_period('M') == month].copy()

            # Phí theo tháng: lọc theo step_log timestamp
            sl_steps = set(sl_month['step'].values)
            fi_month = {
                'total_fee_usd':     fi.get('total_fee_usd', 0.0) * len(sl_month) / max(len(sl), 1),
                'n_fee_events':      fi.get('n_fee_events', 0) * len(sl_month) // max(len(sl), 1),
                'avg_fee_per_trade': fi.get('avg_fee_per_trade', 0.0),
            }

            if len(sl_month) > 0:
                m_month = compute_metrics(sl_month, tl_month, month_label, fee_info=fi_month)
                metrics_list.append(m_month)

    # ---- In report ----
    # Lay khoang thoi gian thuc te tu step_log
    _sl_ref = step_logs[0] if step_logs else pd.DataFrame()
    if len(_sl_ref) > 0 and 'timestamp' in _sl_ref.columns:
        _start = pd.to_datetime(int(_sl_ref['timestamp'].iloc[0]), unit='ms').strftime('%Y-%m-%d')
        _end   = pd.to_datetime(int(_sl_ref['timestamp'].iloc[-1]), unit='ms').strftime('%Y-%m-%d')
        _date_range = f"{_start} -> {_end}"
    else:
        _date_range = ""
    print_report(metrics_list, date_range=_date_range)

    # ---- Luu CSV ----
    print("\n[4] Saving results...")
    save_outputs(step_logs, trade_logs, metrics_list, ts)

    print("\nBacktest hoan thanh.")


if __name__ == "__main__":
    main()
