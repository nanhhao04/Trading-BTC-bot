"""
analyze_reward_stats.py
=======================
Phan tich tinh thong ke cac thanh phan cua ham thuong RewardHandler v6
tren du lieu BTC 5m (4 thang dau 2026).

Chay tu thu muc goc: python analyze_reward_stats.py
"""
import sys, os, yaml
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from reward import RewardHandler

# ─────────────────────────────────────────────
# 1. Config & data
# ─────────────────────────────────────────────
with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

data_full  = cfg['paths']['data_full'].replace('../data/', 'data/')
data_state = cfg['paths']['data_state'].replace('../data/', 'data/')

print("Loading data...")
df_full  = pd.read_csv(data_full)
df_state = pd.read_csv(data_state)
print(f"  {len(df_full):,} rows loaded.")

rw_cfg = cfg['reward']['5m']
SCALING     = rw_cfg.get('scaling',      200.0)
ALPHA       = rw_cfg.get('alpha',        1.5)
BETA        = rw_cfg.get('beta',         0.15)
H_PENALTY   = rw_cfg.get('holding_penalty', 0.00005)
DD_THRESH   = rw_cfg.get('dd_threshold', 0.05)
CLIP_LOW    = rw_cfg.get('clip_low',     -2.0)
CLIP_HIGH   = rw_cfg.get('clip_high',    2.0)

LEVERAGE    = cfg['leverage']
USAGE       = cfg.get('max_capital_usage', 1.0)
FEE_RATE    = cfg['env']['fee_rate']
MULTIPLIER  = LEVERAGE * USAGE
FLAT_P      = rw_cfg.get('flat_penalty',      H_PENALTY * 0.5)
FEE_FACTOR  = rw_cfg.get('fee_penalty_factor', 1.0)
ACTIVE_B    = rw_cfg.get('active_bonus_rate',  0.0)   # v8: risk premium per bar

print(f"\n  Scaling={SCALING}, Alpha={ALPHA}, Multiplier={MULTIPLIER}")

# ─────────────────────────────────────────────
# 2. Vectorized component calculation
#    (khong can chay qua RewardHandler, tinh thang tren numpy)
#    => Nhanh hon 100x so voi vong lap Python
# ─────────────────────────────────────────────
close  = df_full['close'].values.astype(np.float64)
trend  = df_state['I_trend'].values.astype(np.float64)

# Log returns (T -> T+1)
log_ret = np.log(close[1:] / close[:-1])   # shape (N-1,)
trend   = trend[1:]                         # align

N = len(log_ret)
print(f"  Analysing {N:,} bars...\n")

# ── Simulate 5 position scenarios ──────────────────────────────────────────
# Long full (0.8), Long half (0.4), Flat (0.0), Short half (-0.4), Short full (-0.8)
positions  = np.array([0.8, 0.4, 0.0, -0.4, -0.8])
pos_labels = ["Long 0.8", "Long 0.4", "Flat 0.0", "Short -0.4", "Short -0.8"]

rows = []
for pos, lbl in zip(positions, pos_labels):
    # ── step_reward (Account-level) ──────────────────────────────────────
    step_reward = pos * log_ret * MULTIPLIER           # shape (N,)

    # ── trend_factor ─────────────────────────────────────────────────────
    tf = np.ones(N)
    if pos > 0:
        tf = np.where(trend == 1.0, 1.1,    # v8: 1.2->1.1
             np.where(trend == 0.0, 0.85, 1.0))
    elif pos < 0:
        tf = np.where(trend == 0.0, 1.15,   # v8: 1.3->1.15
             np.where(trend == 1.0, 0.85, 1.0))

    # ── active_bonus (risk premium per bar when holding) ──────────────────
    active_bonus = np.full(N, abs(pos) * ACTIVE_B) if abs(pos) >= 0.05 else np.zeros(N)

    # ── risk_cost (holding penalty) ──────────────────────────────────────
    risk_cost = np.full(N, abs(pos) * H_PENALTY)

    # ── flat_penalty ─────────────────────────────────────────────────────
    flat_penalty = np.zeros(N)
    if abs(pos) < 0.05:
        flat_penalty = np.where(np.isin(trend, [0.0, 1.0]), FLAT_P, 0.0)  # Fix: dung FLAT_P tu config

    # ── dd_penalty (approximate: assume 0 drawdown, baseline only) ───────
    dd_penalty = np.zeros(N)   # 0 khi khong co drawdown

    # ── fee (only on trade bars — simulate random 5% change bars) ────────
    np.random.seed(42)
    is_trade    = np.random.rand(N) < 0.05
    delta_pos   = np.where(is_trade, np.random.uniform(0.1, 2.0, N), 0.0)
    fee_scaled  = delta_pos * FEE_RATE * MULTIPLIER * FEE_FACTOR   # v8: nhan FEE_FACTOR

    # ── realized (only on trade close bars — simulate) ───────────────────
    # realized_raw: random trade PnL on ~5% bars (same bars as fee)
    realized_raw = np.zeros(N)
    if pos != 0:
        direction = 1 if pos > 0 else -1
        hold = np.random.randint(1, 20, N)
        for i in np.where(is_trade)[0]:
            j = min(i + hold[i], N-1)
            if j+1 < len(close):
                # FIX: nhan abs(pos) de dung account-level (match reward.py v8)
                realized_raw[i] = (np.log(close[j+1] / close[i+1])
                                   * direction * abs(pos) * MULTIPLIER)

    # ── raw_reward ────────────────────────────────────────────────────────
    raw_reward = (
        step_reward * tf
        + ALPHA * realized_raw
        + active_bonus            # v8: risk premium shift mean duong
        - fee_scaled
        - risk_cost
        - flat_penalty
        - dd_penalty
    )
    total_reward = np.clip(raw_reward * SCALING, CLIP_LOW, CLIP_HIGH)

    # ── Collect stats ─────────────────────────────────────────────────────
    def s(arr, name, pos_label):
        return {
            'position':   pos_label,
            'component':  name,
            'mean':       arr.mean(),
            'std':        arr.std(),
            'p5':         np.percentile(arr, 5),
            'p50':        np.percentile(arr, 50),
            'p95':        np.percentile(arr, 95),
            'min':        arr.min(),
            'max':        arr.max(),
            'pct_pos':    (arr > 0).mean() * 100,
            'pct_neg':    (arr < 0).mean() * 100,
        }

    rows += [
        s(step_reward,              'step_reward',    lbl),
        s(step_reward * tf,         'step_x_trend',   lbl),
        s(ALPHA * realized_raw,     'realized_alpha', lbl),
        s(active_bonus,             'active_bonus',   lbl),
        s(fee_scaled,               'fee_scaled',     lbl),
        s(risk_cost,                'risk_cost',      lbl),
        s(flat_penalty,             'flat_penalty',   lbl),
        s(raw_reward,               'raw_reward',     lbl),
        s(total_reward,             'total_reward',   lbl),
    ]

df_out = pd.DataFrame(rows)
df_out = df_out.set_index(['position', 'component'])

# ─────────────────────────────────────────────
# 3. In ket qua
# ─────────────────────────────────────────────
fmt = lambda x: f"{x:+.5f}"
pd.set_option('display.float_format', fmt)
pd.set_option('display.max_rows', 200)
pd.set_option('display.width', 150)

output_path = "reward_analysis_results.txt"
with open(output_path, "w", encoding="utf-8") as f:

    f.write("=" * 80 + "\n")
    f.write("PHAN TICH THANH PHAN HAM THUONG — RewardHandler v6\n")
    f.write(f"Du lieu: BTC/USDT 5m  |  {N:,} bars  |  Multiplier={MULTIPLIER}\n")
    f.write(f"Scaling={SCALING}, Alpha={ALPHA}, FeeRate={FEE_RATE}, Leverage={LEVERAGE}\n")
    f.write("=" * 80 + "\n\n")

    # ── Tong hop theo Position ─────────────────────────────────────────
    for lbl in pos_labels:
        f.write(f"\n{'─'*70}\n")
        f.write(f"  POSITION: {lbl}\n")
        f.write(f"{'─'*70}\n")
        sub = df_out.loc[lbl]
        f.write(sub.to_string() + "\n")

    # ── Summary: So sanh step vs realized ─────────────────────────────
    f.write("\n\n" + "=" * 80 + "\n")
    f.write("RATIO ANALYSIS: step_reward vs realized_alpha (scaled)\n")
    f.write("= Khi nao bot nen trade so voi hold?\n")
    f.write("=" * 80 + "\n\n")

    for lbl in pos_labels:
        if lbl == "Flat 0.0": continue
        step_mean  = abs(df_out.loc[(lbl, 'step_x_trend'), 'std']) * SCALING
        real_mean  = abs(df_out.loc[(lbl, 'realized_alpha'), 'std']) * SCALING
        fee_mean   = abs(df_out.loc[(lbl, 'fee_scaled'), 'mean']) * SCALING
        hold_time  = real_mean / step_mean if step_mean > 0 else float('inf')
        f.write(f"  {lbl}:\n")
        f.write(f"    |step_part_scaled| /bar (std) = {step_mean:+.4f}\n")
        f.write(f"    |realized_scaled| /trade(std) = {real_mean:+.4f}\n")
        f.write(f"    |fee_scaled|      /trade(mean)= {fee_mean:+.4f}\n")
        f.write(f"    => hold_time ~ {hold_time:.1f} bars  "
                f"({'OK 5-10' if 5 <= hold_time <= 10 else 'TOO SHORT' if hold_time < 5 else 'TOO LONG'})\n")
        f.write(f"    => realized/fee ratio = {real_mean/fee_mean:.1f}x "
                f"({'OK' if real_mean > fee_mean else 'BAD: fee>realized'})\n")
        f.write("\n")

    # Fix: tinh % bars thuc su bi clip (dung pct_pos/pct_neg cua total_reward)
    tr_row = df_out.loc[("Long 0.8", "total_reward")]
    raw_08 = df_out.loc[("Long 0.8", "raw_reward")]
    pct_sat_high = (raw_08['p95'] * SCALING > CLIP_HIGH) * 100
    pct_sat_low  = (raw_08['p5']  * SCALING < CLIP_LOW ) * 100
    f.write(f"  % bars > +clip_high ({CLIP_HIGH}): {pct_sat_high:.1f}% (estimated from p95 of raw)\n")
    f.write(f"  % bars < -clip_low  ({CLIP_LOW}): {pct_sat_low:.1f}% (estimated from p5 of raw)\n")
    f.write(f"  total_reward p5={tr_row['p5']:+.4f}  p50={tr_row['p50']:+.4f}  p95={tr_row['p95']:+.4f}\n")
    f.write(f"  raw_reward   p5={raw_08['p5']*SCALING:+.4f}  p50={raw_08['p50']*SCALING:+.4f}  p95={raw_08['p95']*SCALING:+.4f}  (pre-clip)\n\n")

    f.write("Phan tich hoan tat. Xem chi tiet trong: reward_analysis_results.txt\n")

print(f"\nDone! Results saved to: {output_path}")
print("Phu hien:")

# In tom tat ra console
for lbl in ["Long 0.8", "Short -0.8"]:
    step_s = df_out.loc[(lbl, 'step_x_trend'), 'mean'] * SCALING
    real_s = df_out.loc[(lbl, 'realized_alpha'), 'mean'] * SCALING
    fee_s  = df_out.loc[(lbl, 'fee_scaled'), 'mean'] * SCALING
    tot    = df_out.loc[(lbl, 'total_reward'), 'mean']
    print(f"  [{lbl}] step={step_s:+.4f}  realized={real_s:+.4f}  fee={fee_s:+.4f}  |  mean_total_reward={tot:+.4f}")
