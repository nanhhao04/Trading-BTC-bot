import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DQN_CSV = os.path.join(PROJECT_ROOT, 'backtest_results', 'summary_2026-05-18_20-17.csv')
PPO_CSV = os.path.join(PROJECT_ROOT, 'backtest_results', 'summary_2026-05-18_17-40.csv')
OUT_DIR = os.path.join(PROJECT_ROOT, 'backtest_results')

os.makedirs(OUT_DIR, exist_ok=True)

# Load data
df_dqn = pd.read_csv(DQN_CSV)
df_ppo = pd.read_csv(PPO_CSV)

# Clean month labels and extract monthly balance
# DQN monthly data (excluding Row 2 which is TOTAL)
dqn_monthly = df_dqn.iloc[2:].copy()
dqn_monthly['Month'] = dqn_monthly['Label'].apply(lambda x: x.split(' - ')[-1])

# PPO monthly data (excluding Row 2 which is TOTAL)
ppo_monthly = df_ppo.iloc[2:].copy()
ppo_monthly['Month'] = ppo_monthly['Label'].apply(lambda x: x.split(' - ')[-1])

# Convert balance to float
dqn_monthly['Final Balance'] = dqn_monthly['Final Balance'].astype(float)
ppo_monthly['Final Balance'] = ppo_monthly['Final Balance'].astype(float)

# Plot 1: Equity Curve Comparison
plt.figure(figsize=(12, 6))
plt.plot(dqn_monthly['Month'], dqn_monthly['Final Balance'], marker='o', linewidth=2.5, color='#3b82f6', label='DQN (Swing Strategy - Sharpe 9.38)')
plt.plot(ppo_monthly['Month'], ppo_monthly['Final Balance'], marker='s', linewidth=2, color='#ef4444', label='PPO (Momentum Strategy - Sharpe 3.49)')

plt.title('Equity Curve Comparison: DQN vs PPO (Dec 2022 - Feb 2026)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Timeline (Month)', fontsize=12)
plt.ylabel('Account Balance (USDT)', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=11, loc='upper left')
plt.tight_layout()

equity_curve_path = os.path.join(OUT_DIR, 'equity_curve.png')
# Overwrite existing image
if os.path.exists(equity_curve_path):
    os.remove(equity_curve_path)
plt.savefig(equity_curve_path, dpi=300)
plt.close()
print(f"Saved Equity Curve to: {equity_curve_path}")

# Plot 2: Key Metrics Comparison
# Overall metrics are in Row 2 (index 0 of read csv)
dqn_total = df_dqn.iloc[0]
ppo_total = df_ppo.iloc[0]

metrics = ['Total Return %', 'Max Drawdown %', 'Win Rate %', 'Sharpe']
dqn_vals = [float(dqn_total['Total Return %']), float(dqn_total['Max Drawdown %']), float(dqn_total['Win Rate %']), float(dqn_total['Sharpe'])]
ppo_vals = [float(ppo_total['Total Return %']), float(ppo_total['Max Drawdown %']), float(ppo_total['Win Rate %']), float(ppo_total['Sharpe'])]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, dqn_vals, width, label='DQN (NEW Reward v2)', color='#3b82f6')
rects2 = ax.bar(x + width/2, ppo_vals, width, label='PPO (NEW Reward v2)', color='#ef4444')

ax.set_title('Key Performance Metrics Comparison: DQN vs PPO', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.legend(fontsize=11)
ax.grid(True, linestyle='--', alpha=0.3)

# Add values on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
metrics_comparison_path = os.path.join(OUT_DIR, 'metrics_comparison.png')
# Overwrite existing image
if os.path.exists(metrics_comparison_path):
    os.remove(metrics_comparison_path)
plt.savefig(metrics_comparison_path, dpi=300)
plt.close()
print(f"Saved Metrics Comparison to: {metrics_comparison_path}")
