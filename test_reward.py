import sys
sys.path.insert(0, 'src')
from reward import RewardHandler

print("=== Test RewardHandler v2 ===\n")

# Params moi
rw = RewardHandler(scaling=50, alpha=1.5, beta=0.15,
                   holding_penalty=0.0003, dd_threshold=0.05,
                   clip_low=-5, clip_high=5)
rw.reset(10000)

# Case 1: Long dung huong, gia tang 1%, DD=0
r1, i1 = rw.calculate(10050, 93000, 92000, 0.5, 'BUY', 1.0)
print(f"Case 1 (Long+Uptrend, price+1%): reward={r1:.4f}")
print(f"  step={i1['step_reward']:.5f} | dd={i1['dd_penalty']:.5f} | trend={i1['trend_factor']}")

# Case 2: Flat, DD=3% - flat khong bi phat DD
rw.max_net_worth = 10300  # giả lập đỉnh cũ
r2, i2 = rw.calculate(9990, 93000, 93100, 0.0, 'HOLD', 1.0)
print(f"\nCase 2 (Flat, DD=3%): reward={r2:.4f}")
print(f"  dd_penalty={i2['dd_penalty']:.5f} (should be 0.0 because flat)")

# Case 3: Dong lenh co lai (realized reward)
rw2 = RewardHandler(scaling=50, alpha=1.5, beta=0.15,
                    holding_penalty=0.0003, dd_threshold=0.05,
                    clip_low=-5, clip_high=5)
rw2.reset(10000)
rw2.prev_position = 0.5
rw2.entry_price = 90000
r3, i3 = rw2.calculate(10100, 93000, 92500, 0.0, 'CLOSE', 1.0)
print(f"\nCase 3 (Close Long, entry=90k, exit=93k, +3.3%): reward={r3:.4f}")
print(f"  realized_reward={i3['realized_reward']:.5f}")

# So sanh voi OLD params
print("\n--- So sanh OLD vs NEW ---")
ro = RewardHandler(scaling=8, alpha=0.3, beta=0.6,
                   holding_penalty=0.003, dd_threshold=0.0,
                   clip_low=-10, clip_high=10)
ro.reset(10000)
ro_r, ro_i = ro.calculate(10050, 93000, 92000, 0.5, 'BUY', 1.0)
print(f"OLD Case 1: reward={ro_r:.4f} | step={ro_i['step_reward']:.5f} | dd={ro_i['dd_penalty']:.5f}")
print(f"NEW Case 1: reward={r1:.4f}")
print(f"Ratio NEW/OLD: {abs(r1/ro_r) if ro_r != 0 else 'inf':.2f}x stronger signal")
