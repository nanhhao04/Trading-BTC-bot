# Định nghĩa action cho ppo
```
    Xử lý Action Space liên tục (Continuous) cho PPO.
    Space: Box(low=-1, high=1, shape=(1,))

    Ý nghĩa giá trị action (a):
    a > 0: Tỷ trọng LONG (VD: 0.5 là Long 50% vốn)
    a < 0: Tỷ trọng SHORT (VD: -1.0 là Short 100% vốn)
    a ~ 0: Neutral (Cash)
```
Logic hoạt động:
```
Mục tiêu của rl là tìm target_pct [-1,1]. 
current_pos_pct : Vị thế hiện tại
Tính delta = target_pct - current_pos_pct
-> Tính fee_pct = abs(delta) * self.fee_rate
trade_type = 'BUY' if delta > 0 else 'SELL'

Đầu ra của step là (tar_pct, fee , trade_type)