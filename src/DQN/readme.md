# Định nghĩa action cho dqn.

```
1. ĐỊNH NGHĨA ACTION MAP 
 action_map = {0: 'WAIT', 1: 'LONG', 2: 'SHORT', 3: 'CLOSE'}
 
2. ĐỊNH NGHĨA CÁC POS
-1 : SHORT     0 NEUTRAL    1: LONG

3. ĐẦU VÀO CỦA STEP ( ACTION ID, POS_CUR )
-> TÍNH ĐẦU RA LÀ NEW_POS, FEE VÀ EXECUTED LÀ T/F ( CÓ GIỮ LỆNH KHÔNG )

NẾU POS VÀ ACTION MAP NGƯỢC LẠI VỚI NHAU THÌ FEE SẼ NHÂN ĐÔI.