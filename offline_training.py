import os
import sys

os.system("nohup sh -c '" + sys.executable + " main.py' & echo $!")
# os.system("nohup sh -c '" + sys.executable + " main.py' > track_alpha_SIN.out & echo $!")

# DoubleLSTM NYC
# 4096221 short
# 4109817 long
# 4127077 both

# DoubleLSTM SIN
# 4148326 short 2193867
# 4142957 long 2195058
# 4150100 both 2175369

# DoubleLSTM NYC cl POIandCAT
# short 1175932
# long  1180290
# both  1186542
# DoubleTrans NYC cl POIandCAT
# short 1193276
# long 1197222
# both 1198946

# DoubleLSTM SIN cl POIandCAT
# short 3090655
# long 3092629
# both 3090987
# DoubleTrans SIN cl POIandCAT
# short 3079872
# long 3083651
# both 3071483

# 补充实验 transformer blocks 2
# SIN 2,3,4,5
# 2 1475241
# 3 1480257
# 4 1484035
# 5 1490193

# 补充实验 transformer blocks
# NYC 2,3,4,5
# 2 1501351
# 3 1506865
# 4 1512759
# 5 1516136
