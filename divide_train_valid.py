import lmdb
import random

# 打开原始的LMDB数据集：
env = lmdb.open(r'data_lmdb_release\training\MY', readonly=True)
txn = env.begin()
cursor = txn.cursor()

# 获取原始数据集的键值对列表：
data = [item for item in cursor]

# 随机打乱数据集：
random.shuffle(data)

# 计算训练集和验证集的划分点：
split_point = int(0.8 * len(data))  # 80%用于训练集，20%用于验证集

# 创建新的LMDB数据库用于训练集和验证集：
train_env = lmdb.open(r'data_lmdb_release\training\MYTR', map_size=int(1e12))
val_env = lmdb.open(r'data_lmdb_release\training\MYVA', map_size=int(1e12))

# 将数据划分为训练集和验证集并写入到新的LMDB数据库中：
with train_env.begin(write=True) as txn_train, val_env.begin(write=True) as txn_val:
    for i, (key, value) in enumerate(data):
        if i < split_point:
            txn_train.put(key, value)
        else:
            txn_val.put(key, value)

# 关闭所有的LMDB数据库：
env.close()
train_env.close()
val_env.close()
