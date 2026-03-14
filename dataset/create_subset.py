# create_subset.py
import pandas as pd

# 读取原始数据
df = pd.read_csv('/root/autodl-tmp/datasets/data.csv')

# 随机取30%（约23万条）
df_subset = df.sample(frac=0.3, random_state=42)

# 保存到新文件
df_subset.to_csv('/root/autodl-tmp/datasets/data_30percent.csv', index=False)

print(f"原始数据: {len(df)} 条")
print(f"子集数据: {len(df_subset)} 条")