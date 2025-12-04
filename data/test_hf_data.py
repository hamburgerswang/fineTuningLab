import pandas as pd

# 读取parquet文件
df = pd.read_parquet('train-00000-of-00001-dbc2456f802f9fc0.parquet')

# 转换为JSON字符串
json_str = df.to_json(orient='records', indent=2, force_ascii=False)
print(json_str)

# 保存为JSON文件
df.to_json('output.json', orient='records', indent=2, force_ascii=False)