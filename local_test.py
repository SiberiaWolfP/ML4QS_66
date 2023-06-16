
import pandas as pd
DATASET_FNAME = '/Users/young/Downloads/ml4qs_codes/ML4QS_66/datasets/intermediate/after_impute_missing_values/ch3_2_after_missing_values_imputation.csv'
df = pd.read_csv(DATASET_FNAME)
cols = ['acc_x', 'gyr_x', 'mag_x', 'mic_dBFS', 'ori_qx', 'gra_x', 'label']
print(df.columns)
for col in cols:
    print(df[col])

# df = pd.read_csv('/Users/young/Downloads/ml4qs_codes/ML4QS_66/datasets/intermediate/after_removing_outliers/ch3_1_after_outliers_detection.csv')
# # 假设 df 是你的DataFrame，col_name 是你要检查的列名
# df['time']=pd.to_datetime(df['time'], unit='ns')
# df.set_index(df['time'], inplace=True)
# print(type(df['time'][1]))
# # 检查列是否单调递增
# is_increasing = df.index.is_monotonic
# print(type(df.index))
# print(type(df.index[0]))
# # 检查列是否单调递减
# is_decreasing = None
#
# # 打印结果
# if is_increasing:
#     print(f"列 {df['time']} 是单调递增的")
# elif is_decreasing:
#     print(f"列 {df['time']} 是单调递减的")
# else:
#     print(f"列 {df['time']} 不是单调递增或递减的")

# import pandas as pd
# df = pd.read_csv('/Users/young/Downloads/ml4qs_codes/ML4QS_66/datasets/intermediate/after_removing_outliers/ch3_1_after_outliers_detection.csv')
# def calculate_nan_ratio(data):
#     total_rows = data.shape[0]  # 获取总行数
#     rows_with_nan = data.isnull().any(axis=1).sum()  # 统计包含NaN值的行数
#     nan_ratio = rows_with_nan / total_rows  # 计算NaN行的比例
#     return nan_ratio
#
# nan_ratio = calculate_nan_ratio(df)
# print("NaN行的比例: {:.2%}".format(nan_ratio))
# print(len(df))

import torch

if torch.cuda.is_available():
    GPU = True
    device_name = torch.cuda.get_device_name()
    if 'M' in device_name:
        print("使用的是M系列的NVIDIA GPU")
else:
    GPU = False
    print("未检测到可用的GPU")

import torch
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")