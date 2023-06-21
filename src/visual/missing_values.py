import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as md
import matplotlib as mpl
import seaborn as sns

dataset1 = pd.read_csv('/Users/young/Downloads/ml4qs_codes/ML4QS_66/datasets/intermediate/after_removing_outliers/ch3_1_after_outliers_detection.csv').iloc[116:407]
dataset2 = pd.read_csv('/Users/young/Downloads/ml4qs_codes/ML4QS_66/datasets/intermediate/after_impute_missing_values/ch3_2_after_missing_values_imputation.csv').iloc[110:401]
print((dataset1))
print((dataset2))
cols = ['mic_dBFS'] # high percentage of missing values
mpl.rcParams['font.size'] = 22
dataset = pd.DataFrame()
dataset1 = dataset1.head(1000)[cols] + 20
dataset2 = dataset2.head(1000)[cols]
plt.figure(figsize=(26,4))
plt.plot(dataset1.values, label='before', color='blue', linestyle='-', linewidth=2)
plt.plot(dataset2.values, label='after', color='red', linestyle='-', linewidth=2)

plt.legend()
plt.show()

# plt.figure(figsize=(10,4))
# plt.plot(dataset2.values, label='after')
# plt.legend()
# plt.show()

# plt.figure(figsize=(10,4))
# plt.plot(dataset2.values)
# plt.legend()
# plt.show()




