import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as md
import matplotlib as mpl
import seaborn as sns

dataset1 = pd.read_csv('/Users/young/Downloads/ml4qs_codes/ML4QS_66/datasets/intermediate/after_removing_outliers/walking.csv').iloc[111:1001]
dataset2 = pd.read_csv('/Users/young/Downloads/ml4qs_codes/ML4QS_66/datasets/intermediate/after_impute_missing_values/walking.csv').iloc[110:1001]
print((dataset1))
print((dataset2))
cols = ['Orientation qz']

dataset = pd.DataFrame()
dataset1 = dataset1.head(1000)[cols]
dataset2 = dataset2.head(1000)[cols]
plt.figure(figsize=(10,4))
plt.plot(dataset1.values, label='before')
plt.plot(dataset2.values, label='after')
plt.legend()
plt.show()

# plt.figure(figsize=(10,4))
# plt.plot(dataset2.values)
# plt.legend()
# plt.show()




