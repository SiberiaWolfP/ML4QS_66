import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as md
import matplotlib as mpl
import seaborn as sns

dataset = pd.read_csv('/Users/young/Downloads/ml4qs_codes/ML4QS_66/datasets/intermediate/after_filter/raw_100ms.csv')
plt.figure(figsize=(14, 8))
cols = ['Accelerometer z', 'Accelerometer y', 'Accelerometer x','Accelerometer z kalman', 'Accelerometer y kalman', 'Accelerometer x kalman']
mpl.rcParams['font.size'] = 26
print(len(dataset))
outlier_cols = [col for col in dataset.columns if 'outlier' in col]
print(outlier_cols)
# Identify rows where any of the outlier columns have True values
dataset = dataset[~dataset[outlier_cols].any(axis=1)]
print(len(dataset))
# dataset = dataset[['Accelerometer x', 'Accelerometer x kalman']]
sns.lineplot(data=dataset[['Accelerometer x', 'Accelerometer x kalman']])
# sns.scatterplot(data=dataset[['Accelerometer z outlier', 'Accelerometer y outlier', 'Accelerometer x outlier']])
plt.legend()
plt.title('Kalman Filter')
plt.show()


# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.dates as md
# import matplotlib as mpl
# import seaborn as sns
#
# dataset1 = pd.read_csv('/Users/young/Downloads/ml4qs_codes/ML4QS_66/datasets/intermediate/after_impute_missing_values/running.csv')
# dataset2 = pd.read_csv('/Users/young/Downloads/ml4qs_codes/ML4QS_66/datasets/intermediate/after_filter/running.csv')
# print((dataset1))
# print((dataset2))
# cols = ['Orientation qz']
#
# dataset = pd.DataFrame()
# dataset1 = dataset1[cols].head(3000)
# dataset2 = dataset2[cols].head(3000)
# plt.figure(figsize=(10,4))
# plt.plot(dataset1.values, label='before')
# plt.plot(dataset2.values, label='after')
# plt.legend()
# plt.show()
#
# # plt.figure(figsize=(10,4))
# # plt.plot(dataset2.values)
# # plt.legend()
# # plt.show()
#



