import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as md
import matplotlib as mpl
import seaborn as sns

def split_dataset_by_outliers(dataset, outlier_col):
    # 获取非异常值和异常值的索引
    non_outlier_idx = dataset[outlier_col] == False
    outlier_idx = dataset[outlier_col] == True

    # 根据索引划分数据集
    non_outliers = dataset[non_outlier_idx]
    outliers = dataset[outlier_idx]

    return non_outliers, outliers

def outlier_divider(dataset, col):
    non_outliers, outliers = split_dataset_by_outliers(dataset, col+' outlier')
    return non_outliers, outliers

dataset = pd.read_csv('/Users/young/Downloads/ml4qs_codes/ML4QS_66/datasets/intermediate/after_impute_missing_values/raw_100ms.csv')
outlier_columns = ['Accelerometer z', 'Accelerometer y', 'Accelerometer x']
# for col in outlier_columns:
#     non_outliers, outliers = outlier_divider(dataset, col)
#     sns.scatterplot(non_outliers)
#     sns.scatterplot(outliers)
#     plt.legend()
#     plt.show()


dataset['Accelerometer x outlier']=dataset[dataset['Accelerometer x outlier']==True]['Accelerometer x']
dataset['Accelerometer y outlier']=dataset[dataset['Accelerometer y outlier']==True]['Accelerometer y']
dataset['Accelerometer z outlier']=dataset[dataset['Accelerometer z outlier']==True]['Accelerometer z']

plt.figure(figsize=(14, 5))
sns.scatterplot(data=dataset[['Accelerometer z', 'Accelerometer y', 'Accelerometer x','Accelerometer z outlier', 'Accelerometer y outlier', 'Accelerometer x outlier']],s=10)
# sns.scatterplot(data=dataset[['Accelerometer z outlier', 'Accelerometer y outlier', 'Accelerometer x outlier']])
plt.legend()
plt.title('Outliers Detection Example')
plt.show()