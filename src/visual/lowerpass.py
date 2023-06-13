import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as md
import matplotlib as mpl
import seaborn as sns

dataset = pd.read_csv('/Users/young/Downloads/ml4qs_codes/ML4QS_66/datasets/activities/after_transform/result.csv')
plt.figure(figsize=(14, 5))
cols = ['Accelerometer z', 'Accelerometer y', 'Accelerometer x','Accelerometer z kalman', 'Accelerometer y kalman', 'Accelerometer x kalman']

print(len(dataset))
outlier_cols = [col for col in dataset.columns if 'outlier' in col]
print(outlier_cols)
# Identify rows where any of the outlier columns have True values
dataset = dataset[~dataset[outlier_cols].any(axis=1)]
print(len(dataset))
# dataset = dataset[['Accelerometer x', 'Accelerometer x kalman']]
sns.lineplot(data=dataset[['Orientation qz', 'Orientation qz lowpass']])
# sns.scatterplot(data=dataset[['Accelerometer z outlier', 'Accelerometer y outlier', 'Accelerometer x outlier']])
plt.legend()
plt.title('Lowpass Filter')
plt.show()