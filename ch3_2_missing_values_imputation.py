import pandas as pd
from Chapter3.ImputationMissingValues import ImputationMissingValues
import glob
import os

data_folder = 'datasets/intermediate/after_removing_outliers/'
result_folder = 'datasets/intermediate/after_impute_missing_values/'
raw_file = 'datasets/intermediate/after_removing_outliers/ch3_1_after_outliers_detection.csv'
result_file_name = 'ch3_2_after_missing_values_imputation.py'

if not os.path.exists(result_folder):
    os.makedirs(result_folder)

interpolat_cols = ['Accelerometer z', 'Accelerometer y', 'Accelerometer x',
                   'Gravity z', 'Gravity y', 'Gravity x',
                   'Gyroscope z', 'Gyroscope y', 'Gyroscope x',
                   'Magnetometer z', 'Magnetometer y', 'Magnetometer x',
                   'Microphone dBFS',
                   'Orientation qz', 'Orientation qy', 'Orientation qx',
                   'Orientation qw', 'Orientation roll', 'Orientation pitch', 'Orientation yaw']

if not os.path.exists(result_folder):
    os.makedirs(result_folder)

csv_files = glob.glob(data_folder + '/*.csv')

def impute_missing(dataset):
    mean_cols = dataset.columns[dataset.columns.str.contains('Location')]

    dataset = ImputationMissingValues().interpolate_linear(dataset, interpolat_cols)
    dataset = ImputationMissingValues().impute_mean(dataset, mean_cols)
    dataset = dataset.dropna() # 比如最开始的缺失值
    return dataset


# for csv_file in csv_files:
#     dataset = pd.read_csv(csv_file)
#     # locations related columns
#     dataset = impute_missing(dataset)
#     dataset.to_csv(result_folder + os.path.basename(csv_file))
#     print(os.path.basename(csv_file))
#     # break


dataset = pd.read_csv(raw_file)
labeled_cols = dataset.columns[dataset.columns.str.contains('label ')]
new_dataframe = pd.DataFrame()
for col in labeled_cols:
    subset = dataset[dataset[col] == 1].copy()\
        .reset_index(drop=True)
    print(len(subset))
    subset = impute_missing(dataset=subset)
    new_dataframe = pd.concat([new_dataframe, subset], ignore_index=True)

new_dataframe.sort_values(by='time')
new_dataframe.reset_index(drop=True)
new_dataframe.to_csv(result_folder + result_file_name)
print('end')
