import numpy as np
import pandas as pd
from src.missing_value.ImputationMissingValues import ImputationMissingValues
import glob
import os

data_folder = '../../datasets/activities/after_removing_outliers'
result_folder = '../../datasets/activities/after_impute_missing_values/'

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

for csv_file in csv_files:
    dataset = pd.read_csv(csv_file)
    # locations related columns
    mean_cols = dataset.columns[dataset.columns.str.contains('Location')]

    dataset = ImputationMissingValues().interpolate_linear(dataset, interpolat_cols)
    dataset = ImputationMissingValues().impute_mean(dataset, mean_cols)
    dataset = dataset.dropna() # 比如最开始的缺失值
    dataset.to_csv(result_folder + os.path.basename(csv_file))
    print(os.path.basename(csv_file))
    # break
