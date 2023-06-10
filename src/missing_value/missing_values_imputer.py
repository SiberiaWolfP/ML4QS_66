import numpy as np
import pandas as pd
from src.missing_value.ImputationMissingValues import ImputationMissingValues
import glob
import os

activity_folder = '../../datasets/activities/'
result_folder = '../../datasets/activities/after_impute_missing_values/'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

interpolat_cols = ['Accelerometer z', 'Accelerometer y', 'Accelerometer x',
                   'Gravity z', 'Gravity y', 'Gravity x',
                   'Gyroscope z', 'Gyroscope y','Gyroscope x',
                   'Magnetometer z', 'Magnetometer y', 'Magnetometer x',
                   'Microphone dBFS',
                   'Orientation qz', 'Orientation qy', 'Orientation qx',
                   'Orientation qw','Orientation roll', 'Orientation pitch', 'Orientation yaw']
# locations
mean_cols = ['Location bearingAccuracy', 'Location speedAccuracy', 'Location verticalAccuracy', 'Location horizontalAccuracy', 'Location speed', 'Location bearing', 'Location altitude', 'Location longitude', 'Location latitude']

# print(len(dataset))
csv_files = glob.glob(activity_folder + '/*.csv')
for csv_file in csv_files:
    dataset = pd.read_csv(csv_file)
    dataset = ImputationMissingValues().interpolate_missing_values(dataset, interpolat_cols)
    dataset = ImputationMissingValues().impute_mean(dataset, mean_cols)
    dataset.to_csv(result_folder + os.path.basename(csv_file))





#最前面还是会有nah，所以先删掉，一般存在于开头和最后
dataset = dataset.dropna()
dataset.to_csv('../../datasets/intermediate/raw_without_missing_values.csv')

# print(len(dataset))

