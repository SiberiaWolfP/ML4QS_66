from kalman_filters import KalmanFilters
import pandas as pd
import glob
import os

raw_file = '../../datasets/intermediate/raw.csv'  # gets current directory
data_folder = '../../datasets/activities/after_impute_missing_values/'
result_folder = '../../datasets/activities/after_filter/'

filter_columns = ['Accelerometer z', 'Accelerometer y', 'Accelerometer x',
                   # 'Gravity z', 'Gravity y', 'Gravity x',
                   # 'Gyroscope z', 'Gyroscope y', 'Gyroscope x',
                   # 'Magnetometer z', 'Magnetometer y', 'Magnetometer x', 'Microphone dBFS',
                   # 'Orientation yaw', 'Orientation qx', 'Orientation qz',
                   # 'Orientation roll', 'Orientation qw', 'Orientation qy'
                  ]

csv_files = glob.glob(data_folder + '/*.csv')

for csv_file in csv_files:
    print(os.path.basename(csv_file))
    dataset = pd.read_csv(csv_file)
    dataset = KalmanFilters().apply_kalman_filter(dataset, filter_columns)
    dataset = dataset.dropna() # 比如最开始的缺失值
    dataset.to_csv(result_folder + os.path.basename(csv_file))

    # break




