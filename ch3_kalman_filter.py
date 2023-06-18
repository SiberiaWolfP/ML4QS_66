from Chapter3.kalman_filters import KalmanFilters
import glob
import os
import pandas as pd

# slow!
from util.util import del_cols

raw_file = 'datasets/intermediate/raw_100ms.csv'  # gets current directory
data_folder = 'datasets/intermediate/after_impute_missing_values/'
result_folder = 'datasets/intermediate/after_filter/'
result_file_name = 'ch3_1_2_after_kalman_filter.csv'

if not os.path.exists(result_folder):
    os.makedirs(result_folder)

filter_columns = ['acc_z', 'acc_y', 'acc_x',
                    'gra_z', 'gra_y', 'gra_x',
                    'gyr_z', 'gyr_y', 'gyr_x',
                    'mag_z', 'mag_y', 'mag_x',
                    'mic_dBFS',
                    'ori_qz', 'ori_qy', 'ori_qx', 'ori_qw']

# csv_files = glob.glob(data_folder + '/*.csv')

# for csv_file in csv_files:
#     print(os.path.basename(csv_file))
#     dataset = pd.read_csv(csv_file)
#     dataset = KalmanFilters().apply_kalman_filter(dataset, filter_columns)
#     dataset.to_csv(result_folder + os.path.basename(csv_file))

dataset = pd.read_csv(raw_file)
labeled_cols = dataset.columns[dataset.columns.str.contains('label_')]
new_dataframe = pd.DataFrame()
for col in labeled_cols:
    subset = dataset[dataset[col] == 1].copy()\
        .reset_index(drop=True)

    subset = KalmanFilters().apply_kalman_filter(data_table=subset, cols=filter_columns)
    new_dataframe = pd.concat([new_dataframe, subset], ignore_index=True)

new_dataframe.sort_values(by='time')
new_dataframe.reset_index(drop=True)

kalman_check_cols = new_dataframe.columns[new_dataframe.columns.str.contains('_kalman')]
new_dataframe = del_cols(new_dataframe, kalman_check_cols)

new_dataframe.to_csv(result_folder + result_file_name, index=False)
print('end')
