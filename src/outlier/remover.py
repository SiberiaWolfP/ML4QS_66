import pandas as pd

activity_folder = ('../../datasets/activities/')
result_folder = ('../../datasets/intermediate/after_removing_outliers/')
raw_file = '../../datasets/intermediate/raw_100ms.csv'
file_name = 'raw_100ms_outliers.csv'
result_file_name = 'raw_100ms_without_outliers'

dataset = pd.read_csv(result_folder+file_name)
