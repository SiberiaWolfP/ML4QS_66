import numpy as np
import pandas as pd
from OutlierDetection import DistributionBasedOutlierDetection
from OutlierDetection import DistanceBasedOutlierDetection
import argparse
import os
import glob


activity_folder = ('../../datasets/activities/')
result_folder = ('../../datasets/activities/after_removing_outliers/')
raw_file = '../../datasets/intermediate/raw_100ms.csv'
file_name = 'raw_100ms_outliers.csv'
result_name = 'raw_100ms_no_outliers.csv'

if not os.path.exists(result_folder):
    os.makedirs(result_folder)

def main(mode):
    csv_files = glob.glob(activity_folder + '/*.csv')

    for csv_file in csv_files:
        dataset = pd.read_csv(csv_file)
        dataset = removing_outliers(mode, dataset=dataset)
        dataset.to_csv(result_folder + os.path.basename(csv_file))

    dataset = pd.read_csv(raw_file)
    dataset = removing_outliers(mode, dataset=dataset)
    dataset.to_csv(result_folder + file_name)

    # outlier_cols = [col for col in dataset.columns if 'outlier' in col]
    # dataset = dataset[~dataset[outlier_cols].any(axis=1)]
    # dataset.to_csv(result_folder + result_name)


def removing_outliers(mode, dataset):

    outlier_columns = ['Accelerometer z', 'Accelerometer y', 'Accelerometer x',
                       'Gravity z', 'Gravity y', 'Gravity x',
                       'Gyroscope z', 'Gyroscope y', 'Gyroscope x',
                       'Magnetometer z','Magnetometer y', 'Magnetometer x','Microphone dBFS',
                       'Orientation yaw', 'Orientation qx', 'Orientation qz',
                       'Orientation roll', 'Orientation qw', 'Orientation qy']


    if mode == 'chauvenet':
        for col in outlier_columns:
            dataset = DistributionBasedOutlierDetection().chauvenet(data_table=dataset, col=col, C=2)

    elif mode == 'mixture':
        for col in outlier_columns:
            dataset = DistributionBasedOutlierDetection().mixture_model(dataset, col)

    elif mode == 'distance':
        for col in outlier_columns:
            try:
                dataset = DistanceBasedOutlierDetection().simple_distance_based(
                    dataset, [col], 'euclidean', 0.10, 0.99)
            except MemoryError as e:
                print(
                    'Not enough memory available for simple distance-based outlier detection...')
                print('Skipping.')

    elif mode == 'LOF':
        for col in outlier_columns:
            try:
                dataset = DistanceBasedOutlierDetection().local_outlier_factor(
                    dataset, [col], 'euclidean', 5)
            except MemoryError as e:
                print('Not enough memory available for lof...')
                print('Skipping.')

    elif mode == 'final':
        # for col in [c for c in dataset.columns if not 'label' in c]:
        for col in outlier_columns:
            print(f'Measurement is now: {col}')
            dataset = DistributionBasedOutlierDetection.chauvenet(dataset, col, 2)
            # dataset.loc[dataset[f'{col} outlier'] == True, col] = np.nan
            # del dataset[col + '_outlier']

    dataset.loc[dataset[f'{col} outlier'] == True, col] = np.nan
    print(dataset.columns)
    # return dataset


if __name__ == '__main__':
    # 创建参数解析器
    parser = argparse.ArgumentParser()

    # 添加参数
    parser.add_argument('--func', type=str, choices=['chauvenet', 'mixture', 'distance', 'LOF'], default='chauvenet', help='which method')

    # 解析命令行参数
    args = parser.parse_args()

    main(args.func)
    print('end')