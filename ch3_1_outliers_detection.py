import numpy as np
import pandas as pd
from Chapter3.OutlierDetection import DistributionBasedOutlierDetection
from Chapter3.OutlierDetection import DistanceBasedOutlierDetection
import argparse
import os
import glob

activity_folder = 'datasets/activities/'
result_folder = 'datasets/intermediate/after_removing_outliers/'
intermediate_folder = 'datasets/intermediate'
raw_file = 'datasets/intermediate/raw_100ms.csv'
result_file_name = 'ch3_1_after_outliers_detection.csv'

# file_name = 'raw_100ms_outliers.csv'
# result_name = 'raw_100ms_no_outliers.csv'

if not os.path.exists(result_folder):
    os.makedirs(result_folder)


def main(mode):
    dataset = pd.read_csv(raw_file)
    labeled_cols = dataset.columns[dataset.columns.str.contains('label ')]
    new_dataframe = pd.DataFrame()
    for col in labeled_cols:
        subset = dataset[dataset[col] == 1].copy() \
            .reset_index(drop=True)

        subset = removing_outliers(mode, dataset=subset)
        new_dataframe = pd.concat([new_dataframe, subset], ignore_index=True)

    new_dataframe.sort_values(by='time')
    new_dataframe.reset_index(drop=True)
    new_dataframe.to_csv(result_folder + result_file_name, index=False)


def removing_outliers(mode, dataset):
    outlier_columns = ['Accelerometer z', 'Accelerometer y', 'Accelerometer x',
                       'Gravity z', 'Gravity y', 'Gravity x',
                       'Gyroscope z', 'Gyroscope y', 'Gyroscope x',
                       'Magnetometer z', 'Magnetometer y', 'Magnetometer x',
                       'Microphone dBFS',
                       'Orientation qz', 'Orientation qy', 'Orientation qx',
                       'Orientation qw',
                       # 'Orientation roll', 'Orientation pitch', 'Orientation yaw'
                       ]

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
    # print(dataset.columns)
    return dataset


if __name__ == '__main__':
    # 创建参数解析器
    parser = argparse.ArgumentParser()

    # 添加参数
    parser.add_argument('--func', type=str, choices=['chauvenet', 'mixture', 'distance', 'LOF'], default='chauvenet',
                        help='which method')

    # 解析命令行参数
    args = parser.parse_args()

    main(args.func)
    print('end')
