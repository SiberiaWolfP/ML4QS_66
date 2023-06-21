from util.common import GPU
import sys
import copy
import time
from pathlib import Path
import argparse
import pandas as pd
from tqdm import tqdm
from util.VisualizeDataset import VisualizeDataset
from Chapter4.TemporalAbstraction import NumericalAbstraction
from Chapter4.TemporalAbstraction import CategoricalAbstraction
from Chapter4.FrequencyAbstraction import FourierTransformation
from sklearn.model_selection import train_test_split
from Chapter7.add_class import add_class
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


DATA_PATH = Path('./datasets/intermediate/')
DATASET_FNAME = 'ch3_after_pca.csv'
result_train = 'chapter4_result_1_train.csv'
result_test = 'chapter4_result_1_test.csv'

def feature_eng_new(dataset, window_size=10):

    features_df = pd.DataFrame()

    NumAbs = NumericalAbstraction()

    periodic_predictor_cols = ['acc_z', 'acc_y', 'acc_x',
                               'gra_z', 'gra_y', 'gra_x',
                               'gyr_z', 'gyr_y', 'gyr_x',
                               'mag_z', 'mag_y', 'mag_x',
                               'ori_qz', 'ori_qy', 'ori_qx',
                               'ori_qw']

    # each window->feature extra
    loop_times = dataset.shape[0] - window_size + 1
    print('start the loop')
    for i in tqdm(range(0, loop_times)):
        window_data = dataset.iloc[i:i + window_size]
        window_features = {}
        for col in periodic_predictor_cols:
            col_features = NumAbs.compute_new_features_for_window(window_data[col])
            col_features = {f'{col}_{key}': value for key, value in col_features.items()}
            window_features.update(col_features)
        features_df = features_df.append(window_features, ignore_index=True)

    print('end the loop')

    features_df.index = range(window_size - 1, dataset.shape[0])
    dataset = pd.concat([dataset, features_df], axis=1)

    dataset = NumAbs.compute_diff_shift_features_for_window(dataset, periodic_predictor_cols)
    return dataset

def split_dataset_by_time_and_group(dataset, timestamp_col, group_col, target_col, test_size=0.2):
    # dataset[timestamp_col] = pd.to_datetime(timestamp_col)
    dataset = dataset.sort_values(by=[group_col, timestamp_col])
    train_groups, test_groups = train_test_split(dataset[group_col].unique(), test_size=test_size, shuffle=False)
    train_data = dataset[dataset[group_col].isin(train_groups)]
    test_data = dataset[dataset[group_col].isin(test_groups)]

    X_train = train_data.drop(target_col, axis=1)
    y_train = train_data[target_col]
    X_test = test_data.drop(target_col, axis=1)
    y_test = test_data[target_col]

    return X_train, X_test, y_train, y_test


dataset = pd.read_csv(DATA_PATH / DATASET_FNAME)

cols = ['label_cycling', 'label_downstairs', 'label_onsubway',
        'label_playing_phone', 'label_running', 'label_standing',
        'label_upstairs', 'label_walking']
dataset = add_class(dataset=dataset, cols=cols)
print(dataset.columns)
X_train, X_test, y_train, y_test = split_dataset_by_time_and_group(dataset=dataset, timestamp_col='time', group_col='class',target_col='class', test_size=0.2)

X_train_feature = feature_eng_new(X_train)
x_test_feature = feature_eng_new(X_test)

train_data = pd.concat([X_train_feature, y_train], axis=1)
train_data.to_csv(DATA_PATH / result_train)

test_data = pd.concat([x_test_feature, y_test], axis=1)
test_data.to_csv(DATA_PATH / result_test)