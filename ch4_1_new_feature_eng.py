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

df = pd.read_csv('./datasets/intermediate/after_transform/ch3_after_pca.csv')

window_size = 10
features_df = pd.DataFrame()

DATA_PATH = Path('./datasets/intermediate/')
DATASET_FNAME = 'after_transform/ch3_after_pca.csv'
RESULT_FNAME = 'chapter4_result.csv'

NumAbs = NumericalAbstraction()

dataset = pd.read_csv(DATA_PATH / DATASET_FNAME)
print(dataset)

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
dataset.to_csv(DATA_PATH / RESULT_FNAME)