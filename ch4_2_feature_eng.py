from util.common import GPU
import sys
import copy
import time
from pathlib import Path
import argparse
import pandas as pd
from tqdm import tqdm

if GPU is True:
    import cudf as cd

from util.VisualizeDataset import VisualizeDataset
from Chapter4.TemporalAbstraction import NumericalAbstraction
from Chapter4.TemporalAbstraction import CategoricalAbstraction
from Chapter4.FrequencyAbstraction import FourierTransformation
from Chapter4.TextAbstraction import TextAbstraction

# Read the result from the previous chapter, and make sure the index is of the type datetime.
DATA_PATH = Path('./datasets/intermediate/')
DATASET_FNAME = 'after_transform/ch3_after_pca.csv'
# DATASET_FNAME = 'after_impute_missing_values/ch3_2_after_missing_values_imputation.csv'
RESULT_FNAME = 'chapter4_result.csv'


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():
    print_flags()

    start_time = time.time()
    try:
        dataset = pd.read_csv(DATA_PATH / DATASET_FNAME, index_col=0)
        dataset.index = pd.to_datetime(dataset.index, unit='ns')
        # dataset.set_index(dataset['time'], inplace=True)
        # dataset.index = pd.to_datetime(dataset.index)
    except IOError as e:
        print('File not found, try to run previous crowdsignals scripts first!')
        raise e

    # Let us create our visualization class again.
    DataViz = VisualizeDataset(__file__)

    # Compute the number of milliseconds covered by an instance based on the first two rows
    milliseconds_per_instance = (dataset.index[1] - dataset.index[0]).microseconds / 1000

    dataset = dataset.sort_index()
    # print(dataset.head())

    NumAbs = NumericalAbstraction()
    FreqAbs = FourierTransformation()

    if FLAGS.mode == 'aggregation':
        # Chapter 4: Identifying aggregate attributes.

        # Set the window sizes to the number of instances representing 5 seconds, 30 seconds and 5 minutes
        window_sizes = [int(float(5000) / milliseconds_per_instance),
                        int(float(0.5 * 60000) / milliseconds_per_instance),
                        int(float(5 * 60000) / milliseconds_per_instance)]

        # please look in Chapter4 TemporalAbstraction.py to look for more aggregation methods or make your own.

        for ws in window_sizes:
            dataset = NumAbs.abstract_numerical(dataset, ['acc_x'], ws, 'mean')
            dataset = NumAbs.abstract_numerical(dataset, ['acc_x'], ws, 'std')

        # DataViz.plot_dataset(dataset, ['acc_x', 'acc_x_temp_mean', 'acc_x_temp_std', 'label'],
        #                      ['exact', 'like', 'like', 'like'], ['line', 'line', 'line', 'points'])
        print("--- %s seconds ---" % (time.time() - start_time))

    if FLAGS.mode == 'frequency':
        # Now we move to the frequency domain, with the same window size.

        fs = float(1000) / milliseconds_per_instance
        ws = int(float(10000) / milliseconds_per_instance)
        dataset = FreqAbs.abstract_frequency(dataset, ['acc_x'], ws, fs)
        # Spectral analysis.
        # DataViz.plot_dataset(dataset, ['acc_x_max_freq', 'acc_x_freq_weighted', 'acc_x_pse', 'label'],
        #                      ['like', 'like', 'like', 'like'], ['line', 'line', 'line', 'points'])
        print("--- %s seconds ---" % (time.time() - start_time))

    if FLAGS.mode == 'final':
        ws = int(float(0.5 * 60000) / milliseconds_per_instance)
        fs = float(1000) / milliseconds_per_instance

        selected_predictor_cols = [c for c in dataset.columns if not 'label' in c]
        print(selected_predictor_cols)

        dataset = NumAbs.abstract_numerical(dataset, selected_predictor_cols, ws, 'mean')
        dataset = NumAbs.abstract_numerical(dataset, selected_predictor_cols, ws, 'std')
        # TODO: Add your own aggregation methods here

        # TODO: modify the columns to plot
        # DataViz.plot_dataset(dataset, ['acc_x', 'gyr_x', 'mag_x', 'mic_dBFS',
        #                                'ori_qx', 'gra_x', 'label'],
        #                      ['like', 'like', 'like', 'like', 'like', 'like', 'like'],
        #                      ['line', 'line', 'line', 'line', 'line', 'line', 'points'])

        CatAbs = CategoricalAbstraction()

        dataset = CatAbs.abstract_categorical(dataset, ['label'], ['like'], 0.03,
                                              int(float(5 * 60000) / milliseconds_per_instance), 2)

        periodic_predictor_cols = ['acc_z', 'acc_y', 'acc_x',
                                   'gra_z', 'gra_y', 'gra_x',
                                   'gyr_z', 'gyr_y', 'gyr_x',
                                   'mag_z', 'mag_y', 'mag_x',
                                   'ori_qz', 'ori_qy', 'ori_qx',
                                   'ori_qw']

        dataset = FreqAbs.abstract_frequency(copy.deepcopy(dataset), periodic_predictor_cols,
                                             int(float(10000) / milliseconds_per_instance), fs)

        # Now we only take a certain percentage of overlap in the windows, otherwise our training examples will be
        # too much alike.

        # The percentage of overlap we allow
        window_overlap = 0.9
        skip_points = int((1 - window_overlap) * ws)
        dataset = dataset.iloc[::skip_points, :]

        print(len(dataset))


        features_df.index = range(window_size - 1, dataset.shape[0])
        dataset = pd.concat([dataset, features_df], axis=1)

        # dataset = NumAbs.compute_diff_shift_features_for_window(dataset, periodic_predictor_cols)

        dataset.to_csv(DATA_PATH / RESULT_FNAME)

        # DataViz.plot_dataset(dataset, ['acc_x', 'gyr_x', 'mag_x', 'mic_dBFS',
        #                                'ori_qx', 'gra_x', 'label'],
        #                      ['like', 'like', 'like', 'like', 'like', 'like', 'like'],
        #                      ['line', 'line', 'line', 'line', 'line', 'line', 'points'])
        print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='final',
                        help="Select what version to run: final, aggregation or freq \
                        'aggregation' studies the effect of several aggregation methods \
                        'frequency' applies a Fast Fourier transformation to a single variable \
                        'final' is used for the next chapter ", choices=['aggregation', 'frequency', 'final'])

    FLAGS, unparsed = parser.parse_known_args()

    main()
