import sys
import copy
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

import DataTransformation

# Set up the file names and locations.
from src.missing_value.ImputationMissingValues import ImputationMissingValues


activity_folder = ('../../datasets/activities/after_filter/')
result_folder = ('../../datasets/activities/after_transform/')
file_name = '/Users/young/Downloads/ml4qs_codes/ML4QS_66/datasets/activities/after_filter/raw_100ms.csv'

def main():

    # Next, import the data from the specified location and parse the date index.
    try:
        dataset = pd.read_csv(Path(file_name), index_col=0)
        dataset.index = pd.to_datetime(dataset.index)
    except IOError as e:
        print('File not found, try to run previous crowdsignals scripts first!')
        raise e


    # Compute the number of milliseconds covered by an instance based on the first two rows
    milliseconds_per_instance = (dataset.time[1] - dataset.time[0]) // 1000000
    print(milliseconds_per_instance)
    # MisVal = ImputationMissingValues()
    LowPass = DataTransformation.LowPassFilter()
    PCA = DataTransformation.PrincipalComponentAnalysis()


    if FLAGS.mode == 'lowpass':

        # Let us apply a lowpass filter and reduce the importance of the data above 1.5 Hz

        # Determine the sampling frequency.
        fs = float(1000) / milliseconds_per_instance
        cutoff = 1.5
        # Let us study acc_phone_x:
        new_dataset = LowPass.low_pass_filter(copy.deepcopy(
            dataset), 'Orientation qz', fs, cutoff, order=10)

        new_dataset.to_csv('../../datasets/activities/after_transform/result.csv')

    elif FLAGS.mode == 'PCA':

        # first impute again, as PCA can not deal with missing values
        cols = [c for c in dataset.columns if not 'label' in c]
        print(cols)
        dataset = dataset[cols]


        selected_predictor_cols = [c for c in dataset.columns if (
            not ('label' in c)) and (not (c == 'hr_watch_rate'))]
        pc_values = PCA.determine_pc_explained_variance(
            dataset, selected_predictor_cols)


        # We select 7 as the best number of PC's as this explains most of the variance

        n_pcs = 7

        dataset = PCA.apply_pca(copy.deepcopy(
            dataset), selected_predictor_cols, n_pcs)

        print(dataset)


        dataset.to_csv(result_folder + 'result.csv')


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='lowpass',
                        help="Select what version to run: final, imputation, lowpass or PCA \
                        'lowpass' applies the lowpass-filter to a single variable \
                        'imputation' is used for the next chapter \
                        'PCA' is to study the effect of PCA and plot the results\
                        'final' is used for the next chapter", choices=['lowpass', 'PCA'])

    FLAGS, unparsed = parser.parse_known_args()

    main()
