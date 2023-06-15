from Chapter3.pca_transform import PrincipalComponentAnalysis
from util.VisualizeDataset import VisualizeDataset
import pandas as pd
import copy

raw_file = 'datasets/intermediate/after_impute_missing_values/ch3_2_after_missing_values_imputation.csv'  # gets current directory
data_folder = 'datasets/intermediate/after_impute_missing_values/'
result_folder = 'datasets/intermediate/after_transform/'
result_file_name = 'ch3_after_pca.csv'

dataset = pd.read_csv(raw_file)
DataViz = VisualizeDataset(__file__)
PCA = PrincipalComponentAnalysis()

selected_predictor_cols = [c for c in dataset.columns if (
    not ('label' in c))
   # and (not (c == 'hr_watch_rate'))
                           ]

pc_values = PCA.determine_pc_explained_variance(
    dataset, selected_predictor_cols)

DataViz.plot_xy(x=[range(1, len(selected_predictor_cols) + 1)], y=[pc_values],
                xlabel='principal component number', ylabel='explained variance',
                ylim=[0, 1], line_styles=['b-'])

n_pcs = 7

dataset = PCA.apply_pca(copy.deepcopy(
    dataset), selected_predictor_cols, n_pcs)

dataset.to_csv(result_folder+result_file_name, index=False)