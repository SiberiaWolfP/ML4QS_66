import ast

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import numpy as np
import numbers


def plot_grid_search_validation_curve(axs, grid, param_to_vary, param_dict, best_params,
                                      title='Validation Curve', ylim=None,
                                      label=None, to_str=True,
                                      xlim=None, log=None):
    """Plots train and cross-validation scores from a GridSearchCV instance's
    best params while varying one of those params."""

    df_cv_results = grid
    valid_scores_mean = df_cv_results['mean_test_accuracy']
    valid_scores_std = df_cv_results['std_test_accuracy']

    param_cols = [c for c in df_cv_results.columns if c[:6] == 'param_']
    param_ranges = [param_dict[p[6:]] for p in param_cols]
    param_ranges_lengths = [len(pr) for pr in param_ranges]

    valid_scores_mean = np.array(valid_scores_mean).reshape(*param_ranges_lengths)
    valid_scores_std = np.array(valid_scores_std).reshape(*param_ranges_lengths)

    param_to_vary_idx = param_cols.index('param_{}'.format(param_to_vary))

    slices = []
    for idx, param in enumerate(best_params):
        if idx == param_to_vary_idx:
            slices.append(slice(None))
            continue
        best_param_val = best_params[param]
        idx_of_best_param = 0
        if isinstance(param_ranges[idx], np.ndarray):
            idx_of_best_param = param_ranges[idx].tolist().index(best_param_val)
        else:
            idx_of_best_param = param_ranges[idx].index(best_param_val)
        slices.append(idx_of_best_param)

    valid_scores_mean = valid_scores_mean[tuple(slices)]
    valid_scores_std = valid_scores_std[tuple(slices)]

    # plt.clf()

    axs.set_title(title)
    axs.set_xlabel(param_to_vary)
    axs.set_ylabel('Accuracy')

    if ylim is None:
        axs.set_ylim(0.0, 1.1)
    else:
        axs.set_ylim(*ylim)

    if not (xlim is None):
        axs.set_xlim(*xlim)

    lw = 1

    plot_fn = axs.plot
    if log:
        plot_fn = axs.semilogx

    param_range = param_ranges[param_to_vary_idx]
    # if not isinstance(param_range[0], numbers.Number):
    if to_str:
        param_range = [str(x) for x in param_range]

    plot_fn(param_range, valid_scores_mean, label=label,
            lw=lw)
    axs.fill_between(param_range, valid_scores_mean - valid_scores_std,
                     valid_scores_mean + valid_scores_std, alpha=0.1,
                     color='b', lw=lw)

    axs.legend(loc='lower right')


result_csv_path = 'search_result/round1/'

# List of datasets
datasets = ['initial set', 'Chapter 3', 'Selected features (filter)',
            'Selected features (wrapper)', 'Selected features (embedded)']

# Algorithm
algorithms = ['NN', 'SVM', 'RF', 'DT', 'NB']

# Initialize a dictionary to hold dataframes
df_dict = {}

csv_files = glob.glob(result_csv_path + '*.csv')

df_dict = {algorithm: {} for algorithm in algorithms}


# Load each CSV file into a pandas DataFrame and store it in the dictionary
for csv_file in csv_files:
    for algorithm in algorithms:
        if algorithm in csv_file:
            for data in datasets:
                if data in csv_file:
                    df_dict[algorithm][data] = pd.read_csv(csv_file)

# # Create a pairplot for each dataset
# for data in datasets:
#     # Melt DataFrame to have hyperparameters and score in the same column
#     melted_df = pd.melt(df_dict[data], id_vars='mean_test_accuracy', value_vars=['param_hidden_layer_sizes', 'param_activation', 'param_max_iter', 'param_alpha'])
#
#     # Create catplot
#     g = sns.catplot(x='variable', y='mean_test_accuracy', hue='value', data=melted_df, kind='bar', height=4, aspect=3)
#     g.set_xticklabels(rotation=30)
#     plt.title(f'Hyperparameters vs Score for {algorithm} on {data}')
#     plt.show()


params_dicts = {'NN': {'hidden_layer_sizes': [(10,), (50,), (100,), (200,), (100, 10,)],
                       'activation': ['relu', 'logistic'],
                       'max_iter': [1000, 2000], 'alpha': [0.001, 0.01, 0.1, 1, 10]},
                'SVM': {'kernel': ['rbf', 'poly', 'sigmoid'],
                        'gamma': [1e-1, 1e-2, 1e-3, 1e-4],
                        'degree': [2, 3, 4, 5],
                        'C': [1, 10, 100]},
                'RF': {'min_samples_leaf': [2, 10, 50, 100],
                       'n_estimators': [10, 50, 100, 200],
                       'split_criterion': [0, 1],
                       'max_depth': [10, 50, 100]},
                'DT': {'min_samples_leaf': [2, 10, 50, 100, 200],
                       'criterion': ['gini', 'entropy'],
                       'max_depth': [None, 5, 10, 20, 50, 100],
                       'min_samples_split': [2, 10, 50, 100, 200],
                       'max_features': ['sqrt', 'log2', None],
                       'max_leaf_nodes': [None, 2, 5, 10, 20, 50, 100]},
                'NB': {'var_smoothing': np.logspace(0, -9, num=100)}}

for algorithm in algorithms:
    fig, axs = plt.subplots(1, len(params_dicts[algorithm]), figsize=(4 * len(params_dicts[algorithm]), 4), sharey='all')  # Adjust size as needed

    for i, dataset in enumerate(datasets):
        results = df_dict[algorithm][dataset]
        best_params_str = results.iloc[results['mean_test_accuracy'].idxmax()]['params']
        best_params_dict = ast.literal_eval(best_params_str)

        for j, param in enumerate(params_dicts[algorithm].keys()):
            if len(params_dicts[algorithm]) == 1:
                plot_grid_search_validation_curve(axs, results, param, params_dicts[algorithm], best_params_dict,
                                                  title=algorithm + ' validation curve', label=dataset, to_str=False,
                                                  log=True)
            else:
                plot_grid_search_validation_curve(axs[j], results, param, params_dicts[algorithm], best_params_dict,
                                                  title=algorithm + ' validation curve', label=dataset)

    # Show the plot
    plt.tight_layout()
    plt.show()
