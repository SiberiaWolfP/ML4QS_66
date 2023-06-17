import cupy

from util.common import GPU
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import GridSearchCV
from dask.distributed import Client, LocalCluster
from dask_ml.model_selection import GridSearchCV as DaskGridSearchCV
from sklearn.model_selection import GridSearchCV as SklearnGridSearchCV
import os
import pandas as pd
import numpy as np
from Chapter7.MyMLPClassifier import MyMLPClassifier

if GPU:
    import cudf as cd
    import cupy as cp
    import cuml as cm
from sklearnex import patch_sklearn, config_context

patch_sklearn()
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.svm import SVR
from sklearn.svm import LinearSVR

class ClassificationAlgorithms:

    # Apply a neural network for classification upon the training data (with the specified composition of hidden
    # layers and number of iterations), and use the created network to predict the outcome for both the test and
    # training set. It returns the categorical predictions for the training and test set as well as the probabilities
    # associated with each class, each class being represented as a column in the data frame. To improve the speed,
    # one can use a CV of 3 only to make it faster Furthermore, you decrease the number of iteration and increase the
    # learning rate, i.e. 0.001 and use 'adam' as a solver Include n_jobs in the GridSearchCV function and set it to
    # -1 to use all processors which could also increase the speed
    def feedforward_neural_network(self, train_X, train_y, test_X, hidden_layer_sizes=(100,), max_iter=500,
                                   activation='logistic', alpha=0.0001, learning_rate='adaptive', gridsearch=True,
                                   print_model_details=False):
        if gridsearch:
            # With the current parameters for max_iter and Python 3 packages convergence is not always reached, therefore increased +1000.
            tuned_parameters = [{'hidden_layer_sizes': [(10,), (50,), (100,), (200,), (100, 10,)],
                                 'activation': ['relu', 'logistic'],
                                 'max_iter': [1000, 2000], 'alpha': [0.001, 0.01, 0.1, 1, 10]}]
            nn = SklearnGridSearchCV(MyMLPClassifier(early_stopping=True, learning_rate=learning_rate),
                                     tuned_parameters, cv=5, scoring='accuracy', n_jobs=3, verbose=10)
        else:
            # Create the model
            # nn = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, max_iter=max_iter,
            #                    learning_rate=learning_rate, alpha=alpha, random_state=42)
            nn = MyMLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, max_iter=max_iter,
                                 learning_rate=learning_rate, alpha=alpha, random_state=42, verbose=True,
                                 early_stopping=True)

        # Fit the model
        # if GPU:
        #     train_X_g = cd.DataFrame.from_pandas(train_X)
        #     train_y_g = cd.DataFrame.from_pandas(train_y)
        #     nn.fit(train_X_g.to_cupy(), train_y_g.to_cupy())
        # else:
        nn.fit(train_X, train_y.to_numpy())
        search_results = None
        if gridsearch:
            search_results = pd.DataFrame(nn.cv_results_)

        if gridsearch and print_model_details:
            print(nn.best_params_)

        if gridsearch:
            nn = nn.best_estimator_

        # Apply the model
        # pred_prob_training_y = nn.predict_proba(train_X)
        # pred_prob_test_y = nn.predict_proba(test_X)
        pred_training_y = nn.predict(train_X)
        pred_test_y = nn.predict(test_X)
        # frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=nn.classes_.cpu().numpy())
        # frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=nn.classes_.cpu().numpy())

        return pred_training_y, pred_test_y, search_results

    # Apply a support vector machine for classification upon the training data (with the specified value for
    # C, epsilon and the kernel function), and use the created model to predict the outcome for both the
    # test and training set. It returns the categorical predictions for the training and test set as well as the
    # probabilities associated with each class, each class being represented as a column in the data frame.
    # To improve the speed, one can use a CV of 3 only to make it faster
    # Include n_jobs in the GridSearchCV function and set it to -1 to use all processors which could also increase the speed
    def support_vector_machine_with_kernel(self, train_X, train_y, test_X, C=1, kernel='rbf', gamma=1e-3,
                                           gridsearch=True, print_model_details=False):
        # Create the model
        if gridsearch:
            tuned_parameters = [{'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
                                 'gamma': ['scale', 'auto', 1e-3, 1e-4],
                                 'degree': [2, 3, 4, 5],
                                 'C': [1, 10, 100]}]
            svm = SklearnGridSearchCV(SVC(probability=True), tuned_parameters, cv=5, scoring='accuracy',
                                      n_jobs=-1, verbose=10)
        else:
            svm = SVC(C=C, kernel=kernel, gamma=gamma, probability=True, cache_size=100)

        svm.fit(train_X, train_y.values.ravel())

        search_results = None
        if gridsearch:
            search_results = pd.DataFrame(svm.cv_results_)

        if gridsearch and print_model_details:
            print(svm.best_params_)

        if gridsearch:
            svm = svm.best_estimator_
        # Apply the model
        pred_training_y = svm.predict(train_X)
        pred_test_y = svm.predict(test_X)

        return pred_training_y, pred_test_y, search_results

    # Apply a support vector machine for classification upon the training data (with the specified value for
    # C, epsilon and the kernel function), and use the created model to predict the outcome for both the
    # test and training set. It returns the categorical predictions for the training and test set as well as the
    # probabilities associated with each class, each class being represented as a column in the data frame.
    # To improve the speed, one can use a CV of 3 only to make it faster and use fewer iterations
    def support_vector_machine_without_kernel(self, train_X, train_y, test_X, C=1, tol=1e-3, max_iter=1000,
                                              gridsearch=True, print_model_details=False):
        # Create the model
        if gridsearch:
            tuned_parameters = [{'max_iter': [1000, 2000], 'tol': [1e-3, 1e-4],
                                 'C': [1, 10, 100]}]
            # if GPU:
            #     svm = GridSearchCV(cm.LinearSVC(), tuned_parameters, cv=5, scoring='accuracy')
            # else:
            svm = SklearnGridSearchCV(LinearSVC(), tuned_parameters, cv=5, scoring='accuracy')
        else:
            # if GPU:
            #     svm = cm.LinearSVC(C=C, tol=tol, max_iter=max_iter)
            # else:
            svm = LinearSVC(C=C, tol=tol, max_iter=max_iter)

        # Fit the model
        svm.fit(train_X, train_y.values.ravel())

        if gridsearch and print_model_details:
            print(svm.best_params_)

        if gridsearch:
            svm = svm.best_estimator_

        # Apply the model

        distance_training_platt = 1 / (1 + np.exp(svm.decision_function(train_X)))
        pred_prob_training_y = distance_training_platt / distance_training_platt.sum(axis=1)[:, None]
        distance_test_platt = 1 / (1 + np.exp(svm.decision_function(test_X)))
        pred_prob_test_y = distance_test_platt / distance_test_platt.sum(axis=1)[:, None]
        pred_training_y = svm.predict(train_X)
        pred_test_y = svm.predict(test_X)
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=svm.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=svm.classes_)

        return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y

    # Apply a nearest neighbor approach for classification upon the training data (with the specified value for
    # k), and use the created model to predict the outcome for both the
    # test and training set. It returns the categorical predictions for the training and test set as well as the
    # probabilities associated with each class, each class being represented as a column in the data frame.
    # Again, use CV of 3 which will increase the speed of your model
    # Also, usage of n_jobs=-1 could help to increase the speed
    def k_nearest_neighbor(self, train_X, train_y, test_X, n_neighbors=5, gridsearch=True, print_model_details=False):
        # Create the model
        if gridsearch:
            tuned_parameters = [{'n_neighbors': [1, 2, 5, 10]}]
            if GPU:
                knn = SklearnGridSearchCV(cm.KNeighborsClassifier(), tuned_parameters, cv=5, scoring='accuracy')
            else:
                knn = SklearnGridSearchCV(KNeighborsClassifier(), tuned_parameters, cv=5, scoring='accuracy')
        else:
            if GPU:
                knn = cm.KNeighborsClassifier(n_neighbors=n_neighbors)
            else:
                knn = KNeighborsClassifier(n_neighbors=n_neighbors)

        # Fit the model
        knn.fit(train_X, train_y.values.ravel())

        if gridsearch and print_model_details:
            print(knn.best_params_)

        if gridsearch:
            knn = knn.best_estimator_

        # Apply the model
        pred_prob_training_y = knn.predict_proba(train_X)
        pred_prob_test_y = knn.predict_proba(test_X)
        pred_training_y = knn.predict(train_X)
        pred_test_y = knn.predict(test_X)
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=knn.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=knn.classes_)

        return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y

    # Apply a decision tree approach for classification upon the training data (with the specified value for
    # the minimum samples in the leaf, and the export path and files if print_model_details=True)
    # and use the created model to predict the outcome for both the
    # test and training set. It returns the categorical predictions for the training and test set as well as the
    # probabilities associated with each class, each class being represented as a column in the data frame.
    # Again, use CV of 3 which will increase the speed of your model
    # Also, usage of n_jobs in GridSearchCV could help to increase the speed
    def decision_tree(self, train_X, train_y, test_X, min_samples_leaf=50, criterion='gini', print_model_details=False,
                      export_tree_path='./figures/crowdsignals_ch7_classification/', export_tree_name='tree.dot',
                      gridsearch=True):
        # Create the model
        if gridsearch:
            tuned_parameters = [{'min_samples_leaf': [2, 10, 50, 100, 200],
                                 'criterion': ['gini', 'entropy'],
                                 'max_depth': [None, 5, 10, 20, 50, 100],
                                 'min_samples_split': [2, 10, 50, 100, 200],
                                 'max_features': ['sqrt', 'log2', None],
                                 'max_leaf_nodes': [None, 2, 5, 10, 20, 50, 100]}]
            dtree = SklearnGridSearchCV(DecisionTreeClassifier(), tuned_parameters, cv=5, scoring='accuracy',
                                        n_jobs=-1, verbose=0)
        else:
            dtree = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, criterion=criterion)

        # Fit the model
        dtree.fit(train_X, train_y)

        search_results = None
        if gridsearch:
            search_results = pd.DataFrame(dtree.cv_results_)

        if gridsearch and print_model_details:
            print(dtree.best_params_)

        if gridsearch:
            dtree = dtree.best_estimator_

        # Apply the model
        pred_training_y = dtree.predict(train_X)
        pred_test_y = dtree.predict(test_X)

        if print_model_details:
            ordered_indices = [i[0] for i in
                               sorted(enumerate(dtree.feature_importances_), key=lambda x: x[1], reverse=True)]
            print('Feature importance decision tree:')
            for i in range(0, len(dtree.feature_importances_)):
                print(train_X.columns[ordered_indices[i]], end='')
                print(' & ', end='')
                print(dtree.feature_importances_[ordered_indices[i]])
            if not (os.path.exists(export_tree_path)):
                os.makedirs(str(export_tree_path))
            tree.export_graphviz(dtree, out_file=str(export_tree_path) + '/' + export_tree_name,
                                 feature_names=train_X.columns, class_names=dtree.classes_)


        return pred_training_y, pred_test_y, search_results

    # Apply a naive bayes approach for classification upon the training data
    # and use the created model to predict the outcome for both the
    # test and training set. It returns the categorical predictions for the training and test set as well as the
    # probabilities associated with each class, each class being represented as a column in the data frame.
    def naive_bayes(self, train_X, train_y, test_X, gridsearch=True, print_model_details=False):
        # Create the model
        if gridsearch:
            tuned_parameters = [{'var_smoothing': np.logspace(0, -9, num=100)}]
            nb = SklearnGridSearchCV(GaussianNB(), tuned_parameters, cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
        else:
            nb = GaussianNB()

        train_y = train_y.values.ravel()
        # Fit the model
        nb.fit(train_X, train_y)

        search_results = None
        if gridsearch:
            search_results = pd.DataFrame(nb.cv_results_)

        if gridsearch and print_model_details:
            print(nb.best_params_)

        if gridsearch:
            nb = nb.best_estimator_

        # Apply the model
        pred_training_y = nb.predict(train_X)
        pred_test_y = nb.predict(test_X)


        return pred_training_y, pred_test_y, search_results

    # Apply a random forest approach for classification upon the training data (with the specified value for
    # the minimum samples in the leaf, the number of trees, and if we should print some of the details of the
    # model print_model_details=True) and use the created model to predict the outcome for both the
    # test and training set. It returns the categorical predictions for the training and test set as well as the
    # probabilities associated with each class, each class being represented as a column in the data frame.
    # Use CV of 3 to make things faster
    # Use n_jobs = -1 which will make use of all of your processors. This could speed up also the calculation
    def random_forest(self, train_X, train_y, test_X, n_estimators=10, min_samples_leaf=5, criterion='gini',
                      print_model_details=False, gridsearch=True):

        if gridsearch:
            if GPU:
                # split_criterion: 0 for gini, 1 for entropy
                tuned_parameters = [{'min_samples_leaf': [2, 10, 50],
                                     'n_estimators': [10, 50, 100, 200],
                                     'split_criterion': [0, 1],
                                     'max_depth': [10, 50, 100]}]

                rf = SklearnGridSearchCV(cm.RandomForestClassifier(), tuned_parameters, cv=5, scoring='accuracy',
                                         n_jobs=3, verbose=10)
            else:
                tuned_parameters = [{'min_samples_leaf': [2, 10, 50, 100, 200],
                                     'n_estimators': [10, 50, 100],
                                     'criterion': ['gini', 'entropy']}]
                rf = SklearnGridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5, scoring='accuracy')
        else:
            if GPU:
                rf = cm.RandomForestClassifier()(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                              split_criterion=criterion)
            else:
                rf = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                            criterion=criterion)

        # Fit the model
        rf.fit(train_X, train_y.values.ravel())

        search_results = None
        if gridsearch:
            search_results = pd.DataFrame(rf.cv_results_)

        if gridsearch and print_model_details:
            print(rf.best_params_)

        if gridsearch:
            rf = rf.best_estimator_

        pred_training_y = rf.predict(train_X)
        pred_test_y = rf.predict(test_X)

        if print_model_details:
            ordered_indices = [i[0] for i in
                               sorted(enumerate(rf.feature_importances_), key=lambda x: x[1], reverse=True)]
            print('Feature importance random forest:')
            for i in range(0, len(rf.feature_importances_)):
                print(train_X.columns[ordered_indices[i]], end='')
                print(' & ', end='')
                print(rf.feature_importances_[ordered_indices[i]])


        return pred_training_y, pred_test_y, search_results


class RegressionAlgorithms:

    # Apply a neural network for regression upon the training data (with the specified composition of
    # hidden layers and number of iterations), and use the created network to predict the outcome for both the
    # test and training set. It returns the categorical numerical predictions for the training and test set.
    # Use CV of 3 to make things faster and might be already sufficient
    def feedforward_neural_network(self, train_X, train_y, test_X, hidden_layer_sizes=(100,), max_iter=500,
                                   activation='identity', learning_rate='adaptive', gridsearch=True,
                                   print_model_details=False):
        if gridsearch:
            # With the current parameters for max_iter and Python 3 packages convergence is not always reached, therefore increased +1000.
            tuned_parameters = [{'hidden_layer_sizes': [(5,), (10,), (25,), (100,), (100, 5,), (100, 10,), ],
                                 'activation': ['identity'],
                                 'learning_rate': ['adaptive'], 'max_iter': [4000, 10000]}]
            nn = SklearnGridSearchCV(MLPRegressor(), tuned_parameters, cv=5, scoring='neg_mean_squared_error')
        else:
            # Create the model
            nn = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, max_iter=max_iter,
                              learning_rate=learning_rate)

        # Fit the model
        nn.fit(train_X, train_y)

        if gridsearch and print_model_details:
            print(nn.best_params_)

        if gridsearch:
            nn = nn.best_estimator_

        # Apply the model
        pred_training_y = nn.predict(train_X)
        pred_test_y = nn.predict(test_X)

        return pred_training_y, pred_test_y

    # Apply a support vector machine with a given kernel function for regression upon the training data (with the specified value for
    # C, gamma and the kernel function), and use the created model to predict the outcome for both the
    # test and training set. It returns the predictions for the training and test set.
    # Use CV of 3 to make things faster and might be already sufficient
    # This method is rather slow and its fit time complexity is more than quadratic with the number of samples which makes scaling hard
    def support_vector_regression_with_kernel(self, train_X, train_y, test_X, kernel='rbf', C=1, gamma=1e-3,
                                              gridsearch=True, print_model_details=False):
        if gridsearch:
            tuned_parameters = [{'kernel': ['rbf', 'poly'], 'gamma': [1e-3, 1e-4],
                                 'C': [1, 10, 100]}]
            svr = SklearnGridSearchCV(SVR(), tuned_parameters, cv=5, scoring='neg_mean_squared_error')
        else:
            # Create the model
            svr = SVR(C=C, kernel='rbf', gamma=gamma)

        # Fit the model
        svr.fit(train_X, train_y)

        if gridsearch and print_model_details:
            print(svr.best_params_)

        if gridsearch:
            svr = svr.best_estimator_

        # Apply the model
        pred_training_y = svr.predict(train_X)
        pred_test_y = svr.predict(test_X)

        return pred_training_y, pred_test_y

    # Apply a support vector machine without a complex kernel function for regression upon the training data (with the specified value for
    # C, tolerance and max iterations), and use the created model to predict the outcome for both the
    # test and training set. It returns the predictions for the training and test set.
    # Use CV of 3 to make things faster and might be already sufficient
    def support_vector_regression_without_kernel(self, train_X, train_y, test_X, C=1, tol=1e-3, max_iter=1000,
                                                 gridsearch=True, print_model_details=False):
        if gridsearch:
            # With the current parameters for max_iter and Python 3 packages convergence is not always reached, with increased iterations/tolerance often still fails to converge.
            tuned_parameters = [{'max_iter': [1000, 2000], 'tol': [1e-3, 1e-4],
                                 'C': [1, 10, 100]}]
            svr = SklearnGridSearchCV(LinearSVR(), tuned_parameters, cv=5, scoring='neg_mean_squared_error')
        else:
            # Create the model
            svr = LinearSVR(C=C, tol=tol, max_iter=max_iter)

        # Fit the model
        svr.fit(train_X, train_y)

        if gridsearch and print_model_details:
            print(svr.best_params_)

        if gridsearch:
            svr = svr.best_estimator_

        # Apply the model
        pred_training_y = svr.predict(train_X)
        pred_test_y = svr.predict(test_X)

        return pred_training_y, pred_test_y

    # Apply a nearest neighbor approach for regression upon the training data (with the specified value for
    # k), and use the created model to predict the outcome for both the
    # test and training set. It returns the predictions for the training and test set.
    # Use CV of 3 to make things faster
    # Use n_jobs = -1 which will make use of all of your processors. This could speed up also the calculation
    def k_nearest_neighbor(self, train_X, train_y, test_X, n_neighbors=5, gridsearch=True, print_model_details=False):
        # Create the model
        if gridsearch:
            tuned_parameters = [{'n_neighbors': [1, 2, 5, 10]}]
            knn = SklearnGridSearchCV(KNeighborsRegressor(), tuned_parameters, cv=5, scoring='neg_mean_squared_error')
        else:
            # Create the model
            knn = KNeighborsRegressor(n_neighbors=n_neighbors)

        # Fit the model
        knn.fit(train_X, train_y)

        if gridsearch and print_model_details:
            print(knn.best_params_)

        if gridsearch:
            knn = knn.best_estimator_

        # Apply the model
        pred_training_y = knn.predict(train_X)
        pred_test_y = knn.predict(test_X)

        return pred_training_y, pred_test_y

    # Apply a decision tree approach for regression upon the training data (with the specified value for
    # the minimum samples in the leaf, and the export path and files if print_model_details=True)
    # and use the created model to predict the outcome for both the
    # test and training set. It returns the predictions for the training and test set.
    # Use CV of 3 to make things faster and CV of 3 might be already sufficient enough
    def decision_tree(self, train_X, train_y, test_X, min_samples_leaf=50, criterion='mse', print_model_details=False,
                      export_tree_path='./figures/crowdsignals_ch7_regression/', export_tree_name='tree.dot',
                      gridsearch=True):
        # Create the model
        if gridsearch:
            tuned_parameters = [{'min_samples_leaf': [2, 10, 50, 100, 200],
                                 'criterion': ['mse']}]
            dtree = SklearnGridSearchCV(DecisionTreeRegressor(), tuned_parameters, cv=5, scoring='neg_mean_squared_error')
        else:
            # Create the model
            dtree = DecisionTreeRegressor(min_samples_leaf=min_samples_leaf, criterion=criterion)

        # Fit the model
        dtree.fit(train_X, train_y)

        if gridsearch and print_model_details:
            print(dtree.best_params_)

        if gridsearch:
            dtree = dtree.best_estimator_

        # Apply the model
        pred_training_y = dtree.predict(train_X)
        pred_test_y = dtree.predict(test_X)

        if print_model_details:
            print('Feature importance decision tree:')
            ordered_indices = [i[0] for i in
                               sorted(enumerate(dtree.feature_importances_), key=lambda x: x[1], reverse=True)]
            for i in range(0, len(dtree.feature_importances_)):
                print(train_X.columns[ordered_indices[i]], end='')
                print(' & ', end='')
                print(dtree.feature_importances_[ordered_indices[i]])
            if not (os.path.exists(export_tree_path)):
                os.makedirs(str(export_tree_path))
            tree.export_graphviz(dtree, out_file=str(export_tree_path) + '/' + export_tree_name,
                                 feature_names=train_X.columns, class_names=dtree.classes_)

        return pred_training_y, pred_test_y

    # Apply a random forest approach for regression upon the training data (with the specified value for
    # the minimum samples in the leaf, the number of trees, and if we should print some of the details of the
    # model print_model_details=True) and use the created model to predict the outcome for both the
    # test and training set. It returns the predictions for the training and test set.
    # Use CV of 3 to make things faster
    # Use n_jobs = -1 which will make use of all of your processors. This could speed up also the calculation

    def random_forest(self, train_X, train_y, test_X, n_estimators=10, min_samples_leaf=5, criterion='mse',
                      print_model_details=False, gridsearch=True):

        if gridsearch:
            tuned_parameters = [{'min_samples_leaf': [2, 10, 50, 100, 200],
                                 'n_estimators': [10, 50, 100],
                                 'criterion': ['mse']}]
            rf = SklearnGridSearchCV(RandomForestRegressor(), tuned_parameters, cv=5, scoring='neg_mean_squared_error')
        else:
            # Create the model
            rf = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                       criterion=criterion)

        # Fit the model
        rf.fit(train_X, train_y)

        if gridsearch and print_model_details:
            print(rf.best_params_)

        if gridsearch:
            rf = rf.best_estimator_

        # Apply the model
        pred_training_y = rf.predict(train_X)
        pred_test_y = rf.predict(test_X)

        if print_model_details:
            print('Feature importance random forest:')
            ordered_indices = [i[0] for i in
                               sorted(enumerate(rf.feature_importances_), key=lambda x: x[1], reverse=True)]

            for i in range(0, len(rf.feature_importances_)):
                print(train_X.columns[ordered_indices[i]], end='')
                print(' & ', end='')
                print(rf.feature_importances_[ordered_indices[i]])

        return pred_training_y, pred_test_y
