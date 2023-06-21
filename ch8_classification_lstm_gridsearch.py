import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dense
from keras.wrappers.scikit_learn import KerasClassifier

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Embedding, Dense
import matplotlib.pyplot as plt
target = 'class'
sensor_data = pd.read_csv('/Users/young/Downloads/ml4qs_codes/ML4QS_66/datasets/intermediate/chapter5_result_class_time.csv')

print(sensor_data.columns)
sensor_data = sensor_data[['time','gra_y_min', 'gyr_x_diff_2_temp_std_ws_300', 'mag_z_temp_std_ws_300',
                              'gra_y_median_temp_std_ws_300', 'mag_y_diff_2_temp_std_ws_300',
                              'gra_y_autocorr_mean_temp_std_ws_300', 'ori_qz_fft_min_temp_mean_ws_300',
                              'acc_z_fft_std_temp_std_ws_300', 'mag_z_max_temp_mean_ws_300',
                              'gra_y_autocorr_median_temp_std_ws_300', 'acc_y_fft_max', 'acc_y_std',
                              'mag_x_autocorr_max_temp_std_ws_300', 'gra_y', 'mag_x_skew_temp_std_ws_300',
                              'gra_y_median', 'gyr_y_autocorr_median_temp_mean_ws_300', 'gyr_z_std_temp_mean_ws_300',
                              'mic_dBFS', 'ori_qz_min_temp_mean_ws_300', 'class']]

# Split the data into input features and target variable
x = sensor_data.drop(['time', 'class'], axis=1)
y = sensor_data['class']

# Convert the target variable to numerical labels if needed
y = pd.factorize(y)[0]

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Pad the sequences to have equal length
x_train = pad_sequences(x_train.values, padding='post')
x_test = pad_sequences(x_test.values, padding='post')

vocabulary_size = len(np.unique(x_train))
embedding_dim = 100
lstm_units = 256
num_classes = len(np.unique(y))
num_epochs = 20
batch_size = 256

# Define the LSTM model
def create_model(units=64, embedding_dim=100):
    model = Sequential()
    model.add(Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=x_train.shape[1]))
    model.add(LSTM(units=units))
    model.add(Dense(units=num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Create the KerasClassifier wrapper for GridSearchCV
model = KerasClassifier(build_fn=create_model, verbose=0)

# Define the parameter grid
param_grid = {
    'units': [32, 64, 128],
    'embedding_dim': [50, 100, 200]
}

# Create the GridSearchCV instance
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=3)

# Fit the GridSearchCV
grid_search.fit(x_train, y_train)

# Access the best parameters and score
print("Best Parameters: ", grid_search.best_params_)
print("Best Score: ", grid_search.best_score_)
