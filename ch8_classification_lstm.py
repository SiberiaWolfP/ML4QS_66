import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import LSTM, Embedding, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib as mpl

target = 'class'
cols = ['time','gra_y_min', 'gyr_x_diff_2_temp_std_ws_300', 'mag_z_temp_std_ws_300',
                              'gra_y_median_temp_std_ws_300', 'mag_y_diff_2_temp_std_ws_300',
                              'gra_y_autocorr_mean_temp_std_ws_300', 'ori_qz_fft_min_temp_mean_ws_300',
                              'acc_z_fft_std_temp_std_ws_300', 'mag_z_max_temp_mean_ws_300',
                              'gra_y_autocorr_median_temp_std_ws_300', 'acc_y_fft_max', 'acc_y_std',
                              'mag_x_autocorr_max_temp_std_ws_300', 'gra_y', 'mag_x_skew_temp_std_ws_300',
                              'gra_y_median', 'gyr_y_autocorr_median_temp_mean_ws_300', 'gyr_z_std_temp_mean_ws_300',
                              'mic_dBFS', 'ori_qz_min_temp_mean_ws_300', 'class']
sensor_data_train = pd.read_csv('/Users/young/Downloads/ml4qs_codes/ML4QS_66/datasets/intermediate/bugfixed/chapter4_result_train_full.csv')[cols]
sensor_data_test = pd.read_csv('/Users/young/Downloads/ml4qs_codes/ML4QS_66/datasets/intermediate/bugfixed/chapter4_result_test_full.csv')[cols]
# sensor_data_train = pd.read_csv('/Users/young/Downloads/ml4qs_codes/ML4QS_66/datasets/intermediate/bugfixed/chapter4_result_train.csv')[cols]
# sensor_data_test = pd.read_csv('/Users/young/Downloads/ml4qs_codes/ML4QS_66/datasets/intermediate/bugfixed/chapter4_result_test.csv')[cols]


sensor_data_train = sensor_data_train.sample(frac=1).reset_index(drop=True)
sensor_data_test = sensor_data_test.sample(frac=1).reset_index(drop=True)


x_train = sensor_data_train.drop(['time', 'class'], axis=1)
y_train = sensor_data_train['class']

x_test = sensor_data_test.drop(['time', 'class'], axis=1)
y_test = sensor_data_test['class']

# Convert the target variable to numerical labels if needed
y_train = pd.factorize(y_train)[0]
y_test = pd.factorize(y_test)[0]

# Reshape data for LSTM Layer
x_train = np.array(x_train).reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = np.array(x_test).reshape((x_test.shape[0], x_test.shape[1], 1))

embedding_dim = 200
lstm_units = 128
num_classes = len(np.unique(np.concatenate([y_train, y_test])))
num_epochs = 100
batch_size = 1024

# Define the LSTM model
model = keras.Sequential()
# model.add(LSTM(units=lstm_units, input_shape=(x_train.shape[1], 1)))
# model.add(Dense(units=num_classes, activation='softmax'))

model.add(LSTM(units=lstm_units, input_shape=(x_train.shape[1], 1), dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(units=num_classes, activation='softmax', kernel_regularizer=keras.regularizers.l1(0.01)))

optimizer = keras.optimizers.Adam(learning_rate=0.0001)  # Default is 0.01, try smaller values

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(x_test, y_test))
history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(x_test, y_test), verbose=1)

test_loss, test_acc = model.evaluate(x_test, y_test)

model.save("my_model.h5")

hist_df = pd.DataFrame(history.history)
# Save to csv
hist_csv_file = 'history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()


y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_test, y_pred)

mpl.rcParams['font.size'] = 15
plt.figure(figsize=(8, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


print(test_loss)
print(test_acc)
