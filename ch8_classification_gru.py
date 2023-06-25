import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
from keras.layers import GRU, Embedding, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib as mpl

target = 'class'
col = ['time', 'gra_y_min', 'gyr_x_diff_2_temp_std_ws_300', 'mag_z_temp_std_ws_300',
'gra_y_median_temp_std_ws_300', 'mag_y_diff_2_temp_std_ws_300',
'gra_y_autocorr_mean_temp_std_ws_300', 'ori_qz_fft_min_temp_mean_ws_300',
'acc_z_fft_std_temp_std_ws_300', 'mag_z_max_temp_mean_ws_300',
'gra_y_autocorr_median_temp_std_ws_300', 'acc_y_fft_max', 'acc_y_std',
'mag_x_autocorr_max_temp_std_ws_300', 'gra_y', 'mag_x_skew_temp_std_ws_300',
'gra_y_median', 'gyr_y_autocorr_median_temp_mean_ws_300', 'gyr_z_std_temp_mean_ws_300',
'mic_dBFS', 'ori_qz_min_temp_mean_ws_300', 'class']
sensor_data_train = pd.read_csv('/Users/young/Downloads/ml4qs_codes/ML4QS_66/datasets/intermediate/bugfixed/chapter4_result_train.csv')[col]
sensor_data_test = pd.read_csv('/Users/young/Downloads/ml4qs_codes/ML4QS_66/datasets/intermediate/bugfixed/chapter4_result_test.csv')[col]

print(sensor_data_train.columns)
print(sensor_data_test.columns)

x_train = sensor_data_train.drop(['time', 'class'], axis=1)
y_train = sensor_data_train['class']

x_test = sensor_data_test.drop(['time', 'class'], axis=1)
y_test = sensor_data_test['class']

y_train = pd.factorize(y_train)[0]
y_test = pd.factorize(y_test)[0]

x_train = pad_sequences(x_train.values, padding='post')
x_test = pad_sequences(x_test.values, padding='post')

vocabulary_size = len(np.unique(x_train))
embedding_dim = 200
gru_units = 256
num_classes = len(np.unique(y_train))
num_epochs = 10
batch_size = 128



model = keras.Sequential()
model.add(Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=x_train.shape[1]))
model.add(GRU(units=gru_units))
model.add(Dense(units=num_classes, activation='softmax'))
model.add(Dropout(0.2))

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(x_test, y_test))

test_loss, test_acc = model.evaluate(x_test, y_test)

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
