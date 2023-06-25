import pandas as pd
import numpy as np
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib as mpl


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

# Convert the target variable to numerical labels if needed
y = pd.factorize(sensor_data['class'])[0]

x = sensor_data.drop(['time', 'class'], axis=1).values

num_features = x.shape[1]

sequence_length = 1


x = x.reshape((-1, sequence_length, num_features))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Define token and position embedding
embed_dim = 20  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer

inputs = layers.Input(shape=(sequence_length, num_features))
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(inputs)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(len(np.unique(y)), activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

history = model.fit(x_train, y_train,
                    epochs=100,
                    batch_size=256,
                    validation_data=(x_test, y_test))

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