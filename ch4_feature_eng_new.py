import pandas as pd
import numpy as np
from scipy.fft import fft
from scipy import fftpack

df = pd.read_csv('./datasets/intermediate/chapter4_result.csv')

original_predicted_cols = ['acc_z', 'acc_y', 'acc_x',
                                   'gra_z', 'gra_y', 'gra_x',
                                   'gyr_z', 'gyr_y', 'gyr_x',
                                   'mag_z', 'mag_y', 'mag_x',
                                   'ori_qz', 'ori_qy', 'ori_qx',
                                   'ori_qw']

for col in original_predicted_cols:
    df[col + '_diff_1'] = df[col].diff()
    df[col + '_diff_2'] = df[col].diff().diff()
    df[col + 'shift_1'] = df[col].shift(1)
    df[col + 'shift_2'] = df[col].shift(2)

fft_data = np.abs(fft(df[original_predicted_cols]))


# 计算基本统计值特征，并添加到原始DataFrame
for col in original_predicted_cols:
    df[col + '_max'] = df[col].max()
    df[col + '_min'] = df[col].min()
    df[col + '_mean'] = df[col].mean()
    df[col + '_median'] = df[col].median()
    df[col + '_std'] = df[col].std()

# 计算FFT特征，并添加到原始DataFrame
for col in original_predicted_cols:
    fft_values = fftpack.fft(df[col])
    fft_abs = abs(fft_values)
    df[col + '_fft_max'] = np.max(fft_abs)
    df[col + '_fft_min'] = np.min(fft_abs)
    df[col + '_fft_mean'] = np.mean(fft_abs)
    df[col + '_fft_median'] = np.median(fft_abs)
    df[col + '_fft_std'] = np.std(fft_abs)

# 计算自相关特征，并添加到原始DataFrame
for col in original_predicted_cols:
    autocorr = np.correlate(df[col], df[col], mode='full')
    df[col + '_autocorr_max'] = autocorr.max()
    df[col + '_autocorr_min'] = autocorr.min()
    df[col + '_autocorr_mean'] = autocorr.mean()
    df[col + '_autocorr_median'] = np.median(autocorr)
    df[col + '_autocorr_std'] = autocorr.std()

# 计算能量特征，并添加到原始DataFrame
for col in original_predicted_cols:
    energy = np.sum(np.square(df[col]))
    df[col + '_energy'] = energy

print('end')