import os
import glob
import pandas as pd
import argparse
from moviepy.editor import AudioFileClip
import numpy as np
from pytz import timezone

root_folder = 'datasets/'  # gets current directory
activity_folder = 'datasets/activities'
intermediate_folder = 'datasets/intermediate'

sensors = ['Accelerometer', 'Gravity', 'Gyroscope', 'Location', 'Magnetometer', 'Microphone', 'Orientation']


def audio_to_df(path, start_time):
    audio_clip = AudioFileClip(path, fps=11025)
    audio_array = audio_clip.to_soundarray()

    if audio_array.shape[1] == 2:
        audio_array = audio_array.mean(axis=1)

    timestamps = np.arange(len(audio_array)) / audio_clip.fps
    timestamps = timestamps * 10 ** 9
    timestamps = timestamps.astype(int)
    timestamps = timestamps + start_time
    print(len(timestamps))
    df = pd.DataFrame({'time': timestamps, 'audio': audio_array})
    return df


def merge_activity():
    exception_folders = [activity_folder, intermediate_folder]
    event_folders = [f.path for f in os.scandir(root_folder) if f.is_dir() and f.path not in exception_folders]

    # all_dfs = []

    for event_folder in event_folders:
        print(event_folder)
        if event_folder in exception_folders:
            continue

        master_df = pd.DataFrame()  # Master DataFrame to hold all data
        records = [f.path for f in os.scandir(event_folder) if f.is_dir()]

        if not records:
            continue

        for record in records:

            # Load time.csv from the meta folder
            meta_df = pd.read_csv(record + '/Metadata.csv')
            start_time = meta_df['recording time'].values[0]
            start_time = pd.to_datetime(start_time, format='%Y-%m-%d_%H-%M-%S', utc=True)
            local_tz = timezone('Europe/Amsterdam')
            start_time = start_time.astimezone(local_tz)
            start_time = start_time.timestamp()
            start_time = int(start_time * 10 ** 9)

            # List to store all sensor dataframes
            dfs = []
            merged_df = pd.DataFrame()
            # Load each sensor file
            for idx, sensor in enumerate(sensors):
                df = pd.read_csv(record + '/' + sensor + '.csv')
                df.drop(columns=['seconds_elapsed'], inplace=True)

                unchanged_cols = ['time']
                df.columns = [col if col in unchanged_cols else sensor + ' ' + col for col in df.columns]

                df.set_index('time', inplace=True)
                dfs.append(df)
                if idx == 0:
                    merged_df = df
                else:
                    merged_df = merged_df.join(df, how='outer')
            # Convert mp4 to csv
            # audio_df = audio_to_df(record + '/Microphone.mp4', start_time)
            # audio_df.set_index('time', inplace=True)
            # merged_df = merged_df.join(audio_df, how='outer')

            merged_df.reset_index(inplace=True)

            master_df = pd.concat([master_df, merged_df], ignore_index=True)

        # Sort the merged dataframe by time
        master_df.sort_values('time', inplace=True)
        # Add label column
        master_df['label ' + os.path.basename(event_folder)] = 1
        # Save the merged dataframe to a new csv file
        master_df.to_csv(activity_folder + '/' + os.path.basename(event_folder) + '.csv', index=False)
        # all_dfs.append(master_df)


def merge_all():
    csv_files = glob.glob(activity_folder + '/*.csv')
    all_df = pd.DataFrame()
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        all_df = pd.concat([all_df, df], ignore_index=True)
    label_cols = [col for col in all_df.columns if 'label' in col]
    all_df[label_cols] = all_df[label_cols].fillna(0)
    all_df.sort_values('time', inplace=True)
    all_df.to_csv(intermediate_folder + '/raw.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Add the parameters
    parser.add_argument('-a', action='store_true', required=False, help="Merge all activities individually")
    parser.add_argument('-m', action='store_true', required=False, help='Merge all activities into raw.csv')
    args = parser.parse_args()

    if args.a:
        merge_activity()
    if args.m:
        merge_all()
