import os
import glob
import pandas as pd
import argparse

root_folder = 'datasets/'  # gets current directory
activity_folder = 'datasets/activities'
intermediate_folder = 'datasets/intermediate'


def rename_csv(path):
    csv_files = glob.glob(path + '/*.csv')

    for file_name in csv_files:
        if os.path.basename(file_name) == 'Linear Accelerometer.csv':
            os.rename(file_name, path + '/Linear Acceleration.csv')


def rename_columns(dataframe, csv_name):
    file_names = ['Accelerometer.csv', 'Gyroscope.csv', 'Linear Acceleration.csv', 'Magnetometer.csv']
    columns_1 = [['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)'],
                 ['Gyroscope x (rad/s)', 'Gyroscope y (rad/s)', 'Gyroscope z (rad/s)'],
                 ['Linear Acceleration x (m/s^2)', 'Linear Acceleration y (m/s^2)', 'Linear Acceleration z (m/s^2)'],
                 ['Magnetic field x (µT)', 'Magnetic field y (µT)', 'Magnetic field z (µT)']]
    columns_2 = [['X (m/s^2)', 'Y (m/s^2)', 'Z (m/s^2)'],
                 ['X (rad/s)', 'Y (rad/s)', 'Z (rad/s)'],
                 ['X (m/s^2)', 'Y (m/s^2)', 'Z (m/s^2)'],
                 ['X (µT)', 'Y (µT)', 'Z (µT)']]

    for i in range(len(file_names)):
        if csv_name == file_names[i]:
            for j in range(len(columns_2[i])):
                if columns_2[i][j] in dataframe.columns:
                    dataframe.rename(columns={columns_2[i][j]: columns_1[i][j]}, inplace=True)
    return dataframe


def merge_activity():
    exception_folders = [activity_folder, intermediate_folder]
    event_folders = [f.path for f in os.scandir(root_folder) if f.is_dir() and f.path not in exception_folders]

    # all_dfs = []

    for event_folder in event_folders:
        print(event_folder)
        if event_folder in exception_folders:
            continue

        master_df = pd.DataFrame()  # Master DataFrame to hold all data
        events = [f.path for f in os.scandir(event_folder) if f.is_dir()]

        if not events:
            continue

        for event in events:

            # Load time.csv from the meta folder
            meta_df = pd.read_csv(event + '/meta/time.csv')

            # Assuming the event is always START and PAUSE
            # start_times = meta_df[meta_df['event'] == 'START']['system time'].values
            # pause_times = meta_df[meta_df['event'] == 'PAUSE']['system time'].values

            # # Function to convert relative time to absolute timestamp
            # def convert_time_to_timestamp(relative_time, start_times, pause_times):
            #     for start_time, pause_time in zip(start_times, pause_times):
            #         # if relative_time <= pause_time - start_time:
            #         return (start_time + relative_time) * 10**9
            #     return None  # return None if relative_time is out of all start-pause intervals

            # Define the file paths to your sensor csv files
            rename_csv(event)
            sensor_files = ['Accelerometer.csv', 'Gyroscope.csv', 'Linear Acceleration.csv', 'Magnetometer.csv']

            # List to store all sensor dataframes
            dfs = []
            # Load each sensor file
            for file in sensor_files:
                df = pd.read_csv(event + '/' + file)
                df = rename_columns(df, file)

                # Convert relative time to timestamp
                # df['Time (s)'] = df['Time (s)'].apply(convert_time_to_timestamp, args=(start_times, pause_times)).astype('Int64')

                # Rename the time column in sensor_df to match with meta_df
                df.rename(columns={'Time (s)': 'experiment time'}, inplace=True)
                df = pd.concat([meta_df[['experiment time', 'system time']], df]).sort_values('experiment time')
                df['system time'] = df[['system time']].ffill()
                df['experiment time'] = ((df['experiment time'] + df['system time']) * 10 ** 9).astype('Int64')
                df.drop(columns=['system time'], inplace=True)
                df.dropna(inplace=True)
                df.rename(columns={'experiment time': 'Timestamp'}, inplace=True)
                df['Timestamp'] = df['Timestamp'].astype(int)
                # df.set_index('Timestamp', inplace=True)
                # df['Time (s)'] = pd.to_datetime(df['Time (s)'], unit='s')
                dfs.append(df)

            # Resample all sensor dataframes to the lower sampling rate (e.g., 100Hz of sensor4)
            # This assumes time is in seconds, adjust '100L' to match the unit of your time
            # resampled_dfs = [df.resample('10L').mean() for df in dfs]

            # Merge all sensor dataframes
            merged_df = dfs[0]
            for df in dfs[1:]:
                merged_df = pd.merge(merged_df, df, on='Timestamp', how='outer')

            # Reset index to get 'time' back as a column
            # merged_df.reset_index(inplace=True)

            # Sort the merged dataframe by time
            # merged_df.sort_values('Timestamp', inplace=True)

            # Group by time and take the mean of the sensor readings
            # merged_df = merged_df.groupby('Timestamp').mean().reset_index()

            master_df = pd.concat([master_df, merged_df], ignore_index=True)

        # Sort the merged dataframe by time
        master_df.sort_values('Timestamp', inplace=True)
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
    # for label in all_df['label'].unique():
    #     all_df['label ' + label] = (all_df['label'] == label).astype(int)
    # all_df = all_df.drop(columns=['label'])
    label_cols = [col for col in all_df.columns if 'label' in col]
    all_df[label_cols] = all_df[label_cols].fillna(0)
    all_df.sort_values('Timestamp', inplace=True)
    all_df.to_csv(intermediate_folder + '/raw.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Add the parameters
    parser.add_argument('-a', action='store_true', required=False, help="Merge existed activity data into raw.csv")
    parser.add_argument('-all', action='store_true', required=False, help='Merge all activities individually, and '
                                                                          'merge all activities into raw.csv')
    args = parser.parse_args()

    if args.a:
        merge_all()
    elif args.all:
        merge_activity()
        merge_all()
    else:
        merge_activity()
