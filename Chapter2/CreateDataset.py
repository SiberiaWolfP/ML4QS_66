import torch
import os
import glob
GPU = False
if torch.cuda.is_available():
    GPU = True
if GPU is True:
    import cudf as pd
else:
    import pandas as pd


class CreateDataset:
    base_dir = ''
    activities_dir = ''
    intermediate_dir = ''
    sensors = []
    data_table = None
    activities_df = []

    def __init__(self, base_dir, activities_dir, intermediate_dir, sensors):
        self.base_dir = base_dir
        self.activities_dir = activities_dir
        self.intermediate_dir = intermediate_dir
        self.sensors = sensors

    # Merge activity data respectively
    def add_activity(self, activity):
        print(activity)

        master_df = pd.DataFrame()  # Master DataFrame to hold all data
        records = [f.path for f in os.scandir(self.base_dir / activity) if f.is_dir()]

        if not records:
            return None

        for record in records:
            # List to store all sensor dataframes
            dfs = []
            merged_df = pd.DataFrame()
            # Load each sensor file
            for idx, sensor in enumerate(self.sensors):
                df = pd.read_csv(record + '/' + sensor + '.csv')
                df.drop(columns=['seconds_elapsed'], inplace=True)
                if sensor == 'Orientation':
                    df.drop(columns=['yaw', 'roll', 'pitch'], inplace=True)

                unchanged_cols = ['time']
                df.columns = [col if col in unchanged_cols else sensor + ' ' + col for col in df.columns]

                df.set_index('time', inplace=True)
                dfs.append(df)
                if idx == 0:
                    merged_df = df
                else:
                    merged_df = merged_df.join(df, how='outer')

            merged_df.reset_index(inplace=True)

            master_df = pd.concat([master_df, merged_df], ignore_index=True)

        # Sort the merged dataframe by time
        master_df = master_df.sort_values('time')
        # Add label column
        master_df['label ' + os.path.basename(activity)] = 1
        master_df['label ' + os.path.basename(activity)] = master_df['label ' + os.path.basename(activity)].astype(int)
        # Save the merged dataframe to a new csv file
        # master_df.to_csv(self.activities_dir + activity + '.csv', index=False)
        self.activities_df.append(master_df)
        return master_df

    def merge_activities(self):
        self.data_table = pd.DataFrame()
        if self.activities_df is not None:
            self.data_table = pd.concat(self.activities_df, ignore_index=True)
        else:
            csv_files = glob.glob(self.activities_dir / '*.csv')
            for csv_file in csv_files:
                df = pd.read_csv(csv_file)
                self.data_table = pd.concat([self.data_table, df], ignore_index=True)
        label_cols = [col for col in self.data_table.columns if 'label' in col]
        self.data_table[label_cols] = self.data_table[label_cols].fillna(0)
        self.data_table = self.data_table.sort_values('time')
        # self.data_table.to_csv(self.intermediate_dir + '/raw.csv', index=False)
        return self.data_table

    def resample(self, granularity):
        if self.data_table is None:
            self.data_table = pd.read_csv(self.intermediate_dir / '/raw.csv')
        self.data_table['time'] = pd.to_datetime(self.data_table['time'], unit='ns')
        # self.data_table.set_index('time', inplace=True)

        gaps = self.data_table['time'].diff().dt.seconds > 1

        self.data_table['group'] = gaps.cumsum()

        resampled_dfs = []
        for _, group_df in self.data_table.groupby('group'):
            if GPU is True:
                group_df = group_df.to_pandas()
                resampled_df = pd.from_pandas(group_df.resample(on='time', rule=granularity).mean())
            else:
                resampled_df = group_df.resample(on='time', rule=granularity).mean()
                resampled_df.drop(columns=['group'], inplace=True, errors='ignore')
            resampled_dfs.append(resampled_df)
        resampled_df = pd.concat(resampled_dfs)
        resampled_df.reset_index(inplace=True)
        resampled_df['time'] = pd.to_datetime(self.data_table['time'], unit='ns').astype('int64')
        label_cols = [col for col in resampled_df.columns if 'label' in col]
        resampled_df[label_cols] = resampled_df[label_cols].astype('Int64')
        # resampled_df.to_csv(self.intermediate_dir + '/raw_' + g + '.csv', index=False)
        # self.data_table.reset_index(inplace=True)
        return resampled_df
