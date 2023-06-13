# Import the relevant classes.
from Chapter2.CreateDataset import CreateDataset
from util.VisualizeDataset import VisualizeDataset
from util import util
from pathlib import Path
import copy

# Chapter 2: Initial exploration of the dataset.

DATASET_PATH = Path('./datasets/')
ACTIVITIES_PATH = Path('./datasets/activities/')
RESULT_PATH = Path('./datasets/intermediate/')

# Set a granularity (the discrete step size of our time series data). We'll use a course-grained granularity of one
# instance per minute, and a fine-grained one with four instances per second.
GRANULARITIES = ['1min', '1S', '100ms']

ACTIVITIES = ['cycling', 'downstairs', 'onsubway', 'playing_phone',
              'running', 'standing', 'upstairs', 'walking']

SENSORS = ['Accelerometer', 'Gravity', 'Gyroscope', 'Magnetometer', 'Microphone', 'Orientation']

# We can call Path.mkdir(exist_ok=True) to make any required directories if they don't already exist.
[path.mkdir(exist_ok=True, parents=True) for path in [DATASET_PATH, RESULT_PATH]]

print('Please wait, this will take a while to run!')

dataset = CreateDataset(base_dir=DATASET_PATH, activities_dir=ACTIVITIES_PATH, intermediate_dir=RESULT_PATH,
                        sensors=SENSORS)

# Merge activities separately
for activity in ACTIVITIES:
    activity_df = dataset.add_activity(activity)
    if activity_df is not None:
        activity_df.to_csv(ACTIVITIES_PATH / f'{activity}.csv', index=False)

# Merge all activities together
raw_df = dataset.merge_activities()
if raw_df is not None:
    raw_df.to_csv(RESULT_PATH / 'raw.csv', index=False)

DataViz = VisualizeDataset(__file__)
# Resample the data
for granularity in GRANULARITIES:
    resampled_df = dataset.resample(granularity)
    resampled_df.to_csv(RESULT_PATH / f'raw_{granularity}.csv', index=False)
    # Plot all data
    columns = copy.deepcopy(SENSORS)
    columns.append('label')
    DataViz.plot_dataset(resampled_df, columns,
                         ['like', 'like', 'like', 'like', 'like', 'like', 'like'],
                         ['line', 'line', 'line', 'line', 'line', 'line', 'points'])

    # And print a summary of the dataset.q
    util.print_statistics(resampled_df)

print('The code has run through successfully!')