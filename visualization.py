import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd

point_displays = ['+', 'x']  # '*', 'd', 'o', 's', '<', '>']
line_displays = ['-']  # , '--', ':', '-.']

parser = argparse.ArgumentParser()

# Add the parameters
parser.add_argument('-f', type=str, required=True, help="Name of the csv file to be visualized")

args = parser.parse_args()

if args.f:
    # Plot all data
    df = pd.read_csv(args.f)

    df = df.reset_index()
    # df['Time (s)'] = pd.to_datetime(df['Time (s)'], format='%Y-%m-%d %H:%M:%S.%f')
    df['time_diff'] = df['time'].diff()
    # df['Timestamp'] = pd.to_datetime(df['Time (s)'], unit='s')

    threshold = 5 * 10 ** 9  # 5 seconds

    gaps = df.index[df['time_diff'] > threshold].to_list()
    gaps = [0] + gaps + [len(df)]

    columns_name = ['Accelerometer', 'Gravity', 'Gyroscope', 'Location', 'Magnetometer', 'Microphone', 'Orientation', 'label']
    display = ['line', 'line', 'line', 'line', 'line', 'line', 'line', 'points']
    fig, axs = plt.subplots(len(columns_name), 1, figsize=(15, 12), sharex='all', sharey='none')

    for idx, column_name in enumerate(columns_name):
        # Get the columns for each sensor
        columns = [col for col in df.columns if col.startswith(column_name)]
        if column_name == 'Location':
            columns = ['Location latitude', 'Location longitude', 'Location altitude', 'Location speed']

        # axs[idx].set_prop_cycle(color=['b', 'g', 'r', 'c', 'm', 'y', 'k'])

        max_values = []
        min_values = []
        lines = []

        for j, column in enumerate(columns):

            # for i in range(len(gaps) - 1):
            #     start, end = gaps[i], gaps[i + 1]
            #
            #     if i < len(gaps) - 2:
            #         axs[idx].axvline(x=df['index'].iloc[end], color='red', linestyle='dashed')
            mask = np.isfinite(df[column])  # Create a mask where values are not NaN
            max_values.append(df[column][mask].max())
            min_values.append(df[column][mask].min())
            # Display point, or as a line
            line = None
            if display[idx] == 'points':
                # line, = axs[idx].plot(df.index[mask], df[column][mask], point_displays[j % len(point_displays)], linewidth=0.5)
                line, = axs[idx].plot(df.index.values[mask], df[column].values[mask], point_displays[j % len(point_displays)], linewidth=0.5)
            else:
                # line, = axs[idx].plot(df.index[mask], df[column][mask], line_displays[j % len(line_displays)], linewidth=0.5)
                line, = axs[idx].plot(df.index.values[mask], df[column].values[mask], line_displays[j % len(line_displays)], linewidth=0.5)
            lines.append(line)

        axs[idx].tick_params(axis='y', labelsize=10)
        axs[idx].legend(lines, columns, fontsize='xx-small', numpoints=1, loc='upper center',
                        bbox_to_anchor=(0.5, 1.3), ncol=len(columns), fancybox=True, shadow=True)

        axs[idx].set_ylim([min(min_values) - 0.1 * (max(max_values) - min(min_values)),
                           max(max_values) + 0.1 * (max(max_values) - min(min_values))])

    # Make sure we get a nice figure with only a single x-axis and labels there.
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    plt.xlabel('index')
    fig.tight_layout()
    plt.show()
