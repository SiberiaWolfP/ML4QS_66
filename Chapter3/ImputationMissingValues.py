import pandas as pd
class ImputationMissingValues:

    def impute_mean(self, dataset, cols):
        dataset[cols] = dataset[cols].fillna(dataset[cols].mean())
        return dataset

    def interpolate_linear(self, dataset, cols):
        dataset[cols] = dataset[cols].interpolate(method='linear', limit_direction='forward', axis=0)
        return dataset

    def interpolate_missing_values(self, dataset, cols):
        missing_rows = dataset[cols].isnull().any(axis=1)

        missing_ranges = []
        start_index = None
        for index, is_missing in enumerate(missing_rows):
            if is_missing and start_index is None:
                start_index = index
            elif not is_missing and start_index is not None:
                missing_ranges.append((start_index, index))
                start_index = None
        if start_index is not None:
            missing_ranges.append((start_index, len(dataset)))

        for start, end in missing_ranges:
            data_range = dataset.loc[start:end + 1, cols]

            if len(data_range) <= 2:
                # 如果范围内只有1个或2个数据点，则使用前向填充和后向填充
                data_range = data_range.fillna(method='ffill')
                data_range = data_range.fillna(method='bfill')
            else:
                # 否则，使用线性插值填充缺失值
                data_range = data_range.interpolate(limit=1000)

            dataset.loc[start:end + 1, cols] = data_range



        return dataset

    def impute_interpolate(self, dataset, col):
        dataset[col] = dataset[col].interpolate()
        # And fill the initial data points if needed:
        dataset[col] = dataset[col].fillna(method='bfill')
        return dataset
