from pykalman import KalmanFilter
from util.decorator import check_empty_args
import numpy as np

class KalmanFilters:

    @check_empty_args
    def apply_kalman_filter(self, data_table, cols):
        kf = KalmanFilter(transition_matrices=[[1]], observation_matrices=[[1]])

        for col in cols:
            # numpy_array_state = data_table[col].to_numpy(na_value=np.nan)
            numpy_array_state = data_table[col].values
            numpy_array_state = numpy_array_state.astype(np.float32)
            numpy_matrix_state_with_mask = np.ma.masked_invalid(numpy_array_state)
            kf = kf.em(numpy_matrix_state_with_mask, n_iter=5)
            (new_data, filtered_state_covariances) = kf.filter(numpy_matrix_state_with_mask)
            data_table[col + ' kalman'] = new_data
            print(col)
        print(data_table.columns)
        return data_table
