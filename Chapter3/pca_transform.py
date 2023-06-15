from util.util import normalize_dataset
from sklearn.decomposition import PCA

class PrincipalComponentAnalysis:

    pca = []

    def __init__(self):
        self.pca = []

    def determine_pc_explained_variance(self, data_table, cols):
        dt_norm = normalize_dataset(data_table, cols)

        self.pca = PCA(n_components = len(cols))
        self.pca.fit(dt_norm[cols])
        # And return the explained variances.
        return self.pca.explained_variance_ratio_


    def apply_pca(self, data_table, cols, number_comp):
        dt_norm = normalize_dataset(data_table, cols)

        self.pca = PCA(n_components = number_comp)
        self.pca.fit(dt_norm[cols])
        new_values = self.pca.transform(dt_norm[cols])
        for comp in range(0, number_comp):
            data_table['pca_' +str(comp+1)] = new_values[:,comp]
        return data_table