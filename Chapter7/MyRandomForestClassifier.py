from cuml import RandomForestClassifier


class MyRandomForestClassifier(RandomForestClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def predict(self, X):
        y_pred = super().predict(X)
        print(y_pred)
        print(type(y_pred))
        return y_pred
