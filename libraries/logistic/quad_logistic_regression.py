import numpy as np
from .logistic_regression import LogisticRegression


class QuadLogisticRegression(LogisticRegression):
    def __init__(self, l_scaler=1.):
        super().__init__(l_scaler)

    @staticmethod
    def _map_to_quad_space(X: np.ndarray):
        n_samples, n_feats = X.shape
        X_mapped = np.empty([n_samples, n_feats ** 2 + n_feats])

        for i in range(n_samples):
            x_i = X[i, :].reshape([n_feats, 1])
            mat = x_i @ x_i.T
            mat = mat.flatten("F")
            X_mapped[i, :] = np.vstack([mat.reshape([n_feats ** 2, 1]), x_i])[:, 0]
        return X_mapped

    def fit(self, X, y, initial_guess=None):
        X_2 = QuadLogisticRegression._map_to_quad_space(X)
        return super().fit(X_2, y)

    def predict(self, X, return_proba=False):
        X_2 = QuadLogisticRegression._map_to_quad_space(X)
        return super().predict(X_2, return_proba)
