import numpy as np

from .._base import Transformer, NotFittedError
from ..utils import covariance_matrix


class PCA(Transformer):
    def __init__(self, n_components: int = None):
        self.n_components = n_components
        self.P = None

    def fit(self, X, y=None, *, use_svd: bool = False):
        if self.n_components is None:
            n_components = X.shape[0] - 1
        else:
            n_components = self.n_components

        cov = covariance_matrix(X)
        if use_svd:
            eigenvectors, _, _ = np.linalg.svd(cov)
            self.P = eigenvectors[:, 0:n_components]
        else:
            _, eigenvectors = np.linalg.eigh(cov)
            self.P = eigenvectors[:, ::-1][:, 0:n_components]

        return self

    def transform(self, X, y=None):
        if X is None or self.P is None:
            raise NotFittedError("This PCA instance has not been fitted."
                                 "Call fit before calling transform")
        return (self.P.T @ X.T).T
