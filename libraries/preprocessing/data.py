from typing import Union

import numpy as np
from scipy.stats import norm
from .._base import Transformer, NotFittedError


def standardize(X: np.ndarray, mean: float = None, std: float = None):
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
    return (X - mean) / std


def normalize(X: np.ndarray,
              axis: int = -1,
              order: Union[str, int] = 2):
    norm = np.linalg.norm(X, order, axis)
    l_norm = np.atleast_1d(norm)
    l_norm[l_norm == 0] = 1
    return X / np.expand_dims(l_norm, axis)


def cumulative_feature_rank(X: np.ndarray, X_ref: np.ndarray = None):
    n_feats, n_samples = X.shape
    transformed = np.empty([n_feats, n_samples])

    X_ref = X if X_ref is None else X_ref
    _, N = X_ref.shape
    for i in range(n_samples):
        rank = 1. + (X[:, i].reshape([n_feats, 1]) < X_ref).sum(axis=1)
        rank /= (N + 2.)
        transformed[:, i] = norm.ppf(rank)
    return transformed


class StandardScaler(Transformer):
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X, y=None):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        return self

    def transform(self, X, y=None):
        if self.mean is None or self.std is None:
            raise NotFittedError("This StandardScaler is not"
                                 "fitted yet. Call fit first.")
        return standardize(X, self.mean, self.std)


class GaussianScaler(Transformer):
    def __init__(self):
        self.X_ref = None

    def fit(self, X, y=None):
        self.X_ref = X.T
        return self

    def transform(self, X, y=None):
        if self.X_ref is None:
            raise NotFittedError("This GaussianScaler is not"
                                 "fitted yet. Call fit first.")
        return cumulative_feature_rank(X.T, self.X_ref).T
