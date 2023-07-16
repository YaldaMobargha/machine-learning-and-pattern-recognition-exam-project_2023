import numpy as np
from typing import List

from .._base import Transformer, Estimator
from .split import KFold


class CrossValidator:
    def __init__(self, n_folds: int = 5):
        self._n_folds = n_folds
        self._scores = None

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            model: Estimator,
            transformers: List[Transformer] = None,
            *,
            shuffle: bool = True,
            seed: int = 0):
        scores = np.zeros([X.shape[0], ])

        kfold = KFold(n_folds=self.n_folds, shuffle=shuffle, seed=seed)
        for idx_train, idx_test in kfold.split(X):
            X_train, X_test = X[idx_train], X[idx_test]
            y_train, y_test = y[idx_train], y[idx_test]

            if transformers is not None:
                for transformer in transformers:
                    t = transformer.fit(X_train)
                    X_train = t.transform(X_train)
                    X_test = t.transform(X_test)

            model.fit(X_train, y_train)
            _, score = model.predict(X_test, return_proba=True)
            scores[idx_test] = score

        self._scores = scores

    @property
    def scores(self) -> np.ndarray:
        return self._scores

    @property
    def n_folds(self) -> int:
        return self._n_folds
