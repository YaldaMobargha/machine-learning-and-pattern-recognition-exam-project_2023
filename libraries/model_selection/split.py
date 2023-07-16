import numpy as np


def train_test_split(X, y, test_size: float = .25, seed: int = 0):
    np.random.seed(seed)
    n_train = int(X.shape[0] * (1. - test_size))

    idx = np.random.permutation(X.shape[0])
    idx_train = idx[0:n_train]
    idx_test = idx[n_train:]

    return X[idx_train, :], X[idx_test, :], y[idx_train], y[idx_test]


def k_fold_indices(X,
                   n_folds: int = 5,
                   *,
                   shuffle: bool = True,
                   seed: int = 0):
    if n_folds <= 1:
        raise ValueError(
            "k-fold cross-validation requires at least one"
            " train/test split by setting n_splits=2 or more,"
            " got n_splits=%s." % n_folds
        )

    n_samples = len(X)
    if shuffle:
        np.random.seed(seed)
        indices = np.random.permutation(n_samples)
    else:
        indices = np.arange(n_samples)

    chunks = np.array_split(indices, n_folds)
    for i in range(n_folds):
        yield (
            np.hstack([chunks[j] for j in range(n_folds) if j != i]),
            chunks[i]
        )


def k_fold_split(X, y,
                 n_folds: int = 5,
                 *,
                 shuffle: bool = True,
                 seed: int = 0):
    for idx_train, idx_test in k_fold_indices(X, n_folds, shuffle=shuffle, seed=seed):
        X_train, X_test = X[idx_train], X[idx_test]
        y_train, y_test = y[idx_train], y[idx_test]
        yield X_train, y_train, X_test, y_test


def leave_one_out_split(X, y,
                        *,
                        shuffle: bool = False,
                        seed: int = 0):
    return k_fold_split(X, y, len(y), shuffle=shuffle, seed=seed)


class KFold:
    def __init__(self,
                 n_folds: int = 5,
                 *,
                 shuffle: bool = True,
                 seed: int = 0
                 ):
        if n_folds <= 1:
            raise ValueError(
                "k-fold cross-validation requires at least one"
                " train/test split by setting n_splits=2 or more,"
                " got n_splits=%s." % n_folds
            )

        self.n_folds = n_folds
        self.shuffle = shuffle
        self.seed = seed

    def split(self, X: np.ndarray):
        return k_fold_indices(X,
                              self.n_folds,
                              shuffle=self.shuffle,
                              seed=self.seed)
