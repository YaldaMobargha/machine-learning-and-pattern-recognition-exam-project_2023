import numpy as np
from scipy.special import logsumexp

from .._base import Estimator
from ..utils.probability import (
    covariance_matrix,
    multivariate_normal_logpdf
)


class GaussianClassifier(Estimator):
    
    def __init__(self):
        self.estimates = []
        self.posterior = None

    def fit(self, X, y):
        return self._fit_helper(X, y, covariance_matrix)

    def _fit_helper(self, X, y, cov_fn: callable):
        self.estimates = []
        y_un, counts = np.unique(y, return_counts=True)
        for label, count in zip(y_un, counts):
            mat = X[y == label, :]
            cov = cov_fn(mat)
            estimate = (
                label,
                np.mean(mat, 0),
                cov,
                count / X.shape[0]
            )
            self.estimates.append(estimate)
        return self

    def predict(self, X, return_proba=False):
        scores = []
        for label, mu, cov, prob in self.estimates:
            distro = multivariate_normal_logpdf(X.T, mu.reshape(-1, 1), cov)
            scores.append(distro)

        joint_mat = np.hstack([value.reshape(-1, 1) for value in scores])
        logsum = logsumexp(joint_mat, axis=1)
        self.posterior = joint_mat - logsum.reshape(1, -1).T

        score = np.exp(self.posterior[:, 1] - self.posterior[:, 0])
        y_pred = np.argmax(self.posterior, axis=1)

        if return_proba:
            return y_pred, score
        return y_pred


class NaiveBayes(GaussianClassifier):
    
    def fit(self, X, y):
        return self._fit_helper(X, y, lambda m: np.diag(np.var(m, 0)))


class TiedGaussian(GaussianClassifier):
    
    def fit(self, X, y):
        super().fit(X, y)
        tied_cov = 1. / y.shape[0] * sum([cov * np.sum(y == label) for label, _, cov, _ in self.estimates])
        self.estimates = [(label, mu, tied_cov, prob) for label, mu, _, prob in self.estimates]
        return self


class TiedNaiveBayes(NaiveBayes):
    
    def fit(self, X, y):
        super().fit(X, y)
        tied_cov = 1. / y.shape[0] * sum([cov * np.sum(y == label) for label, _, cov, _ in self.estimates])
        self.estimates = [(label, mu, tied_cov, prob) for label, mu, _, prob in self.estimates]
        return self
