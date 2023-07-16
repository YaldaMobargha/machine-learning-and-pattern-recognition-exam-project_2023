import numpy as np
from .._base import Estimator
from ..utils.probability import multivariate_normal_logpdf
from scipy.special import logsumexp
from typing import List


def gmm_logpdf(X, gmm: np.ndarray):
    n_feats, n_samples = X.shape
    n_params = len(gmm)

    S = np.empty(shape=(n_params, n_samples))

    for g in range(len(gmm)):
        mean = gmm[g][1]
        cov = gmm[g][2]
        ll = multivariate_normal_logpdf(X, mean, cov)
        S[g, :] = ll + np.log(gmm[g][0])

    marginal_log_density = logsumexp(S, axis=0)
    log_gamma = S - marginal_log_density
    gamma = np.exp(log_gamma)
    return marginal_log_density, gamma


def cov_eig_constraint(cov: np.ndarray, eig_bound: float = .01):
    U, s, _ = np.linalg.svd(cov)
    s[s < eig_bound] = eig_bound
    return U @ (s.reshape(s.size, 1) * U.T)


def em_estimation(X: np.ndarray,
                  gmm: List,
                  psi: float = .01,
                  tol: float = 1e-6,
                  *,
                  tied: bool = False,
                  diag: bool = False):
    n_params = len(gmm)
    n_feats, n_samples = X.shape

    curr_params = np.array(gmm, dtype=object)
    ll_current = np.NaN

    while True:
        ll_previous = ll_current
        marginals, gamma = gmm_logpdf(X, curr_params)

        ll_current = sum(marginals) / n_samples

        if np.abs(ll_current - ll_previous) < tol:
            return curr_params

        Z = np.sum(gamma, axis=1)
        for g in range(n_params):
            F = np.sum(gamma[g] * X, axis=1)
            S = (gamma[g] * X) @ X.T

            mean = (F / Z[g]).reshape(n_feats, 1)
            cov = S / Z[g] - mean @ mean.T
            w = Z[g] / sum(Z)

            if diag:
                cov *= np.eye(cov.shape[0])

            if not tied:
                cov = cov_eig_constraint(cov, psi)
            curr_params[g] = [w, mean, cov]

        if tied:
            tied_cov = sum(Z * curr_params[:, 2]) / n_samples
            tied_cov = cov_eig_constraint(tied_cov, psi)
            curr_params[:, 2].fill(tied_cov)


def lbg_estimation(X: np.ndarray,
                   n_components: int = 2,
                   alpha: float = .1,
                   psi: float = .01,
                   tol: float = 1e-6,
                   *,
                   diag: bool = False,
                   tied: bool = False):
    n = X.shape[0]

    mean = X.mean(axis=1).reshape([n, 1])
    cov = np.cov(X)
    cov = cov_eig_constraint(cov)

    gmm_1 = [(1.0, mean, cov)]

    for _ in range(int(np.log2(n_components))):
        gmm = []
        for param in gmm_1:
            w = param[0] / 2
            mean = param[1]
            cov = param[2]

            U, s, _ = np.linalg.svd(cov)
            d = U[:, 0:1] * s[0] ** 0.5 * alpha

            gmm.append((w, mean + d, cov))
            gmm.append((w, mean - d, cov))

        gmm_1 = em_estimation(X, gmm, psi, tol, diag=diag, tied=tied)

    return gmm_1


class GaussianMixture(Estimator):
    _valid_cov_types = ['full', 'diag', 'full-tied', 'diag-tied']

    def __init__(self,
                 n_components: int = 2,
                 alpha: float = .1,
                 psi: float = .01,
                 *,
                 cov_type: str = 'full',
                 tol: float = 1e-6) -> None:
        if cov_type == 'full':
            self.diag, self.tied = (False, False)
        elif cov_type == 'diag':
            self.diag, self.tied = (True, False)
        elif cov_type == 'full-tied':
            self.diag, self.tied = (False, True)
        elif cov_type == 'diag-tied':
            self.diag, self.tied = (True, True)
        else:
            raise ValueError("Unknown covariance type %s."
                             "Valid types are %s"
                             % (cov_type, self._valid_cov_types))

        self.n_components = n_components
        self.alpha = alpha
        self.psi = psi
        self.tol = tol
        self.gmm_estimates = {}

    def fit(self, X, y):
        n_labels = np.unique(y)
        for label in n_labels:
            x_ = X[y == label, :].T
            params = lbg_estimation(x_,
                                    self.n_components,
                                    alpha=self.alpha,
                                    psi=self.psi,
                                    tol=self.tol,
                                    diag=self.diag,
                                    tied=self.tied)

            self.gmm_estimates[label] = params

        return self

    def predict(self, X, return_proba=False):
        log_densities = []
        for label in self.gmm_estimates:
            gm = self.gmm_estimates[label]
            marginals, _ = gmm_logpdf(X.T, gm)
            log_densities.append(marginals)

        score = log_densities[1] - log_densities[0]
        y_pred = (score >= 0).astype(np.int32)
        if return_proba:
            return y_pred, score

        return y_pred

    def __str__(self):
        return "Tied" * self.tied + \
               "Diag" * self.diag + \
               f"GMM(n_components={self.n_components}, " \
               f"alpha={self.alpha})"
