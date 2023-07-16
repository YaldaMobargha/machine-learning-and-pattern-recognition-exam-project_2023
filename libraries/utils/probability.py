import numpy as np
from typing import Union, List


def mean_squared_error(
        y_true: Union[np.ndarray, List], y_pred: Union[np.ndarray, List]
):
    if isinstance(y_true, list):
        y_true = np.asarray(y_true)

    if isinstance(y_pred, list):
        y_pred = np.asarray(y_pred)

    return np.mean((y_true - y_pred) ** 2)


def covariance_matrix(X: np.ndarray):
    n_samples, _ = X.shape
    dc = X - X.mean(axis=0)
    return (dc.T @ dc) / (n_samples - 1)


def within_class_covariance(X, y):
    n_samples, n_feats = X.shape
    sw = np.zeros((n_samples, n_samples))

    for i in range(len(y)):
        selected = X[:, y == i]
        sw += np.cov(selected, bias=True) * float(selected.shape[1])
    return sw / float(n_feats)


def between_class_covariance(X, y):
    n_samples, n_feats = X.shape
    sb = np.zeros((n_samples, n_samples))

    mu = np.row_stack(X.mean(axis=1))
    for i in range(len(y)):
        selected = X[:, y == i]
        muc = np.row_stack(selected.mean(axis=1))
        muc -= mu
        sb += float(selected.shape[1]) * np.dot(muc, muc.T)
    return sb / float(n_feats)


def normal_pdf(X: np.ndarray, mu: float = 0., sigma: float = 1.):
    k = 1 / np.sqrt(2 * np.pi * sigma)
    up = .5 * (X - mu) ** 2 / sigma
    return k * np.exp(up)


def normal_logpdf(X: np.ndarray, mu: float = 0., sigma: float = 1.):
    return -.5 * (np.log(2 * np.pi) - np.log(sigma) - (X - mu) ** 2 / sigma)


def multivariate_normal_logpdf(X: np.ndarray, mu: np.ndarray, cov: np.ndarray):
    M = X.shape[0]
    _, log_det_sigma = np.linalg.slogdet(cov)
    cov_inv = np.linalg.inv(cov)

    quad_term = (X - mu).T @ cov_inv @ (X - mu)

    if X.shape[1] == 1:
        log_n = - .5 * (M * np.log(2 * np.pi) + log_det_sigma + quad_term)
    else:
        log_n = - .5 * (M * np.log(2 * np.pi) + log_det_sigma + np.diagonal(quad_term))
    return log_n
