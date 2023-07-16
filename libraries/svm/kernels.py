import numpy as np


def linear_kernel(xi, xj):
    return xi.T @ xj


def polynomial_kernel(xi, xj, gamma=None, d=3, c=0, csi=0):
    k = xi.T @ xj
    if gamma is None:
        gamma = 1. / xi.shape[1]

    return (gamma * k + c) ** d + csi


def rbf_kernel(xi, xj, gamma=None, csi=0):
    if gamma is None:
        gamma = 1. / xi.shape[1]

    kern = np.zeros([xi.shape[1], xj.shape[1]])
    for i in range(xi.shape[1]):
        for j in range(xj.shape[1]):
            norm = ((xi[:, i] - xj[:, j])**2).sum()
            kern[i, j] = np.exp(-gamma * norm) + csi
    return kern