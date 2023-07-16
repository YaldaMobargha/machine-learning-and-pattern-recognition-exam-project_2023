import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from .kernels import rbf_kernel, polynomial_kernel, linear_kernel
from functools import partial
from .._base import Estimator

_valid_kernels = ['linear', 'rbf', 'poly']


class SVClassifier(Estimator):
    
    def __init__(self,
                 kernel='linear',
                 C=1,
                 degree=3,
                 gamma=None,
                 coef=0,
                 csi=1,
                 pi_t=None):
        if kernel == 'linear':
            self.kernel = linear_kernel
        elif kernel == 'rbf':
            self.kernel = partial(rbf_kernel, gamma=gamma, csi=csi)
        elif kernel == 'poly':
            self.kernel = partial(polynomial_kernel, d=degree, gamma=gamma, csi=csi, c=coef)
        else:
            raise ValueError(f"Unknown kernel parameter {kernel}."
                             f"Accepted values are {_valid_kernels}.")

        self.kernel_type = kernel
        self.C = C
        self.csi = csi
        self.W = None
        self.b = None
        self.alpha = None
        self.x = None
        self.z = None
        self.pi_t = pi_t

    def fit(self, X, y):
        self.x = X.T
        n_samples = self.x.shape[1]
        z = np.where(y == 1, 1, -1)
        self.z = z
        DTRc = np.row_stack((self.x, self.csi * np.ones(n_samples)))

        x0 = np.zeros(n_samples)
        bounds = self._check_balance_features(X, y)

        if self.kernel_type == 'linear':
            H = self.kernel(DTRc, DTRc)
        else:
            H = self.kernel(self.x, self.x)
    
        H *= z.reshape(z.shape[0], 1)
        H *= z

        def _dual_kernel_obj(alpha):
            obj = .5 * alpha.T @ H @ alpha
            obj -= sum(alpha)
            grad = H @ alpha - 1
            return obj, grad.reshape(DTRc.shape[1])

        m, _, _ = fmin_l_bfgs_b(_dual_kernel_obj, x0, bounds=bounds,
                                approx_grad=False, factr=10000.)
        self.alpha = m
        res = np.sum(m * z.T * DTRc, axis=1)
        self.W = res[:-1]
        self.b = res[-1]
        return self

    def _check_balance_features(self, X, y):
        n_samples, _ = X.shape
        if self.pi_t is None:
            bounds = [(0, self.C) for _ in range(n_samples)]
        else:  # balance features
            nt = sum(y == 1) / y.shape[0]
            bounds = np.zeros([n_samples, 2])
            Ct = self.C * self.pi_t / nt
            Cf = self.C * (1 - self.pi_t) / (1 - nt)
            bounds[y == 1, 1] = Ct
            bounds[y == 0, 1] = Cf
        return bounds

    def predict(self, X, return_proba=False):
        if self.kernel_type == 'linear':
            score = self.W.T @ X.T + self.b * self.csi
        else:
            score = (self.alpha * self.z) @ self.kernel(self.x, X.T)

        if return_proba:
            return np.where(score > 0, 1, 0), score
        else:
            return np.where(score > 0, 1, 0)
