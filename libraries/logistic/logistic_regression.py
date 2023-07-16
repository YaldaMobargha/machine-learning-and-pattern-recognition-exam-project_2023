import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from .._base import Estimator, NotFittedError


class LogisticRegression(Estimator):
   
    def __init__(self, l_scaler: float = 1., pi_t: float = .5):
        self.l_scaler = l_scaler
        self.pi_t = pi_t
        self.w = None
        self.b = None

    def fit(self, X, y, initial_guess=None):
        if initial_guess is None:
            initial_guess = np.zeros(X.shape[1] + 1)

        def objective_function(v):
            w, b = v[:-1], v[-1]
            regular = self.l_scaler / 2 * np.linalg.norm(w.T, 2) ** 2
            pos_f = X.T[:, y == 1]
            neg_f = X.T[:, y == 0]

            nt = pos_f.shape[1]
            nf = neg_f.shape[1]

            s_pos = w.T @ pos_f + b
            s_neg = w.T @ neg_f + b

            sum_pos = np.sum(np.log1p(np.exp(- s_pos)))
            sum_neg = np.sum(np.log1p(np.exp(s_neg)))

            return regular + (1 - self.pi_t)/nf * sum_neg + self.pi_t / nt * sum_pos

        m, _, _ = fmin_l_bfgs_b(objective_function,
                                initial_guess,
                                approx_grad=True)

        self.w = m[:-1]
        self.b = m[-1]
        return self

    def predict(self, X, return_proba=False):
        if self.w is None:
            raise NotFittedError("This LogisticRegression object"
                                 "is not fitted yet. Call fit before"
                                 " predict")

        score = self.w @ X.T + self.b
        y_pred = score > 0

        if return_proba:
            return y_pred, score
        else:
            return y_pred


