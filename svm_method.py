import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from libraries.dataset import (
    load_fingerprint_train,
    load_fingerprint_test
)

from libraries.model_selection import (
    train_test_split,
    CrossValidator
)

from libraries.preprocessing import (
    standardize,
    StandardScaler,
    GaussianScaler
)
from libraries.metrics import min_detection_cost_fun

from libraries.utils import Writer
from libraries.svm import SVClassifier

C_list = [.1, 1., 10.]


def single_split_svm(writer, kernel: str, X, y, pi=.5, **kwargs):
    writer("Single Split")
    writer("----------------")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.2)

    dfc_score = []
    dfc_score_balance = []
    progress_bar = tqdm(C_list)
    for c in progress_bar:
        for pi_t in [None, .5]:
            balance = pi_t is not None
            progress_bar.set_description(
                "KERNEL %s | C: %f | balance: %s | single split"
                % (kernel, c, balance)
            )
            svc = SVClassifier(C=c, kernel=kernel, pi_t=pi_t, **kwargs)
            svc.fit(X_train, y_train)

            _, score = svc.predict(X_val, return_proba=True)
            min_dcf, _ = min_detection_cost_fun(score, y_val, pi)
            writer("C: %s | %f | balance %s | pi %s"
                   % (c, min_dcf, balance, pi))
            if balance:
                dfc_score_balance.append(min_dcf)
            else:
                dfc_score.append(min_dcf)

    return dfc_score, dfc_score_balance


def k_fold_svm(writer, k, kernel: str,
               gauss: bool,
               std: bool,
               X, y, pi,
               **kwargs):
    writer("5-fold cross validation")
    writer("----------------")

    transformers = []
    if gauss:
        transformers.append(GaussianScaler())
    elif std:
        transformers.append(StandardScaler())

    cv = CrossValidator(n_folds=k)

    dfc_score = []
    dfc_score_balance = []
    progress_bar = tqdm(C_list)
    for c in progress_bar:
        for pi_t in [None, .5]:
            balance = pi_t is not None
            progress_bar.set_description(
                "KERNEL %s | C: %f | balance: %s | 5-fold | pi %s"
                % (kernel, c, balance, pi)
            )
            svc = SVClassifier(C=c, kernel=kernel, pi_t=pi_t, **kwargs)
            cv.fit(X, y, svc, transformers)
            scores = cv.scores

            min_dcf, _ = min_detection_cost_fun(scores, y, pi)
            writer("C: %s | %f | balance %s" % (c, min_dcf, balance))
            if balance:
                dfc_score_balance.append(min_dcf)
            else:
                dfc_score.append(min_dcf)

    return dfc_score, dfc_score_balance


def svm_eval(writer, kernel: str, data_t: str,
             X, y, X_ts, y_ts, pi=.5, **kwargs):

    progress_bar = tqdm(C_list)
    for c in progress_bar:
        for pi_t in [None, .5]:
            balance = pi_t is not None
            progress_bar.set_description(
                "KERNEL: %s | C: %s | data: %s | balance: %s | pi %s"
                % (kernel, c, data_t, balance, pi)
            )
            svc = SVClassifier(C=c, kernel=kernel, pi_t=pi_t, **kwargs)
            svc.fit(X, y)
            _, score = svc.predict(X_ts, return_proba=True)

            min_dcf, _ = min_detection_cost_fun(score, y_ts, pi)
            writer("C: %s | %f | balance: %s" % (c, min_dcf, balance))


def save_plots(dcf_raw, dcf_gauss, dcf_std, fig_name):
    plt.figure()
    plt.plot(C_list, dcf_raw)
    plt.plot(C_list, dcf_gauss)
    plt.plot(C_list, dcf_std)
    plt.xscale("log")
    plt.xlabel("C")
    plt.ylabel("min DCF")
    plt.legend(["Raw", "Gaussianized", "Z-normalized"])
    plt.savefig("report/images/%s.png" % fig_name)


def save_plots_quad(dcf, dcf_balance, fig_name):
    plt.figure()
    plt.plot(C_list, dcf)
    plt.plot(C_list, dcf_balance)
    plt.xscale("log")
    plt.xlabel("C")
    plt.ylabel("min DCF")
    plt.legend(["no balancing", "balancing"])
    plt.savefig("report/images/%s.png" % fig_name)


def save_plots_rbf(fig_name, *scores):
    plt.figure()
    for score in scores:
        plt.plot(C_list, score)
    plt.xscale("log")
    plt.xlabel("C")
    plt.ylabel("min DCF")
    plt.legend([r"$log \gamma = -1$",
                r"$\log \gamma = -1$, balancing",
                r"$\log \gamma = -2$",
                r"$\log \gamma = -2$, balancing"])
    plt.savefig("report/images/%s.png" % fig_name)


if __name__ == '__main__':
    n_folds = 5

    X, y = load_fingerprint_train(feats_first=False)
    X_test, y_test = load_fingerprint_test(feats_first=False)

    sc = StandardScaler().fit(X)
    X_std = sc.transform(X)
    X_test_std = sc.transform(X_test)

    X_gauss = np.load('results/gaussian_feats.npy').T
    X_test_gauss = np.load('results/gaussian_feats_test.npy').T

    writer = Writer("results/svc_results.txt")

    writer("kernel: linear")
    writer("----------------")

    writer("Raw data")
    scores_1_raw, scores_1_raw_bal = single_split_svm(writer, 'linear', X, y)
    scores_5_raw, scores_5_raw_bal = k_fold_svm(writer, n_folds, 'linear', pi=.5, gauss=False, std=False, X=X, y=y)

    writer("\nGaussianized data")
    scores_1_gauss, scores_1_gauss_bal = single_split_svm(writer, 'linear', X_gauss, y)
    scores_5_gauss, scores_5_gauss_bal = k_fold_svm(writer, n_folds, 'linear', pi=.5, gauss=True, std=False, X=X, y=y)

    writer("\nStandardized data")
    scores_1_std, scores_1_std_bal = single_split_svm(writer, 'linear', X_std, y)
    scores_5_std, scores_5_std_bal = k_fold_svm(writer, n_folds, 'linear', pi=.5, gauss=False, std=True, X=X, y=y)

    save_plots(scores_1_raw, scores_1_gauss, scores_1_std, "svc_lin_single")
    save_plots(scores_5_raw, scores_5_gauss, scores_5_std, "svc_lin_kfold")
    save_plots(scores_1_raw_bal, scores_1_gauss_bal, scores_1_std_bal, "svc_lin_single_bal")
    save_plots(scores_5_raw_bal, scores_5_gauss_bal, scores_5_std_bal, "svc_lin_kfold_bal")

    writer("\n\n----------------")
    writer("kernel : poly (degree=2)")
    writer("----------------")
    writer("Standardized data")
    scores_1, scores_1_bal = single_split_svm(writer, 'poly', X_std, y, gamma=1, degree=2, coef=1)
    scores_5, scores_5_bal = k_fold_svm(writer, n_folds, 'poly', pi=.5, gauss=False, std=True, X=X, y=y,
                                        gamma=1, degree=2, coef=1)

    save_plots_quad(scores_1, scores_1_bal, "quad_svc")
    save_plots_quad(scores_5, scores_5_bal, "quad_svc_kfold")

    writer("\n\n----------------")
    writer("kernel : RBF (log gamma = -1)")
    writer("----------------")
    writer("Standardized data")

    scores_1_m1, scores_1_bal_m1 = single_split_svm(writer, 'rbf', X_std, y, gamma=np.exp(-1), csi=1)
    scores_5_m1, scores_5_bal_m1 = k_fold_svm(writer, n_folds, 'rbf', pi=.5, gauss=False, std=True,
                                              X=X, y=y, gamma=np.exp(-1), csi=1)

    # log gamma = -2
    writer("\n\n----------------")
    writer("kernel : RBF (log gamma = -2)")
    writer("----------------")
    writer("\nStandardized data")
    scores_1_m2, scores_1_bal_m2 = single_split_svm(writer, 'rbf', X_std, y, gamma=np.exp(-2), csi=1)
    scores_5_m2, scores_5_bal_m2 = k_fold_svm(writer, n_folds, 'rbf', pi=.5, gauss=False, std=True, X=X, y=y,
                                              gamma=np.exp(-2), csi=1)

    save_plots_rbf("svc_rbf_single", scores_1_m1, scores_1_bal_m1, scores_1_m2, scores_1_bal_m2)
    save_plots_rbf("svc_rbf_kfold", scores_5_m1, scores_5_bal_m1, scores_5_m2, scores_5_bal_m2)
    writer.destroy()

    
    ################# evaluation on test set #################
    
    writer = Writer("results/svc_results_eval.txt")
    for pi in [.1, .5, .9]:
        writer("************* pi = %s *************\n" % pi)
        writer("----------------")
        writer("kernel : linear")
        writer("----------------")
        writer("Raw data")
        svm_eval(writer, 'linear', 'raw', X, y, X_test, y_test, pi)
        writer("\nGaussianized data")
        svm_eval(writer, 'linear', 'gauss', X_gauss, y, X_test_gauss, y_test, pi)
        writer("\nStandardized data")
        svm_eval(writer, 'linear', 'std', X_std, y, X_test_std, y_test, pi)

        writer("\n\n----------------")
        writer("kernel : poly (degree=2)")
        writer("----------------")
        writer("Standardized data")
        svm_eval(writer, 'poly', 'std', X_std, y, X_test_std, y_test, pi,
                 gamma=1, degree=2, coef=1)

        writer("\n\n----------------")
        writer("kernel : RBF (log gamma = -2)")
        writer("----------------")
        writer("Standardized data")
        svm_eval(writer, 'rbf', 'std', X_std, y, X_test_std, y_test, pi,
                 gamma=np.exp(-2), csi=1)

        writer("\n\n----------------")
        writer("kernel : RBF (log gamma = -1)")
        writer("----------------")
        writer("Standardized data")
        svm_eval(writer, 'rbf', 'std', X_std, y, X_test_std, y_test, pi,
                 gamma=np.exp(-1), csi=1)
        writer("\n")
    writer.destroy()
