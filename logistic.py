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

from libraries.preprocessing import standardize, StandardScaler, GaussianScaler
from libraries.metrics import min_detection_cost_fun
from libraries.reduction import PCA

from libraries.utils import Writer
from libraries.logistic import (
    LogisticRegression,
    QuadLogisticRegression
)

l_list = [0, 1e-6, 1e-4, 1e-2, 1, 100]


def lr_lambda_search(writer,
                     lr_t: str, data_t: str,
                     X_train, y_train,
                     X_test, y_test,
                     pi=.5):
    lr = QuadLogisticRegression() if lr_t == 'quadratic' \
        else LogisticRegression()

    dcf_scores = []
    progress_bar = tqdm(l_list)
    for l in progress_bar:
        progress_bar.set_description(
            "LR %s | lambda: %f | Data %s | single split | pi %s"
            % (lr_t, l, data_t, pi)
        )

        lr.l_scaler = l
        lr.fit(X_train, y_train)
        _, score = lr.predict(X_test, return_proba=True)
        min_dcf, _ = min_detection_cost_fun(score, y_test, pi)
        writer("lambda: %s | %f" % (l, min_dcf))
        dcf_scores.append(min_dcf)
    return dcf_scores


def split_data_lr(writer, lr_t: 'str', data_t: 'str', X, y, pi=.5):
    writer("----------------")
    writer(" Single split ")
    writer("----------------")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.2)
    return lr_lambda_search(writer, lr_t, data_t, X_train, y_train, X_val, y_val, pi)


def k_fold_lr(writer, lr_t: str,
              data_t: str,
              X, y,
              n_folds=5,
              pi=.5,
              *,
              std=False,
              gauss=False,
              use_pca=False,
              n_components=None):
    writer("----------------")
    writer("5 fold cross validation")
    writer("----------------")

    cv = CrossValidator(n_folds=n_folds)

    transformers = []
    if std:
        transformers.append(StandardScaler())
    elif gauss:
        transformers.append(GaussianScaler())

    if use_pca:
        transformers.append(PCA(n_components=n_components))

    lr = QuadLogisticRegression() if lr_t == 'quadratic' \
        else LogisticRegression()

    dcf_scores = []
    progress_bar = tqdm(l_list)
    for l in progress_bar:
        progress_bar.set_description(
            "LR %s | lambda: %f | Data %s | k fold | pi %s"
            % (lr_t, l, data_t, pi)
        )

        lr.l_scaler = l
        cv.fit(X, y, lr, transformers)
        scores = cv.scores
        min_dcf, _ = min_detection_cost_fun(scores, y, pi)
        writer("lambda: %s | %f" % (l, min_dcf))
        dcf_scores.append(min_dcf)
    writer("----------------")
    return dcf_scores


def save_plots(dcf_raw, dcf_gauss, dcf_std, fig_name):
    plt.figure()
    plt.plot(l_list, dcf_raw)
    plt.plot(l_list, dcf_gauss)
    plt.plot(l_list, dcf_std)
    plt.xscale("log")
    plt.xlabel(r"$\lambda$")
    plt.ylabel("min DCF")
    plt.legend(["Raw", "Gaussianized", "Z-normalized"])
    plt.savefig("report/images/logistic/%s.png" % fig_name)


if __name__ == '__main__':
    n_folds = 5

    X, y = load_fingerprint_train(feats_first=False)
    X_gauss = np.load('results/gaussian_feats.npy').T
    X_std = standardize(X)

    writer = Writer("results/lr_results.txt")

    for pi in [.1, .5, .9]:
        writer("*********** pi = %s ***********\n" % pi)
        writer("----------------")
        writer("LR Type : linear")
        writer("----------------")
        writer("Raw data")
        scores_1_raw = split_data_lr(writer, 'linear', 'raw', X, y, pi)
        scores_5_raw = k_fold_lr(writer, 'linear', 'raw', X, y, n_folds, pi)

        writer("Gaussianized data")
        scores_1_gauss = split_data_lr(writer, 'linear', 'gauss', X_gauss, y, pi)
        scores_5_gauss = k_fold_lr(writer, 'linear', 'gauss', X, y, n_folds, pi, gauss=True)

        writer("Standardized data")
        scores_1_std = split_data_lr(writer, 'linear', 'std', X_std, y, pi)
        scores_5_std = k_fold_lr(writer, 'linear', 'std', X, y, n_folds, pi, std=True)

        save_plots(scores_1_raw, scores_1_gauss, scores_1_std, "lr_single_%s" % pi)
        save_plots(scores_5_raw, scores_5_gauss, scores_5_std, "lr_kfold_%s" % pi)

        writer("\n----------------")
        writer("LR type : quadratic")
        writer("----------------")
        writer("Raw data")
        scores_1_raw = split_data_lr(writer, 'quadratic', 'raw', X, y, pi)
        scores_5_raw = k_fold_lr(writer, 'quadratic', 'raw', X, y, n_folds, pi)

        writer("Gaussianized data")
        scores_1_gauss = split_data_lr(writer, 'quadratic', 'gauss', X_gauss, y, pi)
        scores_5_gauss = k_fold_lr(writer, 'quadratic', 'gauss', X, y, n_folds, pi, gauss=True)

        writer("Standardized data")
        scores_1_std = split_data_lr(writer, 'quadratic', 'std', X_std, y, pi)
        scores_5_std = k_fold_lr(writer, 'quadratic', 'std', X, y, n_folds, pi, std=True)

        save_plots(scores_1_raw, scores_1_gauss, scores_1_std, "qlr_single_%s" % pi)
        save_plots(scores_5_raw, scores_5_gauss, scores_5_std, "qlr_kfold_%s" % pi)
        writer("\n")
    writer.destroy()


    ################# evaluation on test set #################

    X_test, y_test = load_fingerprint_test(feats_first=False)
    gs = GaussianScaler().fit(X)
    sc = StandardScaler().fit(X)

    X_std = sc.transform(X)
    X_test_std = sc.transform(X_test)

    X_gauss = gs.transform(X)
    X_test_gauss = gs.transform(X_test)

    pca7 = PCA(n_components=7).fit(X_std)
    X_pca7 = pca7.transform(X_std)
    X_test_pca7 = pca7.transform(X_test_std)

    pca9 = PCA(n_components=9).fit(X_std)
    X_pca9 = pca9.transform(X_std)
    X_test_pca9 = pca9.transform(X_test_std)

    writer = Writer("results/lr_results_eval.txt")
    for pi in [.1, .5, .9]:
        writer("************* pi = %s *************\n" % pi)
        for lr_t in ["linear", "quadratic"]:
            writer("----------------")
            writer("LR Type : %s" % lr_t)
            writer("----------------")

            writer("----------------")
            writer("Raw data")
            writer("----------------")
            lr_lambda_search(writer, lr_t, 'raw', X, y, X_test, y_test, pi)

            writer("----------------")
            writer("Gaussian data")
            writer("----------------")
            lr_lambda_search(writer, lr_t, 'gauss', X_gauss, y, X_test_gauss, y_test, pi)

            writer("----------------")
            writer("Standardized data")
            writer("----------------")
            lr_lambda_search(writer, lr_t, 'std', X_std, y, X_test_std, y_test, pi)

            writer("----------------")
            writer("Standardized data, PCA(n_components=9)")
            writer("----------------")
            lr_lambda_search(writer, lr_t, 'std', X_pca9, y, X_test_pca9, y_test, pi)

            writer("----------------")
            writer("Standardized data, PCA(n_components=7)")
            writer("----------------")
            lr_lambda_search(writer, lr_t, 'std', X_pca7, y, X_test_pca7, y_test, pi)
            writer("\n\n")
    writer.destroy()
