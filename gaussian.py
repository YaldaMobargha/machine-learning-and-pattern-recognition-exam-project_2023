import numpy as np
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
    StandardScaler,
    GaussianScaler,
)

from libraries.metrics import min_detection_cost_fun

from libraries.utils import Writer
from libraries.gaussian import (
    GaussianClassifier,
    TiedGaussian,
    NaiveBayes,
    TiedNaiveBayes
)

from libraries.reduction import PCA


def gaussian_classifiers_eval(
        writer, models,
        X_train, y_train,
        X_test, y_test,
        pi=.5,
        *,
        use_pca=False,
        # ignored if "use_pca" is False
        n_components=None
):
    
    if use_pca:
        pca = PCA(n_components=n_components)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    progress_bar = tqdm(models)
    for model in progress_bar:
        progress_bar.set_description(
            "%s | single split | pi %s" %
            (type(model).__name__, pi)
        )
        model.fit(X_train, y_train)
        _, score = model.predict(X_test, return_proba=True)
        min_dcf, _ = min_detection_cost_fun(score, y_test, pi)
        writer("model: %s \t| dcf: %f"
               % (type(model).__name__, min_dcf))


def single_split_gauss(writer, models, X, y,
                       pi=.5,
                       *,
                       use_pca=False,
                       # ignored if "use_pca" is False
                       n_components=None
                       ):
    
    writer("----------------")
    writer("Single split")
    writer("----------------")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.2)
    gaussian_classifiers_eval(writer, models, X_train, y_train, X_val, y_val, pi,
                              use_pca=use_pca, n_components=n_components)


def k_fold_gauss(writer, models, X, y,
                 n_folds=5,
                 pi=.5,
                 *,
                 std=False,
                 gauss=False,
                 use_pca=False,
                 # ignored if "use_pca" is False
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

    progress_bar = tqdm(models)
    for model in progress_bar:
        progress_bar.set_description(
            "%s | 5-fold cross val | pi %s"
            % (type(model).__name__, pi)
        )
        cv.fit(X, y, model, transformers)
    
        scores = cv.scores
        min_dcf, _ = min_detection_cost_fun(scores, y, pi)
        writer("model: %s \t| dcf: %f"
               % (type(model).__name__, min_dcf))


if __name__ == '__main__':
    n_folds = 5

    X, y = load_fingerprint_train(feats_first=False)

    X_gauss = np.load('results/gaussian_feats.npy').T

    models = [
        GaussianClassifier(),
        NaiveBayes(),
        TiedGaussian(),
        TiedNaiveBayes()
    ]

    writer = Writer("results/gaussian_results.txt")

    for pi in [.1, .5, .9]:
        writer("*********** pi = %s ***********\n" % pi)
        writer("----------------")
        writer("Raw data")
        writer("----------------")
        single_split_gauss(writer, models, X, y, pi)
        k_fold_gauss(writer, models, X, y, n_folds, pi)

        writer("\n----------------")
        writer("Gaussian data")
        writer("----------------")
        single_split_gauss(writer, models, X_gauss, y, pi)
        k_fold_gauss(writer, models, X, y, n_folds, pi, gauss=True)

        writer("\n----------------")
        writer("Gaussian data, PCA(n_components=9)")
        writer("----------------")
        single_split_gauss(writer, models, X_gauss, y, pi, use_pca=True, n_components=9)
        k_fold_gauss(writer, models, X, y, n_folds, pi, gauss=True, use_pca=True, n_components=9)

        writer("\n----------------")
        writer("Gaussian data, PCA(n_components=7)")
        writer("----------------")
        single_split_gauss(writer, models, X_gauss, y, pi, use_pca=True, n_components=7)
        k_fold_gauss(writer, models, X, y, n_folds, pi, gauss=True, use_pca=True, n_components=7)

    writer.destroy()

    ################# evaluation on test set #################

    X_test, y_test = load_fingerprint_test(feats_first=False)
    gs = GaussianScaler().fit(X)

    X_gauss = gs.transform(X)
    X_test_gauss = gs.transform(X_test)

    writer = Writer("results/gaussian_results_eval.txt")
    for pi in [.1, .5, .9]:
        writer("*********** pi = %s ***********\n" % pi)
        writer("----------------")
        writer("Raw data")
        writer("----------------")
        gaussian_classifiers_eval(writer, models, X, y, X_test, y_test, pi)

        writer("\n----------------")
        writer("Gaussian data")
        writer("----------------")
        gaussian_classifiers_eval(writer, models, X_gauss, y, X_test_gauss, y_test, pi)

        writer("\n----------------")
        writer("Gaussian data, PCA(n_components=9)")
        writer("----------------")
        gaussian_classifiers_eval(writer, models, X_gauss, y, X_test_gauss, y_test, pi,
                                  use_pca=True, n_components=9)

        writer("\n----------------")
        writer("Gaussian data, PCA(n_components=7)")
        writer("----------------")
        gaussian_classifiers_eval(writer, models, X_gauss, y, X_test_gauss, y_test, pi,
                                  use_pca=True, n_components=7)

    writer.destroy()

