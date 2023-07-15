import matplotlib.pyplot as plt
import numpy as np
from mlprlib.dataset import (
    load_fingerprint_train,
    load_fingerprint_test,
    classes,
    features
)

import seaborn as sb
from mlprlib.preprocessing import cumulative_feature_rank


def feats_histogram(X, y, label: str = ''):
    for i, feat in enumerate(features):
        plt.figure()
        plt.xlabel(feat)
        plt.title("%s distribution" % feat)
        plt.hist(X[i, y == 0], bins=20, density=True, alpha=.5, edgecolor='black', color='yellow')
        plt.hist(X[i, y == 1], bins=20, density=True, alpha=.5, edgecolor='black', color='blue')
        plt.legend(classes)
        plt.savefig("report/images/" + label + str(i) + ".png")
        plt.close()


def feat_heatmap(X, label: str = ""):
    plt.figure()
    sb.heatmap(np.corrcoef(X), cmap='coolwarm')
    plt.savefig(f'report/images/{label}.png')


def scatter_plots(X, y, label: str = ''):
    n_features = X.shape[0]
    for i in range(n_features):
        for j in range(i+1, n_features):
            plt.figure()
            indices_class0 = np.where(y == 0)[0]
            indices_class1 = np.where(y == 1)[0]
            plt.scatter(X[i, indices_class0], X[j, indices_class0], color='yellow', label=classes[0])
            plt.scatter(X[i, indices_class1], X[j, indices_class1], color='blue', label=classes[1])
            plt.xlabel(features[i])
            plt.ylabel(features[j])
            plt.legend()
            plt.title(f"{features[i]} vs {features[j]} Scatter Plot")
            plt.savefig(f"report/images/{label}_{features[i]}_{features[j]}.png")
            plt.close()


if __name__ == "__main__":
    X_train, y_train = load_fingerprint_train()

    print("ones are %d" % np.sum(y_train))
    print("zeros are %d" % (len(y_train) - np.sum(y_train)))

    # plot histograms of raw features
    feats_histogram(X_train, y_train, "raw_hist")

    X_gauss = cumulative_feature_rank(X_train)
    feats_histogram(X_gauss, y_train, "gauss_hist")

    # features correlations using heatmap
    feat_heatmap(X_gauss, "gauss_feat_heat")
    feat_heatmap(X_gauss[:, y_train == 0], "gauss_feat_heat0")
    feat_heatmap(X_gauss[:, y_train == 1], "gauss_feat_heat1")
    feat_heatmap(X_gauss, "raw_feat_heat")
    feat_heatmap(X_gauss[:, y_train == 0], "raw_feat_heat0")
    feat_heatmap(X_gauss[:, y_train == 1], "raw_feat_heat1")

    np.save("results/gaussian_feats.npy", X_gauss)

    # Gaussianise test data using training
    # data in the comparison
    X_test, _ = load_fingerprint_test()
    X_test_gauss = cumulative_feature_rank(X_test, X_train)
    np.save("results/gaussian_feats_test.npy", X_test_gauss)


    scatter_plots(X_train, y_train, "scatter_plot")

