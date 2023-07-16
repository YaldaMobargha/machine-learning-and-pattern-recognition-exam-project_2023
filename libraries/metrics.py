import numpy as np
import matplotlib.pyplot as plt


def confusion_matrix(y, y_pred):
    n_labels = len(np.unique(y))
    cm = np.zeros((n_labels, n_labels))

    for i in range(n_labels):
        for j in range(n_labels):
            cm[i, j] = sum((y_pred == i) & (y == j))
    return cm


def roc_curve(llr, y):
    sorted_llr = np.sort(llr)
    fpr = np.zeros(sorted_llr.shape[0])
    tpr = np.zeros(sorted_llr.shape[0])

    for i, t in enumerate(sorted_llr):
        cm = confusion_matrix(y, llr > t)
        fnr = cm[0, 1] / (cm[0, 1] + cm[1, 1])
        fpr[i] = cm[1, 0] / (cm[0, 0] + cm[1, 0])
        tpr[i] = 1 - fnr

    return fpr, tpr


def plot_roc_curve(llr, y, label=None):
    fpr, tpr = roc_curve(llr, y)
    plt.xlabel("fpr")
    plt.ylabel("tpr")
    plt.grid(b=True)
    plt.plot(fpr, tpr, label=label)


def accuracy_score(y, y_pred):
    return np.sum(y_pred == y) / len(y)


def f_score(y, y_pred):
    cm = confusion_matrix(y, y_pred)
    return cm[0, 0] / (cm[0, 0] * .5 * (cm[0, 1] + cm[1, 0]))


def bayes_risk(cm, pi=.5, cfn=1, cfp=1):
    fnr = cm[0, 1] / (cm[0, 1] + cm[1, 1])
    fpr = cm[1, 0] / (cm[1, 0] + cm[0, 0])
    return pi * cfn * fnr + (1 - pi) * cfp * fpr


def normalized_bayes_risk(cm, pi=.5, cfn=1, cfp=10):
    dcf_u = bayes_risk(cm, pi, cfn, cfp)
    return dcf_u / min(pi * cfn, (1 - pi) * cfp)


def optimal_bayes_decision(llr, pi=.5, cfn=1, cfp=1):
    threshold = np.log(pi * cfn / ((1 - pi) * cfp))
    return llr > - threshold


def detection_cost_fun(llr, y, pi=.5, cfn=1, cfp=1):
    opt_decision = optimal_bayes_decision(llr, pi, cfn, cfp)
    cm = confusion_matrix(y, opt_decision)

    return normalized_bayes_risk(cm, pi, cfn, cfp)


def min_detection_cost_fun(llr, y, pi=.5, cfn=1, cfp=1):
    min_dcf = float('inf')
    opt_threshold = 0

    for t in np.sort(llr, kind="mergesort"):
        pred = (llr > t).astype(int)
        cm = confusion_matrix(y, pred)
        dcf = normalized_bayes_risk(cm, pi, cfn, cfp)
        if dcf < min_dcf:
            min_dcf = dcf
            opt_threshold = t

    return min_dcf, opt_threshold
