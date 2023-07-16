import numpy as np

features = ["1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10"]

classes = ["Spoofed fingerprint", "Authentic fingerprint"]

n_feats = len(features)
n_classes = len(classes)


def load_from(file_path: str, *, feats_first: bool = True):
    data = np.loadtxt(file_path,delimiter=",")

    samples = data[:, :-1]
    labels = data[:, -1].astype(np.int32)

    if feats_first:
        return samples.T, labels
    else:
        return samples, labels


def load_fingerprint_train(*, feats_first=True):
    return load_from('projectdataset/Train.txt', feats_first=feats_first)


def load_fingerprint_test(*, feats_first=True):
    return load_from('projectdataset/Test.txt', feats_first=feats_first)
