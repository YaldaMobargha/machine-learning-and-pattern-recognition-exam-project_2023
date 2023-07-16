from abc import ABC, abstractmethod


class NotFittedError(Exception):
    def __init__(self, message=""):
        super().__init__(message)


class Estimator(ABC):
    
    @abstractmethod
    def fit(self, X, y):
        raise NotImplemented("Must have implemented this.")

    @abstractmethod
    def predict(self, X):
        raise NotImplemented("Must have implemented this.")

    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict(X)


class Transformer(ABC):
    
    @abstractmethod
    def fit(self, X, y=None):
        raise NotImplemented("Must have implemented this.")

    @abstractmethod
    def transform(self, X, y=None):
        raise NotImplemented("Must have implemented this.")

    def fit_transform(self, X, y=None):
        if y is None:
            return self.fit(X).transform(X)
        else:
            return self.fit(X, y).transform(X, y)
