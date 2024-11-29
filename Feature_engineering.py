from sklearn.decomposition import PCA
import numpy as np
import joblib


class FeatureManipulator:
    def __init__(self, train, test):
        self.train = self._ensure_2d(train)
        self.test = self._ensure_2d(test)

    def _ensure_2d(self, data):
        if data is None:
            raise ValueError("Input data cannot be None")
        if data.ndim == 1:
            return data.reshape(-1, 1)
        return data

    def extraction(self):
        self.pca = PCA(n_components=10)
        train = self.pca.fit_transform(self.train)
        test = self.pca.transform(self.test)
        return train, test

    def save(self):
        joblib.dump(self.pca, "pca.joblib")
