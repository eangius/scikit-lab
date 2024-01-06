#!usr/bin/env python


# External libraries
from abc import ABC
from sklearn.base import BaseEstimator, TransformerMixin


# ABOUT: This abstract class to scikit transformer interface should be inherited
# by vectorizer components. These are components that transform their input across
# their domain (ie: text-->vector, $US-->vector, ..).
class ScikitVectorizer(ABC, BaseEstimator, TransformerMixin):
    @property
    def _estimator_type(self):
        return "vectorizer"

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X

    # Convenience for sub classes.
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
