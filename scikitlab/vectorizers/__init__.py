#!usr/bin/env python


# External libraries
from abc import ABC
from sklearn.base import BaseEstimator, TransformerMixin


class ScikitVectorizer(ABC, BaseEstimator, TransformerMixin):
    """
    Base implementation of scikit transformers that are responsible for
    converting input objects into their vectorized form. The input objects
    `X` are domain specific such as: text, currencies, times, enumerations,
    ect & the resulting vectors are numerical representations of these
    objects suitable for machine learning.
    """

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
