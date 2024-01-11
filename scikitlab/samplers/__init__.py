#!usr/bin/env python


# External libraries
from abc import ABC
from imblearn.base import BaseSampler
from overrides import overrides
from imblearn.utils._validation import ArraysTransformer


class ScikitSampler(ABC, BaseSampler):
    """
    Base implementation of scikit transformers that are responsible for
    sampling input data. These are components that modify the volume &
    distribution of data at learn time.
    """

    @property
    def _estimator_type(self):
        return "sampler"

    # NOTE: method copied & simplified from imblearn to bypass strict checks on non
    # vector (textual or column) input X, & classification y output.
    @overrides
    def fit_resample(self, X, y):
        output = self._fit_resample(X, y)
        X_, y_ = ArraysTransformer(X, y).transform(output[0], output[1])
        return (X_, y_) if len(output) == 2 else (X_, y_, output[2])
