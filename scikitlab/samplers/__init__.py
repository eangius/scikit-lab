#!usr/bin/env python


# External libraries
from abc import ABC
from imblearn.base import BaseSampler
from overrides import overrides
from imblearn.utils._validation import ArraysTransformer


# ABOUT: This abstract class to scikit sampler interface should be inherited
# by data transformer components. These are components that transform the
# volume of data-points (ie: over/under/random sampling).
class ScikitSampler(ABC, BaseSampler):
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
