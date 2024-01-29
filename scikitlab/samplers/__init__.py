#!usr/bin/env python


# External libraries
from abc import ABC
from imblearn.base import BaseSampler
from overrides import overrides


class ScikitSampler(ABC, BaseSampler):
    """
    Base implementation of scikit transformers that are responsible for
    sampling input data. These are components that modify the volume &
    distribution of data at learn time.
    """

    @property
    def _estimator_type(self):
        return "sampler"

    @overrides
    def fit_resample(self, X, y=None):
        # NOTE: overwritten to bypass strict imblearn checks on non vector
        # (textual or column) input X & classification y output.
        return self._fit_resample(X, y)
