#!usr/bin/env python

# Internal libraries
from source.ml.samplers import ScikitSampler

# External libraries
import warnings
from overrides import overrides
from typing import Callable
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd


class RegressionBalancer(ScikitSampler):
    """
    Over or under samples a regression dataset based on a category mapping
    over the target variables. This is useful when certain ranges in the
    predict regress variable are rare.
    """

    def __init__(
        self,
        sampling_mode: str,         # either "over" or "under" sampling
        fn_classifier: Callable,    # how to triage regressor variable into classes
        random_state: int = 0,      # determinism
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sampling_mode = sampling_mode
        self.fn_classifier = fn_classifier
        self.random_state = random_state
        return

    @overrides
    def _fit_resample(self, X, y):
        # classify output into a temporary target
        y = y.to_numpy().ravel()
        y_cls = pd.Series(y).apply(self.fn_classifier).astype('category')
        sampling_dist = y_cls.value_counts().to_dict()

        # clone minority classes to match the majority
        if self.sampling_mode.lower() == "over":
            target_cls = max(sampling_dist)
            target_n = sampling_dist[target_cls]
            sampler = RandomOverSampler(
                sampling_strategy={cls: target_n for cls in sampling_dist if cls != target_cls},
                random_state=self.random_state
            )

        # delete from majority classes to match the minority
        elif self.sampling_mode.lower() == "under":
            target_cls = min(sampling_dist)
            target_n = sampling_dist[target_cls]
            sampler = RandomUnderSampler(
                sampling_strategy={cls: target_n for cls in sampling_dist if cls != target_cls},
                random_state=self.random_state
            )

        # do nothing
        else:
            warnings.warn(
                f"Unrecognized '{self.sampling_mode}' sampling mode "
                f"in {self.__class__.__name__} will not have any effect."
            )
            return X, y

        # resample & align indices of classified outputs back to original
        X_resample, y_cls = sampler.fit_resample(X, y_cls)
        y_resample = y[sampler.sample_indices_]

        return X_resample, y_resample
