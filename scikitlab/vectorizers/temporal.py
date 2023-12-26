#!usr/bin/env python


# External libraries
from joblib import wrap_non_picklable_objects
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion
from collections.abc import Iterable
from typing import Tuple, Callable
import numpy as np
import pandas as pd


class DateTimeVectorizer(FeatureUnion):
    """
    Trigonometrically encodes cyclical date-time attributes proportional to each of
    their weights. This reduces the curse of high dimensionality from one-hot-encoding
    each of the parts.
    """

    def __init__(
        self,
        season_weight: float = 1.0,
        month_weight: float = 1.0,
        weekday_weight: float = 1.0,
        hour_weight: float = 1.0,
        minute_weight: float = 1.0,
        second_weight: float = 1.0,
        microsec_weight: float = 1.0,
        **kwargs
    ):
        self.season_weight = season_weight
        self.month_weight = month_weight
        self.weekday_weight = weekday_weight
        self.hour_weight = hour_weight
        self.minute_weight = minute_weight
        self.second_weight = second_weight
        self.microsec_weight = microsec_weight

        transformer_list = []
        transformer_weights = dict()
        for weights, feats in [
            self._build("season", season_weight, 4, lambda dt: (dt.month % 12 // 3) + 1),  # approx & hemisphere independent
            self._build("month", month_weight, 12, lambda dt: dt.month),
            self._build("weekday", weekday_weight, 7, lambda dt: dt.weekday),
            self._build("hour", hour_weight, 24, lambda dt: dt.hour),
            self._build("minute", minute_weight, 60, lambda dt: dt.minute),
            self._build("second", second_weight, 60, lambda dt: dt.second),
            self._build("microsecond", microsec_weight, 1000000, lambda dt: dt.microsecond),
        ]:
            transformer_list.extend(feats)
            transformer_weights.update(weights)

        super().__init__(
            transformer_list=transformer_list,
            transformer_weights=transformer_weights,
            n_jobs=kwargs.get("n_jobs"),
            verbose=kwargs.get("verbose", False),
            **kwargs
        )

    @staticmethod
    def _build(lbl: str, weight: float, period: int, fn: Callable) -> Tuple[dict, list]:
        weights = dict()
        features = []
        if weight != 0:
            weights = {
                f"{lbl}_sin": weight,
                f"{lbl}_cos": weight,
            }
            features = [
                (f"{lbl}_sin", DateTimeVectorizer._sin_feature(period, fn)),
                (f"{lbl}_cos", DateTimeVectorizer._cos_feature(period, fn)),
            ]
        return weights, features

    @staticmethod
    def _sin_feature(period: int, fn: Callable):
        return FunctionTransformer(
            func=wrap_non_picklable_objects(
                lambda X: np.sin(2 * np.pi * DateTimeVectorizer._convert(X, fn) / period)
            ),
            feature_names_out='one-to-one',
            check_inverse=False,
        )

    @staticmethod
    def _cos_feature(period: int, fn: Callable):
        return FunctionTransformer(
            func=wrap_non_picklable_objects(
                lambda X: np.cos(2 * np.pi * DateTimeVectorizer._convert(X, fn) / period)
            ),
            feature_names_out='one-to-one',
            check_inverse=False,
        )

    # NOTE: numpy dates don't support extracting date parts!
    # Also column transformer may pass in data frames which
    # complicates shapes
    @staticmethod
    def _convert(X, fn: Callable):
        if isinstance(X, Iterable):
            conv = [fn(dt) for dt in pd.to_datetime(
                X.values if isinstance(X, pd.DataFrame) else X
            )]
        else:
            conv = fn(pd.to_datetime(X))

        return np.array(conv)
