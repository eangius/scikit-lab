#!usr/bin/env python


# External libraries
from joblib import wrap_non_picklable_objects
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion
from typing import *
from functools import *
import numpy as np
import pandas as pd
from overrides import overrides
from datetime import datetime, timezone


class PeriodicityTransformer(FunctionTransformer):
    """
    Trigonometrically encodes a periodic signal as combination of sine &
    cosine values. This is useful to fairly capture cyclical distances
    between final & start of periods such as for dates. This encoding
    also caps the number of dimensions to 2 acting as a tradeoff between
    reducing the curse of high dimensionality from needing to one-hot-encode
    signal values & having fine approximation from using various radial-
    basis-functions.

    Note that since both sine & cosine intercept the axis twice per period,
    both dimensions are required to precisely disambiguate where along
    the original signal the encoding lies.
    """

    def __init__(
        self,
        period: int,    # <<dbg can we infer at fit?
        fn: Callable,
        **kwargs
    ):
        self.period = period
        self.fn = wrap_non_picklable_objects(fn)
        super().__init__(
            func=self._forward_func,  # <<dbg define inverse func
            feature_names_out=None,
            check_inverse=False,
            **kwargs
        )
        return

    @overrides
    def get_feature_names_out(self, input_features=None):
        return np.array(["sin", "cos"])

    def _forward_func(self, X, **kwargs):
        if not X.size:
            return np.ndarray(shape=(0, 2))

        X = X[0].tolist() if isinstance(X, pd.DataFrame) else X
        X = [self.fn(x) for x in pd.to_datetime(X)]  # <<dbg hack belongs at call site

        angle = 2 * np.pi * np.array(X) / self.period
        return np.array([
            np.sin(angle), np.cos(angle)
        ]).transpose()


class DateTimeVectorizer(FeatureUnion):
    """
    Encodes a date-time into a collection of trigonometrically encoded attribute
    parts. This vectorizer allows to emphasize on certain time attributes by weight
    as well as standardize to a common timezone.
    """

    def __init__(
        self,
        weights: Dict[str, float] = None,  # time attributes, default all
        utc_norm: bool = False,            # converts to coordinated universal time
        **kwargs
    ):
        self.weights = self._validate(weights)
        self.utc_norm = utc_norm

        transformer_list, transformer_weights = self._build()
        super().__init__(
            transformer_list=transformer_list,
            transformer_weights=transformer_weights,
            n_jobs=kwargs.pop("n_jobs", None),
            verbose=kwargs.pop("verbose", False),
            **kwargs
        )
        return

    def _build(self) -> Tuple[List, Dict]:
        transformer_list = []
        transformer_weights = dict()
        for field, weight in self.weights.items():
            if weight != 0:
                feature = self.time_attribute[field]
                transformer_list.append((field, feature))
                transformer_weights[field] = weight
        return transformer_list, transformer_weights

    def _validate(self, weights: Optional[Dict[str, float]]) -> Dict[str, float]:
        # default all attributes if empty
        weights = weights or {
            attr: 1
            for attr in self.time_attribute.keys()
        }
        # validate all weights
        for k, v in weights.items():
            if k not in self.time_attribute.keys():
                raise ValueError(f"Unrecognized time attribute: '{k}'")
            if v < 0 or v > 1.0:
                raise ValueError(f"Weight of time attribute '{k}' is out of range")
        return weights

    # supported attributes with their frequency & date-time extraction function.
    @cached_property
    def time_attribute(self):
        return {
        'season':   PeriodicityTransformer(period=4, fn=wrap_non_picklable_objects(lambda dt: (self._conv(dt).month % 12 // 3) + 1)),  # approx hemisphere independent
        'month':    PeriodicityTransformer(period=12, fn=wrap_non_picklable_objects(lambda dt: self._conv(dt).month)),
        'weekday':  PeriodicityTransformer(period=7, fn=wrap_non_picklable_objects(lambda dt: self._conv(dt).weekday())),
        'hour':     PeriodicityTransformer(period=24, fn=wrap_non_picklable_objects(lambda dt: self._conv(dt).hour)),
        'minute':   PeriodicityTransformer(period=60, fn=wrap_non_picklable_objects(lambda dt: self._conv(dt).minute)),
        'second':   PeriodicityTransformer(period=60, fn=wrap_non_picklable_objects(lambda dt: self._conv(dt).second)),
        'microsec': PeriodicityTransformer(period=1000000, fn=wrap_non_picklable_objects(lambda dt: self._conv(dt).microsecond)),
    }

    def _conv(self, dt: datetime):
        return dt.astimezone(timezone.utc) if self.utc_norm else dt
