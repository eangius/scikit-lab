#!usr/bin/env python


# External libraries
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from typing import *
import numpy as np
import pandas as pd
from overrides import overrides


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
        period: int,    # TODO: infer this at fit time
        **kwargs
    ):
        self.period = period
        super().__init__(
            func=self._forward_func,
            inverse_func=self._inverse_func,
            feature_names_out=None,
            check_inverse=False,
            **kwargs
        )
        return

    @overrides
    def get_feature_names_out(self, input_features=None):
        return np.array(["sin", "cos"])

    def _forward_func(self, X, **kwargs):
        angle = (2 * np.pi) * X / self.period
        return np.column_stack(
            (np.sin(angle), np.cos(angle))
        )

    # NOTE: this doesn't quite re-constructs original signal as
    # its absoluteness is lost in the cos & sin periodicity of its
    # parts.
    def _inverse_func(self, X, **kwargs):
        const = self.period / (2 * np.pi)
        return np.hstack(
            (np.arcsin(X), np.arccos(X))
        ) * const


class DateTimeVectorizer(ColumnTransformer):
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
            transformers=transformer_list,
            transformer_weights=transformer_weights,
            n_jobs=kwargs.pop("n_jobs", None),
            verbose=kwargs.pop("verbose", False),
            **kwargs
        )
        return

    @overrides
    def transform(self, X):
        if not X.size:
            return np.ndarray(shape=(0, 2 * len(self.transformer_weights.keys())))

        # convert to pandas to access time parts.
        X = \
            X[0].tolist() if isinstance(X, pd.DataFrame) else \
            X.to_list() if isinstance(X, pd.Series) else \
            X
        X = pd.to_datetime(X)
        X = X.tz_localize(tz='utc', ambiguous='infer') if self.utc_norm else X
        X = pd.DataFrame(X)[0]

        # parse out timestamp into config parts.
        X = np.array([
            [
                DateTimeVectorizer._possible_attributes[name][1](x)  # time parser fn
                for name in self.transformer_weights.keys()
            ]
            for x in X
        ])
        return super().fit_transform(X)  # avoid unfitted checks

    def _build(self) -> Tuple[List, Dict]:
        transformer_list = []
        transformer_weights = dict()
        for i, (field, weight) in enumerate(self.weights.items()):
            if weight != 0:
                feature, _ = self._possible_attributes[field]
                transformer_list.append((field, feature, i))
                transformer_weights[field] = weight
        return transformer_list, transformer_weights

    @staticmethod
    def _validate(weights: Optional[Dict[str, float]]) -> Dict[str, float]:
        # default all attributes if empty
        weights = weights or {
            attr: 1
            for attr in DateTimeVectorizer._possible_attributes.keys()
        }
        # validate all weights
        for k, v in weights.items():
            if k not in DateTimeVectorizer._possible_attributes.keys():
                raise ValueError(f"Unrecognized time attribute: '{k}'")
            if v < 0 or v > 1.0:
                raise ValueError(f"Weight of time attribute '{k}' is out of range")
        return weights

    # supported attributes with their frequency & date-time extraction function.
    _possible_attributes = {
        'season':   (PeriodicityTransformer(period=4), lambda dt: (dt.month % 12 // 3) + 1),  # approx hemisphere independent
        'month':    (PeriodicityTransformer(period=12), lambda dt: dt.month),
        'weekday':  (PeriodicityTransformer(period=7), lambda dt: dt.weekday()),
        'hour':     (PeriodicityTransformer(period=24), lambda dt: dt.hour),
        'minute':   (PeriodicityTransformer(period=60), lambda dt: dt.minute),
        'second':   (PeriodicityTransformer(period=60), lambda dt: dt.second),
        'microsec': (PeriodicityTransformer(period=1000000), lambda dt: dt.microsecond),
    }
