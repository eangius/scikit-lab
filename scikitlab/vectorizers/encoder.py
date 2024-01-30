#!usr/bin/env python

# Internal libraries
from scikitlab.vectorizers import ScikitVectorizer

# External libraries
from overrides import overrides
from functools import cached_property
import numpy as np


class EnumeratedEncoder(ScikitVectorizer):
    """
    Encodes a list of arrays into
    -- encodes a set of dimensions
    -- similar to label encoder but allows Nd array.
    -- -1 reserved for unknown
    -- acts as both input x or output y or multi xy encoder.
    """

    def __init__(self, handle_unknown: str = "error"):
        """
        :param handle_unknown: behaviour when un-known objects or classes are seen.
                               Either `error` or `ignore`
        """
        super().__init__()
        self._clear()
        self.handle_unknown = handle_unknown
        self.n_features = None
        self.n_classes = None
        return

    @overrides
    def fit(self, X, y=None):
        super().fit(X, y)
        self._clear()
        idx = 0
        for arr in X:
            key = arr.tobytes()
            if key not in self._encoding:
                self._encoding[key] = idx
                self._dtypes[idx] = arr.dtype
                idx += 1

        if X.ndim != 2:
            raise ValueError(
                f"Expecting 2d arrays but input is of shape {X.shape} instead"
            )
        self.n_features = X.shape[1]
        self.n_classes = idx
        return self

    @overrides
    def transform(self, X, y=None):
        return np.array(
            [
                self._encoding[key]
                if self.handle_unknown == "error"
                else self._encoding.get(key, -1)
                for key in (arr.tobytes() for arr in X)
            ]
        )

    def inverse_transform(self, X):
        def get_key_tpe(idx) -> tuple:
            if self.handle_unknown == "error":
                key = self._decoding[idx]
                tpe = self._dtypes[idx]
            else:
                key = self._decoding.get(idx, self._default_class)
                tpe = self._dtypes.get(idx, float)
            return key, tpe

        return np.array(
            [
                np.ndarray(buffer=key, dtype=tpe, shape=(self.n_features,))
                for key, tpe in (get_key_tpe(idx) for idx in X)
            ]
        )

    @cached_property
    def _decoding(self) -> dict:
        return {idx: key for key, idx in self._encoding.items()}

    @cached_property
    def _default_class(self):
        return np.full((self.n_features,), np.nan)

    def _clear(self):
        self._encoding = dict()  # key --> idx
        self._dtypes = dict()  # idx --> np.dtype
        return

    @overrides
    def get_feature_names_out(self, input_features=None):
        obj_name = self.__class__.__name__
        obj_id = hex(abs(hash(tuple(self._encoding.items()))))[2:]  # by content
        return np.array([f"{obj_name}_{obj_id}"])
