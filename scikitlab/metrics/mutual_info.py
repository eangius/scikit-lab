#!usr/bin/env python


# External libraries
from enum import Enum
from scipy.stats import entropy
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.utils import check_X_y
from typing import Optional
import numpy as np
import scipy


class VectorType(Enum):
    DISCRETE = 1  # all binary, ordinal
    CONTINUOUS = 2  # all real values
    IMAGINARY = 3  # all with imaginary parts
    ZEROS = 4  # unknown all zeros (discrete or cont)


class MutualInfoMetric:
    """
    Metric calculating the normalized mutual-information between random variables.
    This is a symmetric value indicating the 0-1 degree of dependence between two
    signals. This metric is more general than correlation in that can detect non-linear
    relationships without assumptions as well as work with continuous or discrete
    variables. This implementation wraps around scikit-learn's mutual info estimate
    calculations & can be treated as a learnable component to tune for hyperparameters.
    """

    def __init__(self, y: Optional[np.array] = None, **kwargs):
        """
        :param y: Vector of shape `(n_samples,)` to measure against. Useful to optimize
                  computations when feature selecting against a constant target.
        :param kwargs: other parameters as per scikits mutual_info_regression/classif
        """

        # optional configurable parameters
        self.kwargs = kwargs
        if any(
            arg not in {"discrete_features", "n_neighbors", "copy", "random_state"}
            for arg in kwargs
        ):
            raise TypeError("Unrecognized arguments")

        # learnable parameters
        self.fn_mutual_info, self.y_entropy, self.y = (
            self._set(y) if y is not None else (None, None, None)
        )
        return

    def __call__(self, X, y: Optional[np.array] = None) -> np.array:
        """
        Evaluate this metric between each feature in X & the target y.
        :param X: Vectors of shape `(n_samples, n_features)` to individually score
                  in batch against target `y` vector.
        :param y: Vector of shape `(n_samples, n_features)` to measure against.
                  Uses pre-computed y if not specified.
        :return:  Array of normalized mutual information scores between each of
                  the `X` features & `y` target of shape `(n_features,)`
        """

        # validate inputs
        if y is None and self.y is None:
            raise ValueError(
                "No target vector y was specified at metric "
                "initialization or at calculation time."
            )
        check_X_y(X, y if y is not None else self.y, accept_sparse=True)

        # Use precomputed y when not specified, else temporarily
        # compute new y target.
        if y is None:
            fn_mutual_info = self.fn_mutual_info
            y_entropy = self.y_entropy
            y = self.y
        else:
            fn_mutual_info, y_entropy, y = self._set(y)

        # Ensure consistency in X.
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        vtr_types = [self._vector_type(vtr) for vtr in X]
        if len({t for t in vtr_types if t != VectorType.ZEROS}) > 1:
            raise ValueError(
                f"X mixes vector dimensions types {set(vtr_types)}. "
                "Ensure all dimensions resolve to 'discrete' or 'continuous' "
                "vectors or pass one dimension at a time."
            )

        # Bulk calculate for each X dimension with respects to y
        scores = fn_mutual_info(
            X,
            y=y,
            discrete_features=all(
                t in {VectorType.DISCRETE, VectorType.ZEROS} for t in vtr_types
            ),
            **self.kwargs,
        )

        # Normalize to 0-1 score, neutralizing NaNs
        hX = np.asarray(entropy(X)).ravel()
        hY = np.repeat(y_entropy, X.shape[1])
        scores = np.nan_to_num(scores / np.mean([hX, hY], axis=0))

        # TODO:
        #  - implement adjusted mutual info to handle noise!
        #  - perform early outs to avoid computing entropy or mutual info
        #  - handle or restrict multi dimensional y
        return scores

    # Alias for explicitly calling the metric
    score = __call__

    # check vector type by their values rather than type: ie: 1.0 = Int
    @staticmethod
    def _vector_type(vtr: np.array) -> VectorType:
        vtr = np.asarray(vtr.todense()).ravel() if scipy.sparse.issparse(vtr) else vtr

        if not np.any(vtr):
            return VectorType.ZEROS  # unknown continuous or discrete

        if np.all(np.iscomplex(vtr)):
            return VectorType.IMAGINARY

        if np.all(np.isreal(vtr)):
            if any(abs(dim - int(dim)) > 0 for dim in vtr):
                return VectorType.CONTINUOUS

            if all(dim.is_integer() for dim in vtr):
                return VectorType.DISCRETE

        raise NotImplementedError(f"Unrecognized vector type: {vtr.dtype}")

    @staticmethod
    def _set(y: np.array):
        y_entropy = entropy(y)
        fn_mutual_info = (
            mutual_info_regression
            if MutualInfoMetric._vector_type(y) == VectorType.CONTINUOUS
            else mutual_info_classif
        )
        return fn_mutual_info, y_entropy, y
