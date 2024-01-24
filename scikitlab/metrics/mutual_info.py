#!usr/bin/env python


# External libraries
from enum import Enum, auto
from scipy.stats import entropy
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.utils import check_X_y
from typing import Optional
import numpy as np
import scipy


class VectorType(Enum):
    DISCRETE = auto()  # all binary, ordinal
    CONTINUOUS = auto()  # all real values
    COMPLEX = auto()  # all with imaginary parts
    ZEROS = auto()  # unknown all zeros (discrete or cont)

    def __repr__(self):
        return self.name


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

        # Use precomputed y when not specified, else temporarily
        # compute new y target.
        if y is None:
            fn_mutual_info = self.fn_mutual_info
            y_entropy = self.y_entropy
            y = self.y
        else:
            fn_mutual_info, y_entropy, y = self._set(y)

        X = X.reshape(-1, 1) if X.ndim == 1 else X
        check_X_y(X, y, accept_sparse=True)

        # Ensure type consistency in X.
        vtr_types = (
            [self._vector_type(X.ravel())]
            if X.ndim == 1
            else [self._vector_type(vtr) for vtr in X]
        )
        if len({t for t in vtr_types if t != VectorType.ZEROS}) > 1:
            raise ValueError(
                f"X mixes vector dimensions types {set(vtr_types)}. "
                "Ensure all resolve to 'discrete' or 'continuous' "
                "vectors or pass one dimension at a time."
            )

        # Bulk calculate for each X dimension with respects to y
        mi = fn_mutual_info(
            X,
            y=y,
            discrete_features=all(
                t in {VectorType.DISCRETE, VectorType.ZEROS} for t in vtr_types
            ),
            **self.kwargs,
        )

        # Normalize raw mutual info scores (estimates) to 0-1 score as per
        # https://course.ccs.neu.edu/cs6140sp15/7_locality_cluster/Assignment-6/NMI.pdf
        with np.errstate(over="ignore"):
            n_features = X.shape[1]
            hX = np.nan_to_num(np.asarray(entropy(X)).ravel())
            hy = np.nan_to_num(np.repeat(y_entropy, n_features))
            hTotal = hX + hy
            scores = 2 * np.divide(
                mi,
                hTotal,
                out=np.ones(n_features),
                where=(hTotal != 0) & (hTotal != np.NaN),
            )
        scores = np.minimum(scores, np.ones(n_features))  # cap max at 100%
        scores = np.maximum(scores, np.zeros(n_features))  # cap min at 0%

        # TODO:
        #  - implement adjusted mutual info to handle noise!
        #  - perform early outs to avoid computing entropy or mutual info
        return scores

    # Alias for explicitly calling the metric
    score = __call__

    # check vector type by their values rather than type: ie: 1.0 = Int
    @staticmethod
    def _vector_type(vtr: np.array) -> VectorType:
        vtr = np.asarray(vtr.todense()).ravel() if scipy.sparse.issparse(vtr) else vtr

        if not np.any(vtr):
            return VectorType.ZEROS  # unknown if continuous or discrete

        if np.any(np.iscomplex(vtr)):
            return VectorType.COMPLEX

        if np.all(np.isreal(vtr)):
            if all(abs(dim - int(dim)) == 0 for dim in vtr):  # or dim.is_integer()
                return VectorType.DISCRETE
            else:
                return VectorType.CONTINUOUS

        raise NotImplementedError(f"Unrecognized vector type: {vtr.dtype}")

    @staticmethod
    def _set(y: np.array):
        if y.ndim != 1:
            raise ValueError(
                f"Invalid target vector y of shape {y.shape} " "is not 1 dimensional."
            )

        y_entropy = 0 if not np.all(y) else entropy(y)
        fn_mutual_info = (
            mutual_info_regression
            if MutualInfoMetric._vector_type(y) == VectorType.CONTINUOUS
            else mutual_info_classif
        )
        return fn_mutual_info, y_entropy, y
