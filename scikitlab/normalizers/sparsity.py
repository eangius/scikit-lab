#!usr/bin/env python

# External libraries
from sklearn.preprocessing import FunctionTransformer
from scipy.sparse import csr_matrix
import numpy as np


class DenseTransformer(FunctionTransformer):
    """
    Converts a sparse matrix into a dense format. This may
    consume lots of memory.
    """

    def __init__(self):
        super().__init__(
            func=self.func,
            inverse_func=SparseTransformer.func,
            accept_sparse=True,
            check_inverse=False,
        )

    @staticmethod
    def func(X, **kwargs):
        return np.asarray(X.todense()) if isinstance(X, csr_matrix) else X


class SparseTransformer(FunctionTransformer):
    """
    Converts a dense matrix of data into a compressed sparse
    row format to preserve memory.
    """

    def __init__(self):
        super().__init__(
            func=self.func,
            inverse_func=DenseTransformer.func,
            accept_sparse=False,
            check_inverse=False,
        )

    @staticmethod
    def func(X, **kwargs):
        return csr_matrix(X) if isinstance(X, np.ndarray) else X
