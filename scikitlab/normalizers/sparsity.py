#!usr/bin/env python

# External libraries
from sklearn.preprocessing import FunctionTransformer
from scipy.sparse import csr_matrix


class DenseTransformer(FunctionTransformer):
    """
    Converts a sparse matrix into a dense format. This may
    consume lots of memory.
    """

    def __init__(self):
        super().__init__(
            func=DenseTransformer.func,
            inverse_func=SparseTransformer.func,
            accept_sparse=True,
            check_inverse=False,
        )

    @staticmethod
    def func(X, **kwargs):
        return X.todense()


class SparseTransformer(FunctionTransformer):
    """
    Converts a dense matrix of data into a compressed sparse
    row format to preserve memory.
    """

    def __init__(self):
        super().__init__(
            func=SparseTransformer.func,
            inverse_func=DenseTransformer.func,
            accept_sparse=False,
            check_inverse=False,
        )

    @staticmethod
    def func(X, **kwargs):
        return csr_matrix(X)
