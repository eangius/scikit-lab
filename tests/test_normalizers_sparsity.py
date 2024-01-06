#!usr/bin/env python


# Internal libraries
from tests.common.random_data import RandomData
from scikitlab.normalizers.sparsity import SparseTransformer, DenseTransformer

# External libraries
from sklearn.base import TransformerMixin
from scipy.sparse import csr_matrix
import pytest
import numpy as np


@pytest.fixture
def dense_mtx():
    return np.array([[1, 0, 2], [0, 0, 3]])


@pytest.fixture
def sparse_mtx():
    return csr_matrix(
        (
            np.array([1, 2, 3, 4, 5, 6]),  # data
            np.array([0, 2, 2, 0, 1, 2]),  # indices
            np.array([0, 2, 3]),  # pointers
        ),
        shape=(2, 3),
    )


# Ensure component is a transformer.
@pytest.mark.unit
@pytest.mark.parametrize(
    "component", [SparseTransformer(), DenseTransformer()], ids=["sparse", "dense"]
)
def test__is_transformer(component):
    assert isinstance(component, TransformerMixin)


# Transforming a dense matrix returns an equivalent sparse one.
@pytest.mark.unit
def test__SparseTransformer01(dense_mtx):
    assert_matrix(dense_mtx, SparseTransformer().transform(dense_mtx), csr_matrix)


# Inverse transforming a sparse matrix returns an equivalent dense one.
@pytest.mark.unit
def test__SparseTransformer02(sparse_mtx):
    assert_matrix(
        sparse_mtx, SparseTransformer().inverse_transform(sparse_mtx), np.ndarray
    )


# Transforming a sparse matrix returns an equivalent dense one.
@pytest.mark.unit
def test__DenseTransformer01(sparse_mtx):
    assert_matrix(sparse_mtx, DenseTransformer().transform(sparse_mtx), np.ndarray)


# Inverse transforming a dense matrix should return an equivalent sparse one.
@pytest.mark.unit
def test__DenseTransformer02(dense_mtx):
    assert_matrix(
        dense_mtx, DenseTransformer().inverse_transform(dense_mtx), csr_matrix
    )


# Transforming with zero sized matrices should work.
@pytest.mark.stress
@pytest.mark.parametrize(
    "component,in_mtx,out_type",
    [
        (SparseTransformer(), RandomData.dense_mtx(0, 0), csr_matrix),
        (DenseTransformer(), RandomData.sparse_mtx(0, 0), np.ndarray),
    ],
    ids=["sparse", "dense"],
)
def test__transform_empty(component, in_mtx, out_type):
    assert_matrix(in_mtx, component.transform(in_mtx), out_type)


# Transforming with large dimensional matrices should still work.
@pytest.mark.stress
@pytest.mark.parametrize(
    "component,in_mtx,out_type",
    [
        (SparseTransformer(), RandomData.dense_mtx(), csr_matrix),
        (DenseTransformer(), RandomData.sparse_mtx(), np.ndarray),
    ],
    ids=["sparse", "dense"],
)
def test__transform_large(component, in_mtx, out_type):
    assert_matrix(in_mtx, component.transform(in_mtx), out_type)


# Helper to assert desired state of a matrix.
def assert_matrix(in_mtx, out_mtx, out_type):
    assert isinstance(out_mtx, out_type)
    assert out_mtx.shape == in_mtx.shape
    assert (out_mtx == in_mtx).all()
