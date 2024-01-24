#!usr/bin/env python

# Internal libraries
from scikitlab.metrics.mutual_info import MutualInfoMetric

# External libraries
import numpy as np
import pytest
import scipy
import random


# Unrecognized parameters should fail.
@pytest.mark.unit
def test__mutualinfo_error00():
    with pytest.raises(TypeError):
        MutualInfoMetric(
            random_state=42,  # valid
            invalid_parameter=True,
        )


# Mismatch n_samples of x & y should fail..
@pytest.mark.unit
def test__mutualinfo_error01():
    with pytest.raises(ValueError):
        MutualInfoMetric().score(
            X=np.array(
                [
                    [4, 5, 5],
                    [6, 6, 7],
                    [9, 8, 6],
                ]
            ),
            y=np.array([10, 20, 30, 10, 40]),
        )


# Mixed vector input vector types are not supported.
@pytest.mark.unit
def test__mutualinfo_error02():
    with pytest.raises(ValueError):
        MutualInfoMetric().score(
            X=np.array(
                [
                    [4, 5, 5],
                    [6, 6, 7],
                    [9, 8, 6],
                ]
            ),
            y=np.array([10, 20, 30, 10, 40]),
        )
    with pytest.raises(ValueError):
        MutualInfoMetric().score(
            X=np.array(
                [
                    [4, 5.5, 5 + 1j],
                    [6, 6.6, 7 + 1j],
                    [9, 8.7, 6 + 1j],
                ]
            ),
            y=np.array([10, 20, 30, 10, 40]),
        )


# single input vector x should work
@pytest.mark.unit
def test__mutualinfo_shape01():
    X = np.array([1, 4, 6, 9])
    y = np.array([10, 20, 30, 10])
    assert_scores(
        actual_scores=MutualInfoMetric().score(X, y),
        expected_scores=np.array([0.8421421195050166]),
        n_dims=1,
    )


# multiple y targets as matrix should fail.
@pytest.mark.unit
def test__mutualinfo_shape02():
    with pytest.raises(ValueError):
        MutualInfoMetric().score(
            X=np.array(
                [
                    [1, 2, 3],
                    [4, 5, 5],
                    [6, 6, 7],
                    [9, 8, 6],
                ]
            ),
            y=np.array(
                [
                    [10, 11],
                    [20, 22],
                    [30, 33],
                    [10, 9],
                ]
            ),  # not 1d
        )


# perfect vs no vs inverse vs partial correlation should score accordingly
@pytest.mark.unit
def test__mutualinfo_basic():
    X = np.array(
        [
            [1, 3, 9, -2],
            [4, 3, 6, -2],
            [6, 3, 4, -3],
            [9, 3, 1, -5],
        ]
    )
    y = np.array([1, 4, 6, 9])
    assert_scores(
        actual_scores=MutualInfoMetric().score(X, y),
        expected_scores=np.array(
            [
                1.0,  # perfect
                0.0,  # no correlation
                1.0,  # perfect inverse
                0.8315105705364639,  # partial
            ]
        ),
        n_dims=X.shape[1],
    )


# between discrete vectors
@pytest.mark.unit
def test__mutualinfo_discrete01():
    X = np.array(
        [
            [1, 2, 3],
            [4, 5, 5],
            [6, 6, 7],
            [9, 8, 6],
        ]
    )
    y = np.array([10, 20, 30, 10])
    assert_scores(
        actual_scores=MutualInfoMetric().score(X, y),
        expected_scores=np.array(
            [0.8421421195050166, 0.8096745440221108, 0.7934242079718604]
        ),
        n_dims=X.shape[1],
    )


# unrelated x inputs to y target
@pytest.mark.unit
def test__mutualinfo_discrete02():
    X = np.zeros((4, 3))
    y = np.array([10, 20, 30, 10])
    assert_scores(
        actual_scores=MutualInfoMetric().score(X, y),
        expected_scores=np.zeros((3,)),
        n_dims=X.shape[1],
    )


# unrelated y target to x input
@pytest.mark.unit
def test__mutualinfo_discrete03():
    X = np.array(
        [
            [1, 2, 3],
            [4, 5, 5],
            [6, 6, 7],
            [9, 8, 6],
        ]
    )
    y = np.zeros((4,))
    assert_scores(
        actual_scores=MutualInfoMetric().score(X, y),
        expected_scores=np.zeros((3,)),
        n_dims=X.shape[1],
    )


# random integer data of random size
@pytest.mark.stress
def test__mutualinfo_discrete04():
    n_dim = random.randint(32, 300)
    n_samples = random.randint(500, 3000)

    data = random_discrete(n_samples, n_dim)
    X = data[:, 0 : n_dim - 1]
    y = data[:, -1]

    assert_scores(actual_scores=MutualInfoMetric().score(X, y), n_dims=X.shape[1])


# between real-valued vectors
@pytest.mark.unit
def test__mutualinfo_continuous01():
    X = np.array(
        [
            [1.1, 2.9, 3.1],
            [4.2, 5.8, 5.1],
            [6.3, 6.7, 7.1],
            [9.4, 8.6, 6.1],
        ]
    )
    y = np.array([1.01, 2.02, 3.03, 0.99])
    assert_scores(
        actual_scores=MutualInfoMetric().score(X, y),
        expected_scores=np.array([0, 0, 1.6946262795982085e-16]),
        n_dims=X.shape[1],
    )


# random continuous data of random size
@pytest.mark.stress
def test__mutualinfo_continuous02():
    n_dim = random.randint(32, 300)
    n_samples = random.randint(500, 3000)

    data = random_continuous(n_samples, n_dim)
    X = data[:, 0 : n_dim - 1]
    y = data[:, -1]

    assert_scores(actual_scores=MutualInfoMetric().score(X, y), n_dims=X.shape[1])


# between binary vectors
@pytest.mark.unit
def test__mutualinfo_binary01():
    X = np.array(
        [
            [0, 1, 1],
            [0, 0, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 1, 1],
        ]
    )
    y = np.array([1, 0, 0, 0, 1])
    assert_scores(
        actual_scores=MutualInfoMetric().score(X, y),
        expected_scores=np.array(
            [
                0.039946188043949275,
                0.5299470414358541,
                0.5299470414358541,
            ]
        ),
        n_dims=X.shape[1],
    )


# random one-hot-encoded data of random size
@pytest.mark.stress
def test__mutualinfo_binary02():
    n_dim = random.randint(32, 300)
    n_samples = random.randint(500, 3000)

    data = random_binary(n_samples, n_dim)
    X = data[:, 0 : n_dim - 1]
    y = data[:, -1]

    assert_scores(actual_scores=MutualInfoMetric().score(X, y), n_dims=X.shape[1])


# mixing continuous x with discrete y
@pytest.mark.unit
def test__mutualinfo_mix01():
    n_dim = 32
    n_samples = 100

    X = random_continuous(n_samples, n_dim)
    y = random_discrete(n_samples)

    assert_scores(actual_scores=MutualInfoMetric().score(X, y), n_dims=X.shape[1])


# mixing discrete x with continuous y
@pytest.mark.unit
def test__mutualinfo_mix02():
    n_dim = 32
    n_samples = 100

    X = random_discrete(n_samples, n_dim)
    y = random_continuous(n_samples)

    assert_scores(actual_scores=MutualInfoMetric().score(X, y), n_dims=X.shape[1])


# mixing sparse x with continuous y
@pytest.mark.unit
def test__mutualinfo_mix03():
    n_dim = 32
    n_samples = 100

    X = random_discrete_sparse(n_samples, n_dim)
    y = random_continuous(n_samples)

    assert_scores(actual_scores=MutualInfoMetric().score(X, y), n_dims=X.shape[1])


def assert_scores(
    actual_scores: np.array, n_dims: int, expected_scores: np.array = None
):
    assert isinstance(actual_scores, np.ndarray)
    assert actual_scores.shape == (n_dims,)  # one score per dimension in X
    if expected_scores is not None:
        assert np.array_equal(actual_scores, expected_scores)  # actual values
    else:
        assert np.all(
            (actual_scores >= 0) & (actual_scores <= 1)
        )  # all actual_scores between 0-100%


def random_discrete(
    n_samples: int, n_dim: int = None, v_low: int = None, v_high: int = None
) -> np.ndarray:
    return np.random.randint(
        v_low or random.randint(-20, -1),
        v_high or random.randint(1, 20),
        (n_samples, n_dim) if n_dim else (n_samples,),
    )


def random_continuous(n_samples: int, n_dim: int = None) -> np.ndarray:
    return (
        np.random.rand(n_samples, n_dim)
        if n_dim
        else np.random.rand(
            n_samples,
        )
    )


def random_binary(n_samples: int, n_dim: int = None) -> np.ndarray:
    return np.eye(n_dim)[np.random.choice(n_dim, n_samples)]


def random_discrete_sparse(n_samples, n_dim):
    return scipy.sparse.random(n_samples, n_dim, density=0.5, format="csr", dtype=int)
