#!usr/bin/env python

# Internal libraries
from tests.common.random_data import RandomData
from scikitlab.vectorizers.encoder import EnumeratedEncoder

# External libraries
import numpy as np
import pytest


@pytest.mark.unit
def test__EnumeratedEncoder_transform01():
    X_in = np.array(
        [
            ["foo", 9, False],
            ["bar", 8, True],
            ["foo", 9, False],
            ["zoo", 7, True],
        ]
    )
    component = EnumeratedEncoder()
    X_out = component.fit_transform(X_in)
    X_inv = component.inverse_transform(X_out)
    f_names = component.get_feature_names_out()

    assert np.array_equal(X_out, np.array([0, 1, 0, 2]))
    assert np.array_equal(X_inv, X_in)
    assert component.n_classes == 3
    assert component.n_features == X_in.shape[1]
    assert [f.split("_")[0] for f in f_names] == [component.__class__.__name__]


# Fitting with wrong input shapes should fail.
@pytest.mark.unit
def test__EnumeratedEncoder_error00():
    with pytest.raises(ValueError):
        EnumeratedEncoder().fit(
            X=np.array(
                [
                    [
                        ["foo", 9, False],
                        ["bar", 8, True],
                        ["foo", 9, False],
                        ["zoo", 7, True],
                    ]
                ]
            )
        )


# Mapping from unknown values should fail if specified
@pytest.mark.unit
def test__EnumeratedEncoder_error01():
    component = EnumeratedEncoder(handle_unknown="error").fit(
        X=np.array(
            [
                ["foo", 9, False],
                ["bar", 8, True],
                ["foo", 9, False],
                ["zoo", 7, True],
            ]
        )
    )
    with pytest.raises(KeyError):
        component.transform(X=np.array(["aaa", 0, True]))  # unknown class


# Mapping from unknown values should default to -1 index if ignorable.
@pytest.mark.unit
def test__EnumeratedEncoder_error02():
    component = EnumeratedEncoder(handle_unknown="ignore").fit(
        X=np.array(
            [
                ["foo", 9, False],
                ["bar", 8, True],
                ["foo", 9, False],
                ["zoo", 7, True],
            ]
        )
    )
    X_out = component.transform(X=np.array([["aaa", 0, True]]))  # unknown class
    assert np.array_equal(X_out, np.array([-1]))


# Mapping from unknown values should fail if specified
@pytest.mark.unit
def test__EnumeratedEncoder_error03():
    component = EnumeratedEncoder(handle_unknown="error").fit(
        X=np.array(
            [
                ["foo", 9, False],
                ["bar", 8, True],
                ["foo", 9, False],
                ["zoo", 7, True],
            ]
        )
    )
    with pytest.raises(KeyError):
        component.inverse_transform(X=np.array([9999]))  # unknown idx


# Inverse mapping unknown indices should default to empty array if ignorable
@pytest.mark.unit
def test__EnumeratedEncoder_error04():
    component = EnumeratedEncoder(handle_unknown="ignore").fit(
        X=np.array(
            [
                ["foo", 9, False],
                ["bar", 8, True],
                ["foo", 9, False],
                ["zoo", 7, True],
            ]
        )
    )
    assert np.array_equal(
        component.inverse_transform(X=np.array([9999])),  # unknown idx,
        np.array([[np.nan, np.nan, np.nan]]),
        equal_nan=True,
    )


@pytest.mark.integration
def test__EnumeratedEncoder_02():
    X_in = RandomData.dense_mtx()
    component = EnumeratedEncoder()
    X_out = component.fit_transform(X_in)
    X_inv = component.inverse_transform(X_out)
    f_names = component.get_feature_names_out()

    assert isinstance(X_out, np.ndarray)
    assert X_out.ndim == 1  # 1d array
    assert X_out.shape[0] == X_in.shape[0]  # same cardinality
    assert len(set(X_out.tolist())) <= len(X_out.tolist())  # unique classes
    assert 1 <= component.n_classes <= X_in.shape[1]  # possible repeat values
    assert component.n_features == X_in.shape[1]  # preserve dims
    assert [f.split("_")[0] for f in f_names] == [component.__class__.__name__]
    assert isinstance(X_inv, np.ndarray)
    assert np.array_equal(X_inv, X_in)  # restore
