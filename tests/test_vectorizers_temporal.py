#!usr/bin/env python

# Internal libraries
from tests.common.pytest_parametrized import pytest_mark_polymorphic
from tests.common.random_data import RandomData
from scikitlab.vectorizers.temporal import DateTimeVectorizer

# External libraries
import pytest
import numpy as np
import random
from sklearn.base import TransformerMixin
import datetime


# Ensure component is a transformer.
@pytest.mark.unit
def test__is_transformer(component):
    assert isinstance(component, TransformerMixin)


# Invalid weights will raise exception.
@pytest.mark.unit
def test__DateTimeVectorizer_error01():
    with pytest.raises(ValueError):
        DateTimeVectorizer(
            weights={
                "season": -1,  # too small
                "month": 1.23,  # too large
            }
        )


# Empty inputs should yield empty vectors
@pytest.mark.unit
@pytest_mark_polymorphic
def test__DateTimeVectorizer_transform01(input_container):
    X = input_container([])
    component = DateTimeVectorizer(weights={"month": 1})
    assert_vectors(
        vtrs=component.fit_transform(X),
        n_samples=X.shape[0],
        n_dims=2 * len(component.weights),
    )
    assert_feature_names(
        features=component.get_feature_names_out(), attributes=component.weights.keys()
    )


# Vectorizing one time part should yield 2 dimensions per part
@pytest.mark.unit
@pytest_mark_polymorphic
def test__DateTimeVectorizer_transform02(input_container):
    X = input_container([datetime.datetime.now()])
    component = DateTimeVectorizer(weights={"month": 1})
    assert_vectors(vtrs=component.fit_transform(X), n_samples=X.shape[0], n_dims=2)
    assert_feature_names(
        features=component.get_feature_names_out(), attributes=component.weights.keys()
    )


# Vectorizing time parts as weighted should
@pytest.mark.unit
@pytest_mark_polymorphic
def test__DateTimeVectorizer_transform03(input_container):
    X = input_container([datetime.datetime.now()])
    component = DateTimeVectorizer(weights={"month": 0.9, "weekday": 0.5})
    assert_vectors(
        vtrs=component.fit_transform(X),
        n_samples=X.shape[0],
        n_dims=2 * len(component.weights),
    )
    assert_feature_names(
        features=component.get_feature_names_out(), attributes=component.weights.keys()
    )


@pytest.mark.stress
@pytest_mark_polymorphic
def test__DateTimeVectorizer_transform04(input_container, component):
    X = input_container(RandomData.list(RandomData.date))
    assert_vectors(
        vtrs=component.fit_transform(X),
        n_samples=X.shape[0],
        n_dims=2 * len(component.weights),
    )
    assert_feature_names(
        features=component.get_feature_names_out(), attributes=component.weights.keys()
    )


def assert_vectors(vtrs, n_samples, n_dims):
    assert isinstance(vtrs, np.ndarray)
    assert vtrs.shape == (n_samples, n_dims)
    assert np.all((vtrs >= -1) & (vtrs <= 1))


# Ensure all time attributes requested are have 2 signals
def assert_feature_names(features, attributes):
    assert isinstance(features, np.ndarray)
    assert len(features) == len(attributes) * 2
    assert set(features) == {
        f"{attr}__{trig}" for attr in attributes for trig in {"sin", "cos"}
    }


@pytest.fixture
def component():
    return DateTimeVectorizer(
        utc_norm=bool(random.getrandbits(1)),
        weights={
            "season": random.random(),
            "month": random.random(),
            "weekday": random.random(),
            "hour": random.random(),
            "minute": random.random(),
            "second": random.random(),
            "microsec": random.random(),
        },
    )
