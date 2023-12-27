#!usr/bin/env python

# Internal libraries
from scikitlab.vectorizers.temporal import *

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
def test__DateTimeVectorizer02():
    with pytest.raises(ValueError):
        DateTimeVectorizer(weights={
            "season": -1,       # too small
            "month": 1.23,      # too large
        })


@pytest.mark.unit
def test__DateTimeVectorizer03(X):
    component = DateTimeVectorizer(weights={"month": 1})
    vtrs = component.transform(X)
    assert isinstance(vtrs, np.ndarray)
    assert vtrs.shape == (X.shape[0], 2)
    assert np.all((vtrs >= -1) & (vtrs <= 1))


@pytest.fixture
def X():
    return pd.DataFrame([datetime.datetime.now()])


@pytest.fixture
def component():
    return DateTimeVectorizer(weights={
        "season": random.random(),
        "month": random.random(),
        "weekday": random.random(),
        "hour": random.random(),
        "minute": random.random(),
        "second": random.random(),
        "microsec": random.random(),
    })
