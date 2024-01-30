#!usr/bin/env python

# Internal libraries
from tests.common.pytest_parametrized import pytest_mark_polymorphic_exclude_series
from tests.common.random_data import RandomData
from scikitlab.samplers.balancing import RegressionBalancer, StrataBalancer

# External libraries
import pytest
import random
import functools
import numpy as np
import pandas as pd


# sampling should add or remove the volume of minority or majority classes
@pytest_mark_polymorphic_exclude_series
@pytest.mark.parametrize("sampling_mode", ["under", "over"])
def test__RegressionBalancer_resample01(input_container, sampling_mode):
    X = input_container(
        [["foo", "bar", "baz"], ["zoo", "boo", "goo"], ["goo", "zoo", "xyz"]]
    )
    y = input_container([1, 2, 3])
    component = RegressionBalancer(
        sampling_mode=sampling_mode,
        fn_classifier=fn_classifier,
    )
    sample_X, sample_y = component.fit_resample(X, y)
    assert_data(X, sample_X, input_container, sampling_mode)
    assert_data(y, sample_y, input_container, sampling_mode)


@pytest.mark.stress
@pytest_mark_polymorphic_exclude_series
@pytest.mark.parametrize("sampling_mode", ["under", "over"])
def test__RegressionBalancer_resample02(input_container, sampling_mode, dataset):
    X, y = dataset
    X = input_container(X)
    y = input_container(y)
    component = RegressionBalancer(
        sampling_mode=sampling_mode,
        fn_classifier=fn_classifier,
    )
    sample_X, sample_y = component.fit_resample(X, y)
    assert_data(X, sample_X, input_container, sampling_mode)
    assert_data(y, sample_y, input_container, sampling_mode)


# sampling from dataframes should preserve column types.
@pytest.mark.unit
@pytest.mark.parametrize("sampling_mode", ["under", "over"])
def test__StrataBalancer_resample01(sampling_mode):
    X = np.array(
        [
            [1, 2, 3, "a"],
            [1, 2, 30, "b"],
            [10, 20, 30, "c"],
            [10, 2, 30, "d"],
        ]
    )
    X = pd.DataFrame(X, columns=["in0", "in1", "in2", "in3"])
    X["in0"] = X["in0"].astype(int)
    X["in1"] = X["in1"].astype(float)
    X["in2"] = X["in2"].astype("category")
    X["in3"] = X["in3"].astype(str)

    y = np.array(
        [
            [111, 1],
            [111, 2],
            [222, 3],
            [222, 4],
        ]
    )
    y = pd.DataFrame(y, columns=["out1", "out2"])
    y["out1"] = y["out1"].astype("category")
    y["out2"] = y["out2"].astype(int)

    strata = ["in1", "in2"]
    component = StrataBalancer(sampling_mode=sampling_mode, columns=strata)
    sample_X, sample_y = component.fit_resample(X, y)
    assert_data(X, sample_X, pd.DataFrame, sampling_mode)
    assert_data(y, sample_y, pd.DataFrame, sampling_mode)
    assert (
        len(set(sample_X.groupby(strata, observed=True).size())) == 1
    )  # equal vol per strata
    assert X.dtypes.sort_index().equals(sample_X.dtypes.sort_index())  # preserve types
    assert y.dtypes.sort_index().equals(sample_y.dtypes.sort_index())  # preserve types


def assert_data(orig, sample, input_container, sampling_mode):
    assert isinstance(
        sample,
        input_container if input_container is not np.array else np.ndarray,
    )

    # ensure sample has more (or less) data than orig.
    if sampling_mode == "over":
        assert sample.shape[0] > orig.shape[0]
    elif sampling_mode == "under":
        assert sample.shape[0] < orig.shape[0]
    else:
        assert False  # undefined!

    if orig.ndim > 1:  # don't check this if orig represents y targets
        assert sample.shape[1] == orig.shape[1]


# helper to triage (random) int or float regress target y into classes.
def fn_classifier(y, n_classes: int = 3):
    return round(1000 * y) % n_classes == 0


@pytest.fixture
def dataset() -> tuple:
    n_samples = random.randrange(5000)
    n_dims = random.randrange(200)

    # fixed size vector generator of random types
    rnd_vtr = functools.partial(
        RandomData.list,
        fn=random.choice([RandomData.string, random.random]),
        n_min=None,
        n_max=n_dims,
    )

    # random number of datapoint vectors
    X = RandomData.list(fn=rnd_vtr, n_min=None, n_max=n_samples)

    # random regression vector
    y = RandomData.list(
        fn=random.random,
        n_min=None,
        n_max=n_samples,
    )
    return X, y
