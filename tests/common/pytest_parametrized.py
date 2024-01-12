#!usr/bin/env python

# External libraries
import pytest
import numpy as np
import pandas as pd


# Parametrize certain tests to wrap input data in various container types.
pytest_mark_polymorphic = pytest.mark.parametrize(
    "input_container",
    [pd.DataFrame, pd.Series, np.array],
    ids=["pd.DataFrame", "pd.Series", "np.array"],
)

# Some transformers operate on multiple columns & does not make
# sense to encode into series.
pytest_mark_polymorphic_exclude_series = pytest.mark.parametrize(
    "input_container",
    [pd.DataFrame, np.array],
    ids=["pd.DataFrame", "np.array"],
)
