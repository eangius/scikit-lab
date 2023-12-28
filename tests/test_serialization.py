#!usr/bin/env python

# Internal libraries
from scikitlab.normalizers.sparsity import *
from scikitlab.vectorizers.temporal import *
from scikitlab.vectorizers.spatial import *

# External libraries
import pytest
import joblib


components = [
    SparseTransformer(),
    DenseTransformer(),
    DateTimeVectorizer(),
    GeoVectorizer(resolution=1),
]


# Serializing component state should be preserved when re-loaded.
@pytest.mark.integration
@pytest.mark.parametrize(
    "component", components,
    ids=[component.__class__.__name__ for component in components]
)
def test__joblib_savability(component, tmp_path):
    filename = tmp_path / f"{component.__class__.__name__}.pkl"

    with open(filename, 'wb') as file:
        joblib.dump(component, file)

    with open(filename, 'rb') as file:
        comp_out = joblib.load(file)

    assert type(comp_out) is type(component)
    assert comp_out.get_params() == component.get_params()
