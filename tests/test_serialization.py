#!usr/bin/env python

# Internal libraries
from scikitlab.normalizers.sparsity import SparseTransformer, DenseTransformer
from scikitlab.vectorizers.temporal import PeriodicityTransformer, DateTimeVectorizer
from scikitlab.vectorizers.spatial import GeoVectorizer
from scikitlab.vectorizers.text import WeightedNgramVectorizer, UniversalSentenceEncoder

# External libraries
import pytest
import joblib


components = [
    SparseTransformer(),
    DenseTransformer(),
    PeriodicityTransformer(period=24),
    DateTimeVectorizer(),
    WeightedNgramVectorizer(),
    UniversalSentenceEncoder(),
    GeoVectorizer(resolution=1),
]


# Serializing component state should be preserved when re-loaded.
@pytest.mark.integration
@pytest.mark.parametrize(
    "component",
    components,
    ids=[component.__class__.__name__ for component in components],
)
def test__joblib_savability(component, tmp_path):
    filename = tmp_path / f"{component.__class__.__name__}.pkl"

    with open(filename, "wb") as file:
        joblib.dump(component, file)

    with open(filename, "rb") as file:
        comp_out = joblib.load(file)

    assert type(comp_out) is type(component)
    assert comp_out.get_params(deep=False) == component.get_params(deep=False)
