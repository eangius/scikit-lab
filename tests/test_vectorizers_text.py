#!usr/bin/env python

# Internal libraries
from tests.common.pytest_parametrized import pytest_mark_polymorphic
from scikitlab import RESOURCE_DIR
from scikitlab.vectorizers.text import WeightedNgramVectorizer, UniversalSentenceEncoder

# External libraries
import pytest
import numpy as np
from scipy.sparse import csr_matrix


@pytest.mark.unit
def test__WeightedNgramVectorizer_01(corpus):
    input_container = np.array  # scikit CountVectorizer does not support pd.DataFrames
    X = input_container(corpus)
    component = WeightedNgramVectorizer(ngram_range=(1, 3))
    results = component.fit_transform(X)
    ngrams = component.get_feature_names_out()
    assert_sparse(X, results, ngrams)
    assert_ngrams(component, ngrams)


@pytest.mark.unit
@pytest_mark_polymorphic
def test__UniversalSentenceEncoder_01(input_container, corpus):
    X = input_container(corpus)
    component = UniversalSentenceEncoder(
        resource_dir=f"{RESOURCE_DIR}/use_5",
    )
    dim = component.dimensionality
    results = component.fit_transform(X)
    assert isinstance(results, np.ndarray)  # numpy array
    assert np.issubdtype(results.dtype, np.floating)  # floats platform independent
    assert isinstance(dim, int) and dim > 0  #
    assert results.shape == (X.shape[0], dim)  # fixed sized vectors


def assert_sparse(X, results, ngrams):
    assert isinstance(results, csr_matrix)  # vectors are sparse
    assert results.shape == (X.shape[0], ngrams.shape[0])  # one vector per doc
    assert (results.todense() >= 0).all()  # all values are non-negative


def assert_ngrams(component, ngrams):
    assert len(ngrams.tolist()) == len(set(ngrams))  # no duplicates
    assert all(  # all ngrams within config range
        len(gram.split(" "))
        in range(component.ngram_range[0], component.ngram_range[1] + 1)
        for gram in ngrams
    )


@pytest.fixture
def corpus() -> list:
    return [
        "This is the first document",
        "Others can exist",
        "& other documents can also exist",
    ]
