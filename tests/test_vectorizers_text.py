#!usr/bin/env python

# Internal libraries
from scikitlab.vectorizers.text import *

# External libraries
import pytest
from scipy.sparse import csr_matrix


@pytest.mark.unit
def test__WeightedNgramVectorizer_01():
    input_container = np.array  # scikit CountVectorizer does not support pd.DataFrames
    X = input_container([
        'This is the first document',
        'Others can exist',
        '& other documents can also exist',
    ])
    component = WeightedNgramVectorizer(ngram_range=(1, 3))
    results = component.fit_transform(X)
    ngrams = component.get_feature_names_out()
    assert_matrix(X, results, ngrams)
    assert_ngrams(component, ngrams)


def assert_matrix(X, results, ngrams):
    assert isinstance(results, csr_matrix)                   # vectors are sparse
    assert results.shape == (X.shape[0], ngrams.shape[0])    # one vector per doc
    assert (results.todense() >= 0).all()                    # all values are non-negative


def assert_ngrams(component, ngrams):
    assert len(ngrams.tolist()) == len(set(ngrams))           # no duplicates
    assert all(                                               # all ngrams within config range
        len(gram.split(' ')) in range(component.ngram_range[0], component.ngram_range[1] + 1)
        for gram in ngrams
    )
