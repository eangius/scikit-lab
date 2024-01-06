#!usr/bin/env python

# Internal libraries
from tests.common.pytest_parametrized import pytest_mark_polymorphic
from tests.common.random_data import RandomData
from scikitlab.vectorizers.spatial import GeoVectorizer

# External libraries
from sklearn.base import TransformerMixin
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
import pytest
import random


# Ensure component is a transformer.
@pytest.mark.unit
def test__is_transformer(component):
    assert isinstance(component, TransformerMixin)


# Invalid geohash resolutions should raise an exception
@pytest.mark.unit
def test__GeoVectorizer_error00():
    with pytest.raises(ValueError):
        GeoVectorizer(resolution=-1)


# Invalid geohashing scheme should raise an exception
@pytest.mark.unit
def test__GeoVectorizer_error01():
    with pytest.raises(NotImplementedError):
        GeoVectorizer(resolution=1, index_scheme="unknown")


# Unrecognized items should raise an exception
@pytest.mark.unit
def test__GeoVectorizer_error02():
    with pytest.raises(ValueError):
        GeoVectorizer(resolution=1, items={"unknown"})


# Vectorizing a location cell should yield a sparse matrix with 1 geohash.
@pytest.mark.unit
@pytest_mark_polymorphic
def test__GeoVectorizer_transform01(input_container):
    X = input_container([RandomData.point()])
    component = GeoVectorizer(resolution=5, items={"cells"})
    assert_geohashes(
        vtrs=component.fit_transform(X),
        geos=component.get_feature_names_out(),
        n_samples=X.shape[0],
        n_geohashes=1,
    )


# Vectorizing a location neighbours should yield a sparse matrix with 6 geohashes.
@pytest.mark.unit
@pytest_mark_polymorphic
def test__GeoVectorizer_transform02(input_container):
    X = input_container([RandomData.point()])
    component = GeoVectorizer(resolution=5, items={"neighbors"})
    assert_geohashes(
        vtrs=component.fit_transform(X),
        geos=component.get_feature_names_out(),
        n_samples=X.shape[0],
        n_geohashes=6,
    )


# Vectorizing a location parent should yield a sparse matrix with 1 geohash.
@pytest.mark.unit
@pytest_mark_polymorphic
def test__GeoVectorizer_transform03(input_container):
    X = input_container([RandomData.point()])
    component = GeoVectorizer(resolution=5, items={"parents"})
    assert_geohashes(
        vtrs=component.fit_transform(X),
        geos=component.get_feature_names_out(),
        n_samples=X.shape[0],
        n_geohashes=1,
    )


# Vectorizing a location child should yield a sparse matrix with 1 geohash.
@pytest.mark.unit
@pytest_mark_polymorphic
def test__GeoVectorizer_transform04(input_container):
    X = input_container([RandomData.point()])
    component = GeoVectorizer(resolution=5, items={"children"})
    assert_geohashes(
        vtrs=component.fit_transform(X),
        geos=component.get_feature_names_out(),
        n_samples=X.shape[0],
        n_geohashes=7,
    )


# Vectorizing a location combining items should accumulate multiple geohashes
@pytest.mark.unit
@pytest_mark_polymorphic
def test__GeoVectorizer_transform05(input_container):
    X = input_container([RandomData.point()])
    component = GeoVectorizer(
        resolution=5, items={"cells", "neighbors", "parents", "children"}
    )
    assert_geohashes(
        vtrs=component.fit_transform(X),
        geos=component.get_feature_names_out(),
        n_samples=X.shape[0],
        n_geohashes=15,
    )


# Setting a limit on geohashes should return that many geohashes
@pytest.mark.unit
@pytest_mark_polymorphic
def test__GeoVectorizer_transform06(input_container):
    X = input_container([RandomData.point()])
    component = GeoVectorizer(
        resolution=5,
        items={"cells", "neighbors", "parents", "children"},
        max_items=3,
    )
    assert_geohashes(
        vtrs=component.fit_transform(X),
        geos=component.get_feature_names_out(),
        n_samples=X.shape[0],
        n_geohashes=3,
    )


# Transforming with random component configuration with random volume of random points
# should work.
@pytest.mark.stress
@pytest_mark_polymorphic
def test__GeoVectorizer_transform07(input_container, component):
    X = input_container(RandomData.list(RandomData.point))
    assert_geohashes(
        vtrs=component.fit_transform(X),
        geos=component.get_feature_names_out(),
        n_samples=X.shape[0],
        n_geohashes=None,
    )


# Inverse transforming from vectors should return near original input.
@pytest.mark.integration
@pytest_mark_polymorphic
def test__GeoVectorizer_inverse01(input_container):
    X = input_container([RandomData.point(lat=43.651070, lng=-79.347015)])
    tolerance = 1e-5  # due to geohash centroid & cartesian dist with deg approx
    component = GeoVectorizer(resolution=15)
    vtrs = component.fit_transform(X)
    pts = component.inverse_transform(vtrs)
    assert isinstance(pts, np.ndarray)
    assert all(
        pt1.distance(pt2) < tolerance
        for pt1, pt2 in zip(X[0].tolist() if isinstance(X, pd.DataFrame) else X, pts)
    )


def assert_geohashes(vtrs, geos, n_samples, n_geohashes=None):
    assert isinstance(vtrs, csr_matrix)  # sparse matrix
    assert isinstance(geos, np.ndarray)
    if n_geohashes:  # expected n-geohashes if known
        assert vtrs.shape == (n_samples, n_geohashes)
        assert geos.shape == (n_geohashes,)


@pytest.fixture
def component():
    return GeoVectorizer(
        index_scheme="h3",
        offset=random.randint(1, 2),  # else too many hashes
        resolution=random.randint(7, 12),  # else too many hashes
        items=set(
            random.sample(
                population=["cells", "neighbors", "parents", "children"],
                k=random.randint(1, 4),
            )
        ),
    )
