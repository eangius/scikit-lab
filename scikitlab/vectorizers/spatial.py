#!usr/bin/env python


# Internal libraries
from scikitlab.vectorizers.frequential import ItemCountVectorizer

# External libraries
from shapely.geometry.point import Point
from overrides import overrides
from typing import Set
import h3.api.numpy_int as h3
import numpy as np
import pandas as pd


class GeoVectorizer(ItemCountVectorizer):
    """
    Converts shapely latitude & longitude point coordinates to a geospatial indexing
    scheme. This is useful to quantize areas & model neighboring or hierarchical spatial
    relationships between them. Some relationships are 1:1 but others are 1:many, so
    resulting vectors denote occurrence counts of all train-time known areas. Any
    unrecognized area at inference time is ignored & vectorized as zero. Depending on
    spatial resolution & coverage, vectors can be high-dimensional & are encoded as
    sparse. Users may want to cap dimensionality by size, frequency or perform other
    dimensionality reduction techniques.
    """

    def __init__(
        self,
        resolution: int,           # cell size of this area (range depends on scheme)
        index_scheme: str = 'h3',  # geo indexing scheme
        items: Set[str] = None,    # combo of 'cells', 'neighbors', 'parents' or 'children'.
        offset: int = 1,           # neighbouring or hierarchical cells away from this
        **kwargs                   # see ItemCountVectorizer inputs.
    ):
        # TODO: implement geohash, s2, ..
        self.index_scheme = index_scheme
        if self.index_scheme != 'h3':
            raise NotImplementedError(f"Unrecognized indexing schem {index_scheme}")

        self.resolution = resolution
        if resolution not in range(15 + 1):
            raise ValueError(f"{index_scheme} resolution not in range")

        self.items = items or {'cells'}
        if self.items - {'cells', 'neighbors', 'parents', 'children'} != set():
            raise ValueError("Unrecognized items.")

        self.offset = offset
        if self.offset < 1:
            raise ValueError("Invalid offset")

        super().__init__(**kwargs)
        return

    @overrides
    def transform(self, X, y=None):
        """
        :param X: vector of shapely lat/lng points.
        :param y: unused target variables.
        :return: one-hot-encoded sparse vector of geohashes.
        """
        return super().transform(self._convert(X), y)

    @overrides
    def fit_transform(self, X, y=None):
        return super().fit_transform(self._convert(X), y)

    @overrides
    def inverse_transform(self, X):
        """
        :param X: one-hot-encoded sparse vector of geohashes.
        :return: vector of shapely lat/lng points.
        """
        return np.array([
            Point(h3.h3_to_geo(y)) if y else self.out_of_vocab  # approx to hash centroid
            for y in super().inverse_transform(X)
        ])

    # accumulates item types into the same vector
    def _convert(self, X):
        X = X.values.ravel() if isinstance(X, pd.DataFrame) else X
        docs = []
        for pt in X:
            items = []
            if 'cells' in self.items:
                items.extend(self._cells(pt))
            if 'neighbors' in self.items:
                items.extend(self._neighbors(pt))
            if 'parents' in self.items:
                items.extend(self._parents(pt))
            if 'children' in self.items:
                items.extend(self._children(pt))
            docs.extend(items)
        return np.array(docs).reshape(X.shape[0], -1)

    # returns 1
    def _cells(self, geom: Point) -> list:
        return [h3.geo_to_h3(geom.x, geom.y, self.resolution)]

    # returns 6
    def _neighbors(self, geom: Point) -> list:
        return h3.hex_ring(self._cells(geom)[0], self.offset).tolist()

    # returns 1
    def _parents(self, geom: Point) -> list:
        return [h3.h3_to_parent(self._cells(geom)[0], max(self.resolution - self.offset, 0))]

    # returns many
    def _children(self, geom: Point) -> list:
        return h3.h3_to_children(self._cells(geom)[0], min(self.resolution + self.offset, 15)).tolist()
