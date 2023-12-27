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
        if index_scheme != 'h3':
            # TODO: implement geohash, s2, ..
            raise NotImplementedError(
                f"Unrecognized indexing schem {index_scheme}"
            )

        self.resolution = resolution
        self.items = items or {'cells'}
        self.offset = offset
        self.index_scheme = index_scheme
        super().__init__(**kwargs)
        return

    @overrides
    def transform(self, X, y=None):
        X = self._convert(X)
        return super().transform(X, y)

    @overrides
    def fit_transform(self, X, y=None):
        X = self._convert(X)
        return super().fit_transform(X, y)

    @overrides
    def inverse_transform(self, X):
        return np.array([
            h3.h3_to_geo(y) if y else self.out_of_vocab  # approx to hash centroid
            for y in super().inverse_transform(X)
        ])

    # accumulates item types into the same vector
    def _convert(self, X):
        X = X.values.ravel() if isinstance(X, pd.DataFrame) else X
        items = []
        if 'cells' in self.items:
            items.extend(list(map(self._cells, X)))
        if 'neighbors' in self.items:
            items.extend(list(map(self._neighbors, X)))
        if 'parents' in self.items:
            items.extend(list(map(self._parents, X)))
        if 'children' in self.items:
            items.extend(list(map(self._children, X)))
        return np.array(items).reshape(-1, 1)  # for ItemCountVectorizer

    def _cells(self, geom: Point):
        return h3.geo_to_h3(geom.y, geom.x, self.resolution)  # lon/lat order

    def _neighbors(self, geom: Point):
        return h3.hex_ring(self._cells(geom), self.setps)

    def _parents(self, geom: Point):
        return h3.h3_to_parent(self._cells(geom), self.resolution - self.offset)

    def _children(self, geom: Point):
        return h3.h3_to_children(self._cells(geom), self.resolution + self.offset)
