#!usr/bin/env python


# External libraries
import random
import scipy
import numpy as np
from shapely.geometry.point import Point


class RandomData:
    """
    Helper utility to simulate various types of data values & data volumes.
    Intended only for testing, in particular stress testing.
    """

    @classmethod
    def geo_point(cls, lat: float = None, lng: float = None):
        """ generates a specific or random lat/lng point """
        return Point(
            lat or random.randrange(-90, 90),
            lng or random.randrange(-180, 180),
        )

    @classmethod
    def geo_points(cls, n_min: int = 0, n_max: int = 10000) -> list:
        """ generates a random list of lat/lng points """
        return [
            cls.geo_point()
            for _ in range(random.randint(n_min, n_max))
        ]

    @classmethod
    def dense_mtx(cls, row: int = None, col: int = None, n_min: int = 0, n_max: int = 10000):
        """ generates a random (row x col) dense matrix """
        return np.random.rand(
            row or random.randint(n_min, n_max),
            col or random.randint(n_min, n_max)
        )

    @classmethod
    def sparse_mtx(cls, row: int = None, col: int = None, n_min: int = 0, n_max: int = 10000):
        """ generates a random (row x col) column sparse matrix """
        return scipy.sparse.rand(
            row or random.randint(n_min, n_max),
            col or random.randint(n_min, n_max),
            format='csr'
        )
