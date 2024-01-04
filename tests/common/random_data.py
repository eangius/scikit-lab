#!usr/bin/env python


# External libraries
import random
import scipy
import numpy as np
from shapely.geometry.point import Point
from typing import Callable
from datetime import timedelta, datetime


class RandomData:
    """
    Helper utility to simulate various types of data values & data volumes.
    Intended only for testing, in particular stress testing.
    """

    @classmethod
    def point(cls, lat: float = None, lng: float = None):
        """ generates a specific or random lat/lng point """
        return Point(
            lat or random.randrange(-90, 90),
            lng or random.randrange(-180, 180),
        )

    @classmethod
    def dense_mtx(cls, row: int = None, col: int = None, n_min: int = 0, n_max: int = 5000):
        """ generates a random (row x col) dense matrix """
        return np.random.rand(
            row or random.randint(n_min, n_max),
            col or random.randint(n_min, n_max)
        )

    @classmethod
    def sparse_mtx(cls, row: int = None, col: int = None, n_min: int = 0, n_max: int = 5000):
        """ generates a random (row x col) column sparse matrix """
        return scipy.sparse.rand(
            row or random.randint(n_min, n_max),
            col or random.randint(n_min, n_max),
            format='csr'
        )

    @classmethod
    def date(cls, start: datetime = None, n_min: int = 1000, n_max: int = 1000):
        """ generates a random date around another"""
        start = start or datetime.now()
        day1 = start - timedelta(days=n_min)
        day2 = start + timedelta(days=n_max)
        delta = day2 - day1
        return day1 + timedelta(
            seconds=random.randrange(
                (delta.days * 24 * 60 * 60) + delta.seconds
            )
        )

    @classmethod
    def list(cls, fn: Callable, n_min: int = 0, n_max: int = 5000, **kwargs) -> list:
        """ generates a random list data as per the function """
        return [
            fn(**kwargs)
            for _ in range(random.randint(n_min, n_max))
        ]
