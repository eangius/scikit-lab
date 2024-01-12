#!usr/bin/env python


# External libraries
import random
import string
import scipy
import numpy as np
from shapely.geometry.point import Point
from typing import Callable, Optional
from datetime import timedelta, datetime


class RandomData:
    """
    Helper utility to simulate various types of data values & data volumes.
    Intended only for testing, in particular stress testing.
    """

    @classmethod
    def point(cls, lat: float = None, lng: float = None):
        """generates a specific or random lat/lng point"""
        return Point(
            lat or random.randrange(-90, 90),
            lng or random.randrange(-180, 180),
        )

    @classmethod
    def dense_mtx(
        cls,
        row: int = None,
        col: int = None,
        n_min: Optional[int] = 0,
        n_max: Optional[int] = 5000,
    ):
        """generates a random (row x col) dense matrix"""
        n_dim = random.randint(*cls._calibrate_range(n_min, n_max))
        return np.random.rand(row or n_dim, col or n_dim)

    @classmethod
    def sparse_mtx(
        cls,
        row: int = None,
        col: int = None,
        n_min: Optional[int] = 0,
        n_max: Optional[int] = 5000,
    ):
        """generates a random (row x col) column sparse matrix"""
        n_dim = random.randint(*cls._calibrate_range(n_min, n_max))
        return scipy.sparse.rand(
            row or n_dim,
            col or n_dim,
            format="csr",
        )

    @classmethod
    def date(
        cls,
        start: datetime = None,
        n_min: Optional[int] = 1000,
        n_max: Optional[int] = 1000,
    ):
        """generates a random date around another"""
        start = start or datetime.now()
        day1 = start - timedelta(days=n_min)
        day2 = start + timedelta(days=n_max)
        delta = day2 - day1
        return day1 + timedelta(
            seconds=random.randrange((delta.days * 24 * 60 * 60) + delta.seconds)
        )

    @classmethod
    def string(
        cls,
        n_min: Optional[int] = 1,
        n_max: Optional[int] = 32,
        alphabet: str = string.ascii_letters,
    ):
        """generates a random sequence of characters of a random length"""
        size = random.randint(*cls._calibrate_range(n_min, n_max))
        return "".join(random.choices(alphabet, k=size))

    @classmethod
    def list(
        cls,
        fn: Callable,
        n_min: Optional[int] = 0,
        n_max: Optional[int] = 5000,
        **kwargs,
    ) -> list:
        """generates a random list data as per the function"""
        return [
            fn(**kwargs)
            for _ in range(random.randint(*cls._calibrate_range(n_min, n_max)))
        ]

    # convenience interface, if either endpoint is missing assume its identical to the other
    # & thus not a range but a scalar.
    @staticmethod
    def _calibrate_range(n_min: Optional[int], n_max: Optional[int]) -> tuple:
        if n_min is None and n_max is None:
            raise ValueError(
                "At least one of n_min or n_max parameters must be defined."
            )
        if n_min is None:
            n_min = n_max
        if n_max is None:
            n_max = n_min
        return n_min, n_max
