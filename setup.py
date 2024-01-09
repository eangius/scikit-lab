#!usr/bin/env python

# Internal libraries
import scikitlab as proj


# External libraries
import sys
import setuptools
from distutils.core import setup


# Common & relative values.
PY_VERSION = sys.version_info
PROJ_LICENSE = "MIT"


setup(
    # Metadata
    name=proj.__name__,
    version=proj.__version__,
    description="Custom scikit components",
    url="https://github.com/eangius/scikit-lab/",
    author="Elian Angius",
    license=PROJ_LICENSE,
    keywords=["machine-learning", "scikit", "library"],
    packages=setuptools.find_packages(
        where=".",
        include=[f"{proj.__name__}*"],
        exclude=["tests"],
    ),
    # Dependencies to auto install.
    python_requires=f">={PY_VERSION.major}.{PY_VERSION.minor}",
    install_requires=[],
    platforms=["any"],
)
