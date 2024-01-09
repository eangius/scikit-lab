#!usr/bin/env python

# Internal libraries
import scikitlab as proj


# External libraries
import setuptools
import sys
from distutils.core import setup


# Common & relative values.
PY_VERSION = sys.version_info
PROJ_LICENSE = "MIT"


def parse_requirements(filename: str) -> list:
    """parses a requirements file & get the dependency & version specs ignoring comments"""
    with open(filename, "r") as file:
        return [
            dependency
            for dependency in [line.split("#")[0].strip() for line in file.readlines()]
            if dependency
        ]


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
    # Deployable Dependencies
    python_requires=f">={PY_VERSION.major}.{PY_VERSION.minor}",
    install_requires=parse_requirements(
        f"{proj.PROJ_DIR}/requirements/requirements_programming.txt"
    )
    + parse_requirements(f"{proj.PROJ_DIR}/requirements/requirements_ecosystem.txt")
    + parse_requirements(f"{proj.PROJ_DIR}/requirements/requirements_misc.txt"),
    platforms=["any"],
)
