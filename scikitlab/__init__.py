#!usr/bin/env python


__version__ = '0.0.0'


# External libraries
import os


# Absolute location of various directories relative to project installation.
PROJ_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
RUNTIME_DIR = os.path.join(PROJ_DIR, 'runtime')
