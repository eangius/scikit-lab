#!usr/bin/env python


__version__ = "0.0.0"


# External libraries
import os
import logging
import numpy as np

# Absolute location of various directories relative to project installation.
PROJ_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
RUNTIME_DIR = os.path.join(PROJ_DIR, "runtime")
RESOURCE_DIR = os.path.join(PROJ_DIR, "resources")


# Disable tensorflow library info messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.getLogger("tensorflow").disabled = True
logging.getLogger("tensorflow_hub").disabled = True
logging.getLogger("tensorflow_model_optimization").disabled = True


# Configure numpy
np.set_string_function(lambda x: repr(x), repr=False)  # copy-paste friendly
np.set_printoptions(linewidth=np.inf)  # don't wrap
