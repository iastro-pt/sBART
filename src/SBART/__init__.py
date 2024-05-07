"""
SBART library.
"""

version = "0.5.1"

__version__ = version.replace(".", "-")
__version_info__ = (int(i) for i in __version__.split("-"))

# To avoid having supressed warnings during the SBART application
# https://docs.python.org/3/library/warnings.html
import warnings

warnings.simplefilter("always", UserWarning)

from loguru import logger

logger.disable(__name__)

import os

if os.environ.get("NO_GRAPHICAL_BACKEND", "False") == "True":
    import matplotlib.pyplot as plt

    logger.warning("sbart disabling graphical backend for matplotlib")
    plt.switch_backend("agg")


# Is this a good idea? Guess not...
import pathlib

SBART_LOC = pathlib.Path(__file__).parent

# TODO: check type hints when passing derived classes!

import SBART.Instruments as Instruments
import SBART.data_objects
import SBART.utils
import SBART.template_creation
import SBART.rv_calculation
