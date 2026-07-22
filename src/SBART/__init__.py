"""SBART library."""

version = "1.1.2"

__version__ = version.replace(".", "-")
__version_info__ = (int(i) for i in __version__.split("-"))

# To avoid having supressed warnings during the SBART application
# https://docs.python.org/3/library/warnings.html
import warnings

warnings.simplefilter("always", UserWarning)

import os

from SBART.utils.create_logger import sbart_logger, setup_SBART_logger

if os.environ.get("NO_GRAPHICAL_BACKEND", "False") == "True":
    import matplotlib.pyplot as plt

    sbart_logger.warning("sbart disabling graphical backend for matplotlib")
    plt.switch_backend("agg")


# Is this a good idea? Guess not...
import pathlib

SBART_LOC = pathlib.Path(__file__).parent

# TODO: check type hints when passing derived classes!
