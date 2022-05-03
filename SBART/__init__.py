"""
SBART library.
"""

__version_info__ = (0, 1, 4)
__version__ = "-".join(map(str, __version_info__))

# To avoid having supressed warnings during the SBART application
# https://docs.python.org/3/library/warnings.html
import warnings

warnings.simplefilter("always", UserWarning)

from loguru import logger

logger.disable(__name__)

# Is this a good idea? Guess not...
import pathlib

SBART_LOC = pathlib.Path(__file__).parent

# TODO: check type hints when passing derived classes!
