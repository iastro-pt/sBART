from typing import Iterable

from loguru import logger
from SBART.utils import custom_exceptions


def ensure_value_in_window(tentative_value: float, desired_interval: Iterable[float]):
    """Ensure that a given radial velocity is inside the effective RV limits
    (defined by the window centered in the previous RV estimate)

    Parameters
    ----------
    tentative_RV : float
        Current RV
    effective_RV_limits : Iterable[float]
        Tuple with the RV limits

    Raises
    ------
    Exception
        If the RV is outside the window, raises error
    """
    if not desired_interval[0] <= tentative_value <= desired_interval[1]:
        msg = "Using value outside the effective limit: {} / {}".format(
            tentative_value, desired_interval
        )
        logger.critical(msg)
        raise custom_exceptions.InvalidConfiguration(msg)
