from typing import Iterable

from loguru import logger


def ensure_valid_RV(tentative_RV: float, effective_RV_limits: Iterable[float]):
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
    if not effective_RV_limits[0] <= tentative_RV <= effective_RV_limits[1]:
        msg = "Using RV value outside the effective limit: {} / {}".format(
            tentative_RV, effective_RV_limits
        )
        logger.critical(msg)
        raise Exception(msg)
