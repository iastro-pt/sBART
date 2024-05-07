from loguru import logger

try:
    from .inversion.inverter import diag_inverter as tridiagonal_inverter
    from .partial_derivative.partial_derivative import partial_derivative
    from .second_derivative.second_derivative import second_derivative

    CYTHON_UNAVAILABLE = False
except ImportError:
    logger.critical(
        "Cython interface is not found, please make sure that the installation went smoothly"
    )
    CYTHON_UNAVAILABLE = True
