"""
Utilies for SBART application
"""

from loguru import logger
from .concurrent_tools.open_buffers import open_buffer

try:
    from .cython_codes.matmul import second_term

    CYTHON_UNAVAILABLE = False
except ImportError:
    logger.critical(
        "Cython interface is not found, please make sure that the installation went smoothly"
    )
    CYTHON_UNAVAILABLE = True

from .find_nearby_wavelengths import find_close_lambdas
from .parameter_validators import validator
from .paths_tools.build_filename import build_filename
from .roll_arrays import roll_array
from .RV_utilities import RVerror, build_blocks, find_wavelength_limits
from .spectral_conditions import FNAME_condition, KEYWORD_condition
from .tapas_downloader.web_scrapper import get_TAPAS_data
from .telluric_utilities import calculate_telluric_mask
from .units import kilometer_second, meter_second
