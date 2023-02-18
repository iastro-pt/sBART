import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from astropy.coordinates import EarthLocation
from astropy.io import fits
from loguru import logger
from scipy.constants import convert_temperature

from SBART.Base_Models.Frame import Frame
from SBART.utils import custom_exceptions
from SBART.utils.UserConfigs import BooleanValue, DefaultValues, UserParam
from SBART.utils.status_codes import (
    ERROR_THRESHOLD,
    NAN_DATA,
    SATURATION,
    KW_WARNING
)
from SBART.utils.status_codes import FATAL_KW
from SBART.utils.types import UI_PATH
from SBART.utils.units import kilometer_second, meter_second
from SBART.utils.RV_utilities import airtovac
from SBART.utils.status_codes import (
    ERROR_THRESHOLD,
    MISSING_DATA,
    NAN_DATA,
    SATURATION,
    SUCCESS,
    KW_WARNING
)

from .ESO_PIPELINE import ESO_PIPELINE


class NIRPS(ESO_PIPELINE):
    """
    Interface to handle HARPSN data; S1D **not** supported!

    This class also defines 2 sub-Instruments:

    * HARPSN - Until the ends of time (hopefully)


    The steps to load the S2D data are described in the HARPS `DRS manual <https://www.eso.org/sci/facilities/lasilla/instruments/harps/doc/DRS.pdf>`_. The summary is:

        - Construct the wavelength solution & correct from BERV
        - Load instrumental drift
        - Construct flux noises:

            - Table 10 of `the user manual <https://www.eso.org/sci/facilities/lasilla/instruments/harps/doc/manual/HARPS-UserManual2.4.pdf>`_ gives max RON of 7.07 for red detector
            - Noise = sqrt(obj + sky + n*dark*expTime + nBinY*ron^2)

    **User parameters:**

     Currently there are no HARPSN-specific parameters

    *Note:* Check the **User parameters** of the parent classes for further customization options of SBART

    """

    sub_instruments = {
        # "HARPSpre": datetime.datetime.strptime("2015-05-29", r"%Y-%m-%d"),
        "NIRPS": datetime.datetime.max,
    }

    _name = "NIRPS"
    _default_params = ESO_PIPELINE._default_params

    def __init__(
            self,
            file_path,
            user_configs: Optional[Dict[str, Any]] = None,
            reject_subInstruments=None,
            frameID=None,
            quiet_user_params: bool = True

    ):
        """

        Parameters
        ----------
        file_path
            Path to the S2D (or S1D) file.
        user_configs
            Dictionary whose keys are the configurable options of ESPRESSO (check above)
        reject_subInstruments
            Iterable of subInstruments to fully reject
        frameID
            ID for this observation. Only used for organization purposes by :class:`~SBART.data_objects.DataClass`
        """
        raise NotImplementedError
        logger.info("Creating frame from: {}".format(file_path))

        # TODO: find about this
        coverage = [390, 700]

        super().__init__(
            inst_name="NIRPS",
            array_size={"S2D": (69, 4096)}, # TODO: find about this
            file_path=file_path,
            KW_identifier="TNG",
            frameID=frameID,
            override_KW_map=None,
            user_configs=user_configs,
            reject_subInstruments=reject_subInstruments,
            quiet_user_params=quiet_user_params
        )

        self.instrument_properties["wavelength_coverage"] = coverage

        # TODO: setup info for telfit. Do we need to do anything for the tellurics or is everything handled by the DRS?

        #
        # self.instrument_properties["resolution"] = 115_000
        # self.instrument_properties["EarthLocation"] = EarthLocation.of_site(
        #     "Roque de los Muchachos"
        # )
        #
        # # https://tngweb.tng.iac.es/weather/current
        # self.instrument_properties["site_pressure"] = 770



