import getpass
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from loguru import logger
from scipy.ndimage import median_filter

from SBART.utils import custom_exceptions, get_TAPAS_data
from SBART.utils.UserConfigs import (
    BooleanValue,
    DefaultValues,
    Positive_Value_Constraint,
    PathValue,
    UserParam,
    PathValue,
)
from SBART.utils.shift_spectra import remove_BERV_correction
from .Telluric_Template import TelluricTemplate


class TapasTelluric(TelluricTemplate):
    """
    Create transmittance spectrum from TAPAS web-interface

    This class also provides an interface to automatically request and download data from the TAPAS user interface.

    **User parameters:**

    ================================ ================ ================ ================ ================
    Parameter name                      Mandatory      Default Value    Valid Values    Comment
    ================================ ================ ================ ================ ================
    user_info                            False          ["", ""]            -------       [1]
    download_tapas                      False             True           boolean         [2]
    download_path                       True              ------          string         [3]
    timeout                             False                10         int >= 0         [4]
    request_interval                      False              60         int >= 0         [5]
    ================================ ================ ================ ================ ================

    - [1]: List with two entries: tapas username and password. If the password is not provided, the class will ask the user for it
    - [2]: If False, this class won't request data from TAPAS
    - [3]: Path in which the Tapas is (or will be) stored
    - [4]: Maximum number of minutes during which we are waiting for a response from TAPAS
    - [5]: Time (in seconds) between attempts to contact TAPAS webserver.


    .. note::
        Also check the **User parameters** of the parent classes for further customization options of SBART

    .. note::
        Don't use a very short interval in the "request_interval" parameter to avoid spamming TAPAS server!
    """

    _default_params = TelluricTemplate._default_params + DefaultValues(
        user_info=UserParam(["", ""], quiet=True),
        download_tapas=UserParam(True, constraint=BooleanValue),
        download_path=UserParam(None, constraint=PathValue, mandatory=True),
        timeout=UserParam(10, Positive_Value_Constraint),
        request_interval=UserParam(60, Positive_Value_Constraint),
    )
    method_name = "Tapas"

    def __init__(
        self,
        subInst: str,
        user_configs: Union[None, dict] = None,
        extension_mode: str = "lines",
        application_mode: str = "removal",
        loaded: bool = False,
    ):
        super().__init__(
            subInst=subInst,
            extension_mode=extension_mode,
            user_configs=user_configs,
            loaded=loaded,
            application_mode=application_mode,
        )

        if not loaded:
            logger.info(
                "Creating telluric template from TAPAS transmittance spectra for {}",
                self._associated_subInst,
            )

    def fit_telluric_model_to_frame(self, frame):
        super().fit_telluric_model_to_frame(frame)
        raise NotImplementedError(
            "Tapas template does not implement a correction model for tellurics"
        )

    def _prepare_TAPAS_download(self, dataClass):
        user, password = self._internal_configs["user_info"]
        tapas_dwnl_path = self._internal_configs["download_path"]

        if isinstance(tapas_dwnl_path, Path):
            tapas_dwnl_path = tapas_dwnl_path.as_posix()

        temp_path = Path(tapas_dwnl_path)
        if not temp_path.exists():
            logger.warning("Tapas download path does not exist. Creating new folder from scratch")
            temp_path.mkdir(parents=True, exist_ok=True)

        if os.path.isdir(tapas_dwnl_path):
            logger.info("Using individual Transmittance spectra for each subInstrument")
            tapas_dwnl_path = os.path.join(tapas_dwnl_path, f"{self._associated_subInst}.ipac")
        else:
            logger.info(
                "Using common Transmittance spectra for each subInstrument: {}", tapas_dwnl_path
            )

            if (
                not os.path.isfile(tapas_dwnl_path)
                and not self._internal_configs["force_download"]
                and not self._internal_configs["download_tapas"]
            ):
                msg = "Common ipac template does not exist"
                logger.critical(msg)
                raise custom_exceptions.InvalidConfiguration(msg)

            if not tapas_dwnl_path.endswith("ipac"):
                msg = "Invalid file format for the IPAC template"
                logger.critical(msg)
                raise custom_exceptions.InvalidConfiguration(msg)

        download = True

        if (
            os.path.isfile(tapas_dwnl_path) and not self._internal_configs["force_download"]
        ):  # direct path to the file!
            logger.info("TAPAS file already exists. Skipping download")
            download = False

        if download or self._internal_configs["force_download"]:
            if (
                not self._internal_configs["download_tapas"]
                and not self._internal_configs["force_download"]
            ):
                # Check if the download flag is disabled and if we don't want to force it to go through
                raise Exception("TAPAS download is disabled")
            if len(user) == 0:
                user = input("Tapas username: ")
            if len(password) == 0:
                password = getpass.getpass(prompt="Insert Tapas password:")

            ref_info = {}
            ref_info["mjd_time"] = Time(
                dataClass.get_KW_from_frameID("MJD", self._reference_frameID),
                format="mjd",
            )

            ra, dec = (
                dataClass.get_KW_from_frameID("RA", self._reference_frameID),
                dataClass.get_KW_from_frameID("DEC", self._reference_frameID),
            )
            c = SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame="icrs")

            string_RA, string_DEC = c.to_string("hmsdms", sep=":").split()

            ref_info["RA"] = string_RA.split(".")[0].replace("+", "")
            ref_info["DEC"] = string_DEC.split(".")[0].replace("+", "")

            ref_info["instrument"] = dataClass.get_instrument_information()["name"]
            ref_info["spectralRange"] = dataClass.get_instrument_information()[
                "wavelength_coverage"
            ]

            logger.info("Preparing to get TAPAS data")
            tapas_path = get_TAPAS_data(
                user,
                password,
                ref_info,
                tapas_dwnl_path,
                timeout=self._internal_configs["timeout"],
                request_interval=self._internal_configs["request_interval"],
            )

        self._tapas_file_path = tapas_dwnl_path

    # TODO: fix the input args of this template!
    @custom_exceptions.ensure_invalid_template
    def create_telluric_template(self, dataClass, custom_frameID: Optional[int] = None) -> None:
        """
        Create a telluric template from TAPAS transmission spectra [1], that was created for
        the date in which the reference observation was made.

        It estimates the continuum level and classifies each point that shows a decrease of 10% as a telluric line.

        How does it work:

        - The wavelengths input is the wavelength solution of the reference frame, at a rest frame, i.e. BERV_0 = 0 & RV = 0
        - The TAPAS template was built for this given date, and corrected from a given BERV (BERV_1) at the time it was created.
        In order to account for small offsets between BERV_1 and BERV_0, and to use templates built at differente times, we must shift
        the wavelength solution of the TAPAS template: -> remove the correction from BERV_1, and shift the template by BERV_0
        - In this process, the TAPAS transmittance spectra & the wavelength solution of the template are a good match!


        Parameters
        ----------
        dataClass:
            DataClass object
        custom_frameID :
            If Not None, does not search for the "optimal" frameID to use as a basis
        Returns
        -------
        numpy.ndarray
            Telluric (binary) spectrum, for the wavelengths present in the input array

        Notes
        -----------
        [1] http://cds-espri.ipsl.fr/tapas/project?methodName=home_en
        """
        try:
            super().create_telluric_template(dataClass, custom_frameID=custom_frameID)
        except custom_exceptions.StopComputationError:
            return

        self._prepare_TAPAS_download(dataClass)

        logger.info("Using TAPAS transmittance path from: " f"{self._tapas_file_path}")

        data = np.loadtxt(self._tapas_file_path, skiprows=40)
        self.transmittance_spectra = data[:, 1]
        self.transmittance_wavelengths = data[:, 0]

        # The wavelengths are not sorted -> leads to problems further ahead
        sort_inds = np.argsort(self.transmittance_wavelengths)
        self.transmittance_spectra = self.transmittance_spectra[sort_inds]
        self.transmittance_wavelengths = self.transmittance_wavelengths[sort_inds]

        with open(self._tapas_file_path) as file:
            for line_ind, line in enumerate(file):
                if r"baryvcor" in line:
                    tapas_BERV = float(line.split("=")[1])
                    break

                if line_ind >= 40:
                    msg = "Read 40 lines, did not find BERV"
                    logger.critical(msg)
                    raise Exception(msg)

        logger.info("Tapas template over-riding BERV associated with the reference wavelengths")
        self._associated_BERV = tapas_BERV

        # Tapas spectra comes BERV-corrected
        self.transmittance_wavelengths = remove_BERV_correction(
            self.transmittance_wavelengths, self._associated_BERV
        )

        ###
        # Compute the binary template
        ###
        n_points_filter = 1001
        continuum_level = median_filter(self.transmittance_spectra, n_points_filter)

        # avoid problems in the edges -> calculate median filter with less points near the edges
        continuum_level[0 : n_points_filter + 1] = median_filter(
            self.transmittance_spectra[0 : n_points_filter + 1], 51
        )
        continuum_level[-(n_points_filter + 1) :] = median_filter(
            self.transmittance_spectra[-(n_points_filter + 1) :], 51
        )
        self._continuum_level = continuum_level
        self.wavelengths = self.transmittance_wavelengths * 10

        self._finish_template_creation()
