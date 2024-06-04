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
from SBART.utils.status_codes import ERROR_THRESHOLD, NAN_DATA, SATURATION, KW_WARNING
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
    KW_WARNING,
)

from .ESO_PIPELINE import ESO_PIPELINE


def val_cheby(coeffs, xvector, domain):
    """
    Using the output of fit_cheby calculate the fit to x  (i.e. y(x))
    where y(x) = T0(x) + T1(x) + ... Tn(x)

    :param coeffs: output from fit_cheby
    :param xvector: x value for the y values with fit
    :param domain: domain to be transformed to -1 -- 1. This is important to
    keep the components orthogonal. For SPIRou orders, the default is 0--4088.
    You *must* use the same domain when getting values with fit_cheby
    :return: corresponding y values to the x inputs
    """
    # transform to a -1 to 1 domain
    domain_cheby = 2 * (xvector - domain[0]) / (domain[1] - domain[0]) - 1
    # fit values using the domain and coefficients
    yvector = np.polynomial.chebyshev.chebval(domain_cheby, coeffs)
    # return y vector
    return yvector


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
        quiet_user_params: bool = True,
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
        logger.info("Creating frame from: {}".format(file_path))

        coverage = [940, 1850]

        super().__init__(
            inst_name="NIRPS",
            array_size={"S2D": (71, 4084)},  # TODO: find about this
            file_path=file_path,
            KW_identifier="ESO",
            frameID=frameID,
            override_KW_map=None,
            user_configs=user_configs,
            reject_subInstruments=reject_subInstruments,
            quiet_user_params=quiet_user_params,
        )

        self.instrument_properties["wavelength_coverage"] = coverage

        # TODO: setup info for telfit. Do we need to do anything for the tellurics or is everything handled by the DRS?

        self.instrument_properties["resolution"] = 100000
        self.instrument_properties["EarthLocation"] = EarthLocation.of_site("La Silla Observatory")

        # ? same as for Paranal?
        # https://www.eso.org/sci/facilities/paranal/astroclimate/site.html
        self.instrument_properties["site_pressure"] = 750

    def is_APERO_data(self) -> bool:
        """
        Check if the frame was reduced with the APERO pipeline
        Returns
        -------

        """
        return "dsff_tcorr" in self.file_path.name

    def load_ESO_DRS_S2D_data(self):
        if self.is_APERO_data():
            self.load_Apero_data()
        else:
            super().load_ESO_DRS_S2D_data("EXT_E2DS")

    def load_Apero_data(self, spectra, header):
        # size of the image
        image = fits.getdata(self.file_path)
        hdr = fits.getheader(self.file_path)

        nbypix, nbxpix = image.shape
        # get the keys with the wavelength polynomials
        wave_hdr = hdr["WAVE0*"]
        # concatenate into a numpy array
        wave_poly = np.array([wave_hdr[i] for i in range(len(wave_hdr))])
        # get the number of orders
        nord = hdr["WAVEORDN"]
        # get the per-order wavelength solution
        wave_poly = wave_poly.reshape(nord, len(wave_poly) // nord)
        # project polynomial coefficiels
        wavesol = np.zeros_like(image)
        # xpixel grid
        xpix = np.arange(nbxpix)
        # loop around orders
        for _order_num in range(nord):
            # calculate wave solution for this order
            owave = val_cheby(wave_poly[_order_num], xpix, domain=[0, nbxpix])
            # push into wave map
            wavesol[_order_num] = owave

        self.wavelengths = wavesol
        self.spectra = image
        self.uncertainties = np.sqrt(image)

        self.build_mask(bypass_QualCheck=False)

    def _load_ESO_DRS_KWs(self, header):
        if self.is_APERO_data():
            self.observation_info["MAX_BERV"] = header[f"BERVMAX"] * kilometer_second
            self.observation_info["BERV"] = header[f"BERV"] * kilometer_second

            self.observation_info["DRS_RV"] = (
                header[f"HIERARCH {self.KW_identifier} QC CCF RV"] * kilometer_second
            )
            self.observation_info["DRS_RV_ERR"] = (
                header[f"HIERARCH {self.KW_identifier} QC CCF RV ERROR"] * kilometer_second
            )

            for order in range(self.instrument_properties["array_sizes"]["S2D"][0]):
                self.observation_info["orderwise_SNRs"].append(
                    header[f"HIERARCH {self.KW_identifier} QC ORDER{order + 1} SNR"]
                )

        else:
            super()._load_ESO_DRS_KWs(header)

    def load_telemetry_info(self, header):
        ambi_KWs = {
            "relative_humidity": "AMBI RHUM",
            "ambient_temperature": "AMBI TEMP",
        }

        for name, endKW in ambi_KWs.items():
            self.observation_info[name] = float(header[f"HIERARCH ESO TEL {endKW}"])
            if "temperature" in name:  # store temperature in KELVIN for TELFIT
                self.observation_info[name] = convert_temperature(
                    self.observation_info[name], old_scale="Celsius", new_scale="Kelvin"
                )

        self.observation_info["airmass"] = header["HIERARCH ESO TEL AIRM START"]
