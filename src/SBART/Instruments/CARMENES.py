import datetime
from pathlib import Path
from typing import Any, Dict, NoReturn, Optional

import numpy as np
from astropy.coordinates import EarthLocation
from astropy.io import fits
from loguru import logger
from scipy.constants import convert_temperature

from SBART.Base_Models.Frame import Frame
from SBART.Masks import Mask
from SBART.utils import custom_exceptions
from SBART.utils.custom_exceptions import FrameError
from SBART.utils.shift_spectra import apply_BERV_correction
from SBART.utils.status_codes import MISSING_DATA, Flag, MISSING_SHAQ_RVS
from SBART.utils.units import kilometer_second
from SBART.utils.UserConfigs import (
    DefaultValues,
    PathValue,
    UserParam,
    BooleanValue,
    ValueFromDtype,
)


class CARMENES(Frame):
    """
    Interface to handle KOBE-CARMENES data (optical arm only),
    loading the initial RV estimate from CCF files stored in a different directory.

    This class defines 1 sub-Instruments:

    * CARMENES - Until the ends of time (hopefully)


    **User parameters:**

    ================================ ================ ================ ================ ================
    Parameter name                      Mandatory      Default Value    Valid Values    Comment
    ================================ ================ ================ ================ ================
    shaq_output_folder                True                ----          str                 Path where SHAQ's outputs are stored
    ================================ ================ ================ ================ ================

    *Note:* Check the **User parameters** of the parent classes for further customization options of SBART

    """

    _default_params = Frame._default_params + DefaultValues(
        shaq_output_folder=UserParam(None, constraint=PathValue, mandatory=True),
        override_BERV=UserParam(True, constraint=BooleanValue, mandatory=False),
        is_KOBE_data=UserParam(True, constraint=BooleanValue, mandatory=False),
    )

    sub_instruments = {
        "CARMENES": datetime.datetime.max,
    }
    _name = "CARMENES"

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

        coverage = [500, 1750]
        self._blaze_corrected = True

        KW_map = {
            "OBJECT": "OBJECT",
            "BJD": "HIERARCH CARACAL BJD",
            "MJD": "MJD-OBS",
            "ISO-DATE": "DATE-OBS",  # TODO: to check this KW name
            "DRS-VERSION": "HIERARCH CARACAL FOX VERSION",
            "RA": "RA",
            "DEC": "DEC",
        }

        super().__init__(
            inst_name="CARMENES",
            array_size={"S2D": [61, 4096]},
            file_path=file_path,
            frameID=frameID,
            KW_map=KW_map,
            available_indicators=("FWHM", "BIS SPAN"),
            user_configs=user_configs,
            reject_subInstruments=reject_subInstruments,
            need_external_data_load=True,
            quiet_user_params=quiet_user_params,
        )

        self.instrument_properties["wavelength_coverage"] = coverage
        self.instrument_properties["is_drift_corrected"] = False

        # TODO: ensure that there are no problem when using the NIR arm of CARMENES!!!!!
        self.instrument_properties["resolution"] = 94600

        # lat/lon from: https://geohack.toolforge.org/geohack.php?pagename=Calar_Alto_Observatory&params=37_13_25_N_2_32_46_W_type:landmark_region:ES
        # height from: https://en.wikipedia.org/wiki/Calar_Alto_Observatory
        lat, lon = 37.223611, -2.546111
        self.instrument_properties["EarthLocation"] = EarthLocation.from_geodetic(
            lat=lat, lon=lon, height=2168
        )

        # from https://www.mide.com/air-pressure-at-altitude-calculator
        # and convert to Pa to mbar
        self.instrument_properties["site_pressure"] = 778.5095

        self.is_BERV_corrected = False

    def get_spectral_type(self):
        name_lowercase = self.file_path.stem
        if "vis_A" in name_lowercase:
            return "S2D"
        else:
            raise custom_exceptions.InternalError(
                f"{self.name} can't recognize the file that it received ( - {self.file_path.stem})!"
            )

    def load_instrument_specific_KWs(self, header):
        self.observation_info["airmass"] = header[f"AIRMASS"]

        # Load BERV info + previous RV
        self.observation_info["MAX_BERV"] = 30 * kilometer_second
        self.observation_info["BERV"] = header["HIERARCH CARACAL BERV "] * kilometer_second

        # TODO: check ambient temperature on CARMENES data TO SEE IF IT IS THE "REAL ONE"
        # Environmental KWs for telfit (also needs airmassm previously loaded)
        ambi_KWs = {
            "relative_humidity": "AMBI RHUM",
            "ambient_temperature": "AMBI TEMPERATURE",
        }

        for name, endKW in ambi_KWs.items():
            self.observation_info[name] = header[f"HIERARCH CAHA GEN {endKW}"]
            if "temperature" in name:  # store temperature in KELVIN for TELFIT
                self.observation_info[name] = convert_temperature(
                    self.observation_info[name], old_scale="Celsius", new_scale="Kelvin"
                )
        for order in range(self.N_orders):
            self.observation_info["orderwise_SNRs"].append(
                header[f"HIERARCH CARACAL FOX SNR {order}"]
            )

    def load_S2D_data(self) -> Mask:
        if self.is_open:
            return
        super().load_S2D_data()

        with fits.open(self.file_path) as hdulist:
            s2d_data = hdulist["SPEC"].data * 100000  # spetra from all olders
            err_data = hdulist["SIG"].data * 100000
            wavelengths = hdulist["WAVE"].data  # vacuum wavelengths; no BERV correction

        self.wavelengths = wavelengths
        self.spectra = s2d_data
        self.uncertainties = err_data

        self.build_mask(bypass_QualCheck=True)
        return 1

    def finalize_data_load(self, bad_flag: Optional[Flag] = None) -> NoReturn:
        bad_flag = MISSING_SHAQ_RVS
        super().finalize_data_load(bad_flag)

    def load_S1D_data(self) -> Mask:
        raise NotImplementedError

    def build_mask(self, bypass_QualCheck: bool = False) -> None:
        super().build_mask(bypass_QualCheck)

        # remove extremely negative points!
        self.spectral_mask.add_indexes_to_mask(
            np.where(self.spectra < -3 * self.uncertainties), MISSING_DATA
        )
        self.spectral_mask.add_indexes_to_mask(np.where(self.uncertainties == 0), MISSING_DATA)

        self.assess_bad_orders()

    def close_arrays(self):
        super().close_arrays()
        self.is_BERV_corrected = False
