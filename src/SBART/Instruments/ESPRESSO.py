import datetime
from typing import Any, Dict, Iterable, Optional

import numpy as np
from astropy.coordinates import EarthLocation
from loguru import logger
from scipy.constants import convert_temperature

from SBART.utils.status_codes import ERROR_THRESHOLD, KW_WARNING
from .ESO_PIPELINE import ESO_PIPELINE


class ESPRESSO(ESO_PIPELINE):
    """
    Interface to handle ESPRESSO observations (S2D and S1D).

    With ESPRESSO data we are considering 3 sub-Instruments:

    * ESPRESSO18 - Before  2019-06-27
    * ESPRESSO19 - Before  2020-12-18
    * ESPRESSO21 - Until the ends of time (hopefully)


    **User parameters:**

    ================================ ================ ================ ================ ================
    Parameter name                      Mandatory      Default Value    Valid Values    Comment
    ================================ ================ ================ ================ ================
    ================================ ================ ================ ================ ================

    .. note::
        Also check the **User parameters** of the parent classes for further customization options of SBART

    """

    _default_params = ESO_PIPELINE._default_params

    sub_instruments = {
        "ESPRESSO18": datetime.datetime.strptime("2019-06-27", r"%Y-%m-%d"),
        "ESPRESSO19": datetime.datetime.strptime("2020-12-18", r"%Y-%m-%d"),
        "ESPRESSO21": datetime.datetime.max,
    }
    _name = "ESPRESSO"

    def __init__(
        self,
        file_path,
        user_configs: Optional[Dict[str, Any]] = None,
        reject_subInstruments: Optional[Iterable[str]] = None,
        frameID: Optional[int] = None,
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
        # Wavelength coverage

        coverage = (350, 900)

        super().__init__(
            inst_name="ESPRESSO",
            array_size={"S2D": (170, 9111), "S1D": (1, 443262)},
            file_path=file_path,
            frameID=frameID,
            KW_identifier="ESO",
            user_configs=user_configs,
            reject_subInstruments=reject_subInstruments,
            quiet_user_params=quiet_user_params,
        )

        self.instrument_properties["wavelength_coverage"] = coverage
        self.instrument_properties["resolution"] = 140_000
        self.instrument_properties["EarthLocation"] = EarthLocation.of_site("Cerro Paranal")
        self.instrument_properties["is_drift_corrected"] = True

        # https://www.eso.org/sci/facilities/paranal/astroclimate/site.html
        self.instrument_properties["site_pressure"] = 750

    def load_telemetry_info(self, header):
        # Find the UT number and load the airmass
        for i in range(1, 5):
            try:
                self.observation_info["airmass"] = header[f"HIERARCH ESO TEL{i} AIRM START"]
                self.UT_number = i
                break
            except KeyError as e:
                if i == 4:
                    msg = "\tCannot find ESO TELx AIRM START key"
                    raise KeyError(msg) from e

        # Environmental KWs for telfit (also needs airmassm previously loaded)
        ambi_KWs = {
            "relative_humidity": "AMBI RHUM",
            "ambient_temperature": "AMBI TEMP",
            "seeing": "AMBI FWHM START",
        }

        for name, endKW in ambi_KWs.items():
            self.observation_info[name] = float(header[f"HIERARCH ESO TEL{self.UT_number} {endKW}"])
            if "temperature" in name:  # store temperature in KELVIN for TELFIT
                self.observation_info[name] = convert_temperature(
                    self.observation_info[name], old_scale="Celsius", new_scale="Kelvin"
                )

        self.observation_info["DET_BINX"] = header["HIERARCH ESO DET BINX"]
        self.observation_info["DET_BINY"] = header["HIERARCH ESO DET BINY"]

    def check_header_QC_ESO_DRS(self, header):
        nonfatal_QC_flags = {
            "HIERARCH ESO INS{} ADC{} RA": 0,  # related with ADC2 problem
            "HIERARCH ESO INS{} ADC{} dec": 0,  # related with ADC2 problem
            "HIERARCH ESO INS{} ADC{} SENS1": 0,  # related with ADC2 problem
            "HIERARCH ESO INS{} ADC{} TEMP": 0,  # related with ADC2 problem
        }
        found_ADC_issue = False
        for flag, bad_value in nonfatal_QC_flags.items():
            found_UT = False
            for UT_KW in ["", "2", "3", "4"]:
                try:
                    for ADC in [1, 2]:
                        ADC_KW = flag.format(UT_KW, ADC)
                        if header[ADC_KW] == bad_value:
                            msg = f"QC flag {ADC_KW} has a value of {bad_value}"
                            logger.warning(msg)
                            self._status.store_warning(KW_WARNING(msg))
                            found_ADC_issue = True
                        found_UT = True
                except:
                    pass
            if not found_UT:
                logger.critical(
                    f"Did not find the entry for the following UT related metric: {flag}"
                )

        if found_ADC_issue:
            self._status.store_warning(KW_WARNING("ADC2 issues found"))

        super().check_header_QC_ESO_DRS(header)

    def build_mask(self, bypass_QualCheck: bool = False) -> None:
        super().build_mask(bypass_QualCheck=bypass_QualCheck, assess_bad_orders=False)

        if self.spectral_format == "S2D":
            # the first two orders of the RED CCD have a large amount of noise in the beginning so we remove a
            # portion from the start of those two orders Now, what is going on: we want to find the indexes,
            # from order 90 and 91 that are below the 5230 \AA
            inds = np.where(self.wavelengths[90:92, :] <= 5230)
            # numpy where returns the indexes assuming the zero to be 90 and the one to be 91. Remember that we
            # sliced the array to only remove from those two orders
            inds_1 = np.where(inds[0], 91, 90)
            # rebuild the 'numpy where' output, to pass to the mask as the proper output
            inds = (inds_1, inds[1])
            self.spectral_mask.add_indexes_to_mask(inds, ERROR_THRESHOLD)

        self.assess_bad_orders()
