from typing import Any, Dict, Iterable, Optional

import numpy as np
from astropy.io import fits
from loguru import logger
from scipy.constants import convert_temperature

from SBART.Base_Models.Frame import Frame
from SBART.utils import custom_exceptions
from SBART.utils.UserConfigs import BooleanValue, DefaultValues, UserParam
from SBART.utils.status_codes import FATAL_KW, KW_WARNING
from SBART.utils.units import kilometer_second


class ESO_PIPELINE(Frame):
    """
    Interface to handle data products (S2D and S1D) of the ESO pipeline (DRS 3.0)

    **User parameters:**

    ================================ ================ ================ ================ ================
    Parameter name                      Mandatory      Default Value    Valid Values    Comment
    ================================ ================ ================ ================ ================
    apply_FluxCorr                    False              False         boolean          Apply polynomial flux correction to flux to preserve red-blue balance
    Telluric_Corrected                False              False         boolean          Use S2D data in the format given by R. Allart's correction tool
    ================================ ================ ================ ================ ================

    .. note::
        Also check the **User parameters** of the parent classes for further customization options of SBART

    """

    _default_params = Frame._default_params + DefaultValues(
        Telluric_Corrected=UserParam(
            False,
            constraint=BooleanValue,
            description="The Frame was already corrected from telluric features",
        ),
        UseMolecfit=UserParam(False, constraint=BooleanValue),
        use_old_pipeline=UserParam(
            default_value=False,
            constraint=BooleanValue,
            description="Use data from the old pipeline. Only available to selected instruments",
        ),
        SCIRED_CHECK_IS_FATAL=UserParam(
            default_value=True,
            constraint=BooleanValue,
            description="Automatically reject frames with QC SCIRED CHECK = 0 ",
        ),
    )

    _default_params.update(
        "apply_FluxCorr",
        UserParam(
            False,
            constraint=BooleanValue,
            description="Apply the blue-red flux correction due to the wavelength dependence of the atmospheric extinction. Only available on data from ESO pipeline (ESPRESSO)",
        ),
    )

    _default_params.update(
        "apply_FluxBalance_Norm",
        UserParam(False, constraint=BooleanValue),
    )

    def __init__(
        self,
        inst_name,
        array_size,
        file_path,
        KW_identifier,
        user_configs: Optional[Dict[str, Any]] = None,
        reject_subInstruments: Optional[Iterable[str]] = None,
        frameID: Optional[int] = None,
        quiet_user_params: bool = True,
        override_KW_map=None,
        override_indicators=None,
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

        self.UT_number = None
        self.KW_identifier = KW_identifier

        KW_map = {
            "OBJECT": "OBJECT",
            "BJD": f"HIERARCH {KW_identifier} QC BJD",
            "MJD": "MJD-OBS",
            "ISO-DATE": "DATE-OBS",
            "DRS-VERSION": f"HIERARCH ESO PRO REC1 PIPE ID",
            "MD5-CHECK": "DATAMD5",
            "RA": "RA",
            "DEC": "DEC",
            "SPEC_TYPE": f"HIERARCH {KW_identifier} QC CCF MASK",
            "EXPTIME": "EXPTIME",
        }
        if override_KW_map is not None:
            for key, value in override_KW_map.items():
                KW_map[key] = value

        super().__init__(
            inst_name=inst_name,
            array_size=array_size,
            file_path=file_path,
            frameID=frameID,
            KW_map=KW_map,
            available_indicators=("CONTRAST", "FWHM", "BIS SPAN")
            if override_indicators is None
            else override_indicators,
            user_configs=user_configs,
            reject_subInstruments=reject_subInstruments,
            quiet_user_params=quiet_user_params,
        )

        self.instrument_properties["is_drift_corrected"] = True

        self.is_BERV_corrected = True

        # CHeck for BLAZE correction
        self.is_blaze_corrected = True
        if "BLAZE" in self.file_path.stem:
            # The S2D_BLAZE_A files do not have the blaze correction!
            self.is_blaze_corrected = False

    def load_instrument_specific_KWs(self, header):
        if self._internal_configs["use_old_pipeline"]:
            self._load_old_DRS_KWs(header)
        else:
            self._load_ESO_DRS_KWs(header)
        self.load_telemetry_info(header)

    def load_S2D_data(self):
        if self.is_open:
            logger.debug("{} has already been opened", self.__str__())
            return
        super().load_S2D_data()

        if self._internal_configs["use_old_pipeline"]:
            # The return is to ensure that we don't do anything after this point!
            return self.load_old_DRS_S2D()
        else:
            self.load_ESO_DRS_S2D_data()

    def load_S1D_data(self):
        super().load_S1D_data()

        if self._internal_configs["use_old_pipeline"]:
            # The return is to ensure that we don't do anything after this point!
            return self.load_old_DRS_S1D()
        else:
            self.load_ESO_DRS_S1D_data()

    def check_header_QC(self, header: fits.header.Header):
        super().check_header_QC(header)
        if self._internal_configs["use_old_pipeline"]:
            self.check_header_QC_old_DRS(header)
        else:
            self.check_header_QC_ESO_DRS(header)

    ## Add the bindings for a possible old DRS version

    def _load_old_DRS_KWs(self, header):
        raise NotImplementedError(f"There is no old pipeline for {self.name}")

    def check_header_QC_old_DRS(self, header):
        raise NotImplementedError(f"There is no old pipeline for {self.name}")

    def load_old_DRS_S2D(self):
        raise NotImplementedError(f"There is no old pipeline for {self.name}")

    def load_old_DRS_S1D(self):
        raise NotImplementedError(f"There is no old pipeline for {self.name}")

    # Load common information to the ESO DRS version across different spectrographs
    def load_telemetry_info(self, header):
        """
        Loads (at least) the following keywords:

        - relative humidity
        - ambient temperature, in Celsius
        - airmass
        - Detector

        Parameters
        ----------
        header

        Returns
        -------

        """

        ambi_KWs = {
            "relative_humidity": "HUMIDITY",
            "ambient_temperature": "TEMP10M",
        }

        for name, endKW in ambi_KWs.items():
            self.observation_info[name] = header[f"HIERARCH {self.KW_identifier} METEO {endKW}"]
            if "temperature" in name:  # store temperature in KELVIN for TELFIT
                self.observation_info[name] = convert_temperature(
                    self.observation_info[name], old_scale="Celsius", new_scale="Kelvin"
                )

        if self.observation_info["relative_humidity"] == 255:
            logger.warning(f"{self.name} has an invalid value in the humidity sensor...")
            self.observation_info["relative_humidity"] = np.nan

        self.observation_info["airmass"] = header["AIRMASS"]

    def _load_ESO_DRS_KWs(self, header):
        if self._internal_configs["use_old_pipeline"]:
            raise custom_exceptions.InvalidConfiguration(
                "Can't load data from new pipeline with the config for the old one"
            )

        # Load BERV info + previous RV
        self.observation_info["MAX_BERV"] = (
            header[f"HIERARCH {self.KW_identifier} QC BERVMAX"] * kilometer_second
        )
        self.observation_info["BERV"] = (
            header[f"HIERARCH {self.KW_identifier} QC BERV"] * kilometer_second
        )

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

        # Chosen activity indicators
        for key in self.available_indicators:
            full_key = f"HIERARCH {self.KW_identifier} QC CCF {key}"
            self.observation_info[key] = header[full_key]

    def load_ESO_DRS_S2D_data(self, overload_SCIDATA_key=None):
        if self._internal_configs["use_old_pipeline"]:
            raise custom_exceptions.InvalidConfiguration(
                "Can't load data from new pipeline with the config for the old one"
            )

        with fits.open(self.file_path) as hdulist:
            self.wavelengths = hdulist["WAVEDATA_VAC_BARY"].data
            self.qual_data = hdulist["QUALDATA"].data

            SCIDATA_KEY = "SCIDATA" if overload_SCIDATA_key is None else overload_SCIDATA_key
            ERRDATA_KEY = "ERRDATA"

            if self._internal_configs["Telluric_Corrected"]:
                logger.info("Loading S2D file from a non-DRS source (telluric corrected file)")
                SCIDATA_KEY += "_CORR"
                ERRDATA_KEY += "_CORR"

            # Fixing dtype to avoid problems with the cython interface
            self.spectra = hdulist[SCIDATA_KEY].data.astype(np.float64)
            self.uncertainties = hdulist[ERRDATA_KEY].data.astype(np.float64)
            if self._internal_configs["apply_FluxCorr"]:
                logger.debug("Starting chromatic flux correction")
                keyword = f"HIERARCH {self.KW_identifier} QC ORDER%d FLUX CORR"
                flux_corr = np.array(
                    [hdulist[0].header[keyword % o] for o in range(1, self.N_orders + 1)]
                )
                fit_nb = (flux_corr != 1.0).sum()

                ignore = self.N_orders - fit_nb
                logger.debug(f"Chromatic correction ignoring {ignore} orders")

                # ? see espdr_scince:espdr_correct_flux
                poly_deg = round(8 * fit_nb / self.N_orders)
                logger.debug("Fitting polynomial (n={})".format(poly_deg))
                llc = hdulist[5].data[:, self.array_size[1] // 2]
                coeff = np.polyfit(llc[ignore:], flux_corr[ignore:], poly_deg - 1)
                # corr_model = np.zeros_like(hdu[5].data, dtype=np.float32)
                corr_model = np.polyval(coeff, hdulist[5].data)

                corr_model[
                    flux_corr == 1
                ] = 1  # orders where the CORR FACTOR are 1 do not have correction!
                self.spectra = self.spectra / corr_model  # correct from chromatic variations
                self.flux_atmos_balance_corrected = True
                # TODO: understand if we want to include the factor in uncertainties or not!
                # self.uncertainties = self.uncertainties / corr_model # maintain the SNR in the corrected spectrum

            else:
                # Disabled the flux correction as we are artifically increasing the SNR of the spectra...
                # Shouldn't we also increase the flux uncertainty?? The DRS does not do it....
                logger.warning("Not applying correction to blue-red flux balance!")
                # / corr_model

            if self._internal_configs["apply_FluxBalance_Norm"]:
                logger.info("Normalizing the flux balance distribution due to dispersion")
                # The physical sizes of the pixels (on the CCD) are the same
                # The flux that reaches eeach pixel is different, due to dispersion
                # The spectra will have a trend, even after removing the instrumental effect
                # This normalizes the spectra by dividing by the flux distribution

                balance_corr_model = hdulist["DLLDATA_VAC_BARY"].data
                self.spectra = self.spectra / balance_corr_model
                # Ensure that we keep the same SNR after the normalization!
                self.uncertainties = self.uncertainties / balance_corr_model
                self.flux_dispersion_balance_corrected = True

        self.build_mask(bypass_QualCheck=False)
        return 1

    def load_ESO_DRS_S1D_data(self):
        if self._internal_configs["use_old_pipeline"]:
            raise custom_exceptions.InvalidConfiguration(
                "Can't load data from new pipeline with the config for the old one"
            )

        with fits.open(self.file_path) as hdulist:
            full_data = hdulist[1].data

        wave_kw = "wavelength"
        if self._internal_configs["use_air_wavelengths"]:
            wave_kw = "wavelength_air"
            logger.warning("SBART using air wavelengths!")

        self.wavelengths = full_data[wave_kw].reshape((1, self.array_size[1]))
        self.spectra = full_data["flux"].reshape((1, self.array_size[1])).astype(np.float64)
        self.uncertainties = full_data["error"].reshape((1, self.array_size[1])).astype(np.float64)
        self.qual_data = full_data["quality"].reshape((1, self.array_size[1]))
        self.build_mask(bypass_QualCheck=False)

    def _compute_BLAZE(self):
        """
        Assume that S2D and S2D_BLAZE live within the same folder (should be true for most cases)
        Returns
        -------

        """
        if "BLAZE" in self.file_path:
            blaze_corr_file = self.file_path
            s2d_file = self.file_path.replace("S2D_BLAZE_A", "S2D_A")
        else:
            s2d_file = self.file_path
            blaze_corr_file = self.file_path.replace("S2D_A", "S2D_BLAZE_A")

        with fits.open(s2d_file) as hdu:
            S2D = hdu[1].data
        with fits.open(blaze_corr_file) as hdu:
            blaze_corr = hdu[1].data

        # this will raise errors on the edges of the orders (zero-division)
        self._blaze_function = blaze_corr / S2D

    def check_header_QC_ESO_DRS(self, header):
        fatal_QC_flags = {}

        nonfatal_QC_flags = {
            f"HIERARCH {self.KW_identifier}" + " QC SCIRED FLUX CORR CHECK": 0,
            f"HIERARCH {self.KW_identifier}" + " QC SCIRED DRIFT CHECK": 0,
            f"HIERARCH {self.KW_identifier}" + " QC SCIRED DRIFT FLUX_RATIO CHECK": 0,
            f"HIERARCH {self.KW_identifier}" + " QC SCIRED DRIFT CHI2 CHECK": 0,
        }

        if self._internal_configs["SCIRED_CHECK_IS_FATAL"]:
            fatal_QC_flags[f"HIERARCH {self.KW_identifier} QC SCIRED CHECK"] = 0
        else:
            nonfatal_QC_flags[f"HIERARCH {self.KW_identifier} QC SCIRED CHECK"] = 0

        for flag, bad_value in fatal_QC_flags.items():
            if header[flag] == bad_value:
                msg = f"\tQC flag {flag} has taken the bad value of {bad_value}"
                logger.critical(msg)
                self.add_to_status(FATAL_KW(msg.replace("\t", "")))

        for flag, bad_value in nonfatal_QC_flags.items():
            try:
                if header[flag] == bad_value:
                    msg = f"QC flag {flag} meets the bad value"
                    logger.warning(msg)
                    self._status.store_warning(KW_WARNING(msg))
            except KeyError:
                msg = f"QC flag {flag} does not exist"
                logger.warning(msg)
                self._status.store_warning(KW_WARNING(msg))

        if self._status.number_warnings > 0:
            logger.warning("Found {} warning flags in the header KWs", self._status.number_warnings)

    @property
    def bare_fname(self) -> str:
        return self.fname.split("_S")[0]
