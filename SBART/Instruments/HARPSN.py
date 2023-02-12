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


class HARPSN(Frame):
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
        "HARPSN": datetime.datetime.max,
    }

    _name = "HARPSN"
    _default_params = Frame._default_params + DefaultValues(
        apply_FluxCorr=UserParam(False, constraint=BooleanValue),
        apply_FluxBalance_Norm=UserParam(False, constraint=BooleanValue),
        use_old_pipeline=UserParam(default_value=False, constraint=BooleanValue)
    )

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

        logger.info("Creating frame from: {}".format(file_path))

        coverage = [390, 700]
        search_status = SUCCESS
        if user_configs.get("use_old_pipeline", False): # For the old pipeline!
            KW_map = {
                "OBJECT": "HIERARCH TNG OBS TARG NAME",  # "OBJECT",
                "BJD": "HIERARCH TNG DRS BJD",
                "MJD": "MJD-OBS",
                "ISO-DATE": "DATE-OBS",
                "DRS-VERSION": "HIERARCH TNG DRS VERSION",
                "RA": "RA",
                "DEC": "DEC",
                "SPEC_TYPE": "HIERARCH TNG TEL TARG SPTYPE",
            }
            file_path, self.ccf_path, search_status = self.find_files(file_path)
            available_act = ("CONTRAST", "FWHM")
            self.is_BERV_corrected = False
        else:
            KW_map = {
                "OBJECT": "HIERARCH TNG OBS TARG NAME",  # "OBJECT",
                "BJD": "HIERARCH TNG QC BJD",
                "MJD": "MJD-OBS",
                "ISO-DATE": "DATE-OBS",
                "DRS-VERSION": "HIERARCH ESO PRO REC1 PIPE ID",
                "MD5-CHECK": "DATAMD5",
                "RA": "RA",
                "DEC": "DEC",
                "SPEC_TYPE": "HIERARCH TNG QC CCF MASK",
            }
            available_act = ("CONTRAST", "FWHM", "BIS SPAN")
            self.is_BERV_corrected = True

        super().__init__(
            inst_name="HARPSN",
            array_size={"S2D": (69, 4096)},
            file_path=file_path,
            frameID=frameID,
            KW_map=KW_map,
            available_indicators=available_act,
            user_configs=user_configs,
            reject_subInstruments=reject_subInstruments,
            init_log=False,
            quiet_user_params=quiet_user_params
        )

        if user_configs["use_old_pipeline"] and not search_status.is_good_flag:
            self.add_to_status(search_status)

        self.instrument_properties["wavelength_coverage"] = coverage
        self.instrument_properties["is_drift_corrected"] = True

        self.instrument_properties["resolution"] = 115_000
        self.instrument_properties["EarthLocation"] = EarthLocation.of_site(
            "Roque de los Muchachos"
        )

        # https://tngweb.tng.iac.es/weather/current
        self.instrument_properties["site_pressure"] = 770

        # CHeck for BLAZE correction
        self.is_blaze_corrected = True
        if "BLAZE" in self.file_path.stem:
            # The S2D_BLAZE_A files do not have the blaze correction!
            self.is_blaze_corrected = False

    def load_instrument_specific_KWs(self, header):
        if self._internal_configs["use_old_pipeline"]:
            self._load_old_DRS_KWs(header)
        else:
            self._load_new_DRS_KWs(header)

    def _load_old_DRS_KWs(self, header):
        if not self._internal_configs["use_old_pipeline"]:
            raise custom_exceptions.InvalidConfiguration("Can't load data from old pipeline without the config")

        self.observation_info["MAX_BERV"] = header["HIERARCH TNG DRS BERVMX"] * kilometer_second
        self.observation_info["BERV"] = header["HIERARCH TNG DRS BERV"] * kilometer_second

        # Environmental KWs for telfit (also needs airmassm previously loaded)

        ambi_KWs = {
            "relative_humidity": "HUMIDITY",
            "ambient_temperature": "TEMP10M",
        }

        for name, endKW in ambi_KWs.items():
            self.observation_info[name] = header[f"HIERARCH TNG METEO {endKW}"]
            if "temperature" in name:  # store temperature in KELVIN for TELFIT
                self.observation_info[name] = convert_temperature(
                    self.observation_info[name], old_scale="Celsius", new_scale="Kelvin"
                )

        if self.observation_info["relative_humidity"] == 255:
            logger.warning(f"{self.name} has an invalid value in the humidity sensor...")
            self.observation_info["relative_humidity"] = np.nan

        for order in range(self.N_orders):
            self.observation_info["orderwise_SNRs"].append(
                header[f"HIERARCH TNG DRS SPE EXT SN{order}"]
            )

        self.observation_info["airmass"] = header["AIRMASS"]

        bad_drift = False
        try:
            flag = "HIERARCH TNG DRS DRIFT QC"
            if header[flag].strip() != "PASSED":
                bad_drift = True
                msg = f"QC flag {flag} meets the bad value"
                logger.warning(msg)
                self._status.store_warning(KW_WARNING(msg))
            else:
                # self.logger.info("DRIFT QC has passed")
                drift = header["HIERARCH TNG DRS DRIFT RV USED"] * meter_second
                drift_err = header["HIERARCH TNG DRS DRIFT NOISE"] * meter_second
        except Exception as e:
            bad_drift = True
            logger.warning("DRIFT KW does not exist")

        if bad_drift:
            logger.warning("Due to previous drift-related problems, setting it to zero [m/s]")
            drift = 0 * meter_second
            drift_err = 0 * meter_second

        self.observation_info["drift"] = drift
        self.observation_info["drift_ERR"] = drift_err
        self._load_ccf_data()

    def _load_ccf_data(self) -> None:
        """
        Load the necessarfy CCF data from the file!
        """
        if not self._internal_configs["use_old_pipeline"]:
            raise custom_exceptions.InvalidConfiguration("Can't load data from old pipeline without the config")

        logger.debug("Loading data from the ccf file")
        header = fits.getheader(self.ccf_path)

        for key in self.available_indicators:
            full_key = "HIERARCH TNG DRS CCF " + key
            self.observation_info[key] = header[full_key]

        self.observation_info["DRS_RV"] = header["HIERARCH TNG DRS CCF RV"] * kilometer_second

        RV_err = header.get("HIERARCH TNG DRS DVRMS", None)
        if RV_err is None:
            logger.critical("Couldn't find DRS error from the appropriate KW. Estimating it with photon noise!")
            RV_err = np.sqrt(
                header["HIERARCH TNG DRS CAL TH ERROR"] ** 2
                +
                # hdulist[0].header['HIERARCH ESO DRS DRIFT NOISE']**2   +
                (1000 * header["HIERARCH TNG DRS CCF NOISE"]) ** 2
            )

        self.observation_info["DRS_RV_ERR"] = RV_err * meter_second

    def _load_new_DRS_KWs(self, header):
        if self._internal_configs["use_old_pipeline"]:
            raise custom_exceptions.InvalidConfiguration("Can't load data from new pipeline with the config for the old one")

        self.observation_info["MAX_BERV"] = header["HIERARCH TNG QC BERVMAX"] * kilometer_second
        self.observation_info["BERV"] = header["HIERARCH TNG QC BERV"] * kilometer_second

        self.observation_info["DRS_RV"] = header["HIERARCH TNG QC CCF RV"] * kilometer_second
        self.observation_info["DRS_RV_ERR"] = (
                header["HIERARCH TNG QC CCF RV ERROR"] * kilometer_second
        )

        # Environmental KWs for telfit (also needs airmassm previously loaded)
        ambi_KWs = {
            "relative_humidity": "HUMIDITY",
            "ambient_temperature": "TEMP10M",
        }

        for name, endKW in ambi_KWs.items():
            self.observation_info[name] = header[f"HIERARCH TNG METEO {endKW}"]
            if "temperature" in name:  # store temperature in KELVIN for TELFIT
                self.observation_info[name] = convert_temperature(
                    self.observation_info[name], old_scale="Celsius", new_scale="Kelvin"
                )

        for order in range(self.N_orders):
            self.observation_info["orderwise_SNRs"].append(
                header[f"HIERARCH TNG QC ORDER{order + 1} SNR"]
            )

        self.observation_info["airmass"] = header["AIRMASS"]  # "HIERARCH ESO TEL AIRM START"]

    def build_HARPS_wavelengths(self, hdr):
        """
        Compute the wavelength solution to this given spectra (EQ 4.1 of DRS manual)
        Convert from air wavelenbgths to vacuum
        """
        if not self._internal_configs["use_old_pipeline"]:
            raise custom_exceptions.InvalidConfiguration("Can't construct wavelengths for new pipeline")

        # degree of the polynomial
        d = hdr["HIERARCH TNG DRS CAL TH DEG LL"]
        # number of orders
        omax = hdr.get("HIERARCH TNG DRS CAL LOC NBO", self.array_size[0])
        xmax = self.array_size[1]

        # matrix X:
        #
        # axis 0: the entry corresponding to each coefficient
        # axis 1: each pixel number

        x = np.empty((d + 1, xmax), "int64")
        x[0].fill(1)  # x[0,*] = x^0 = 1,1,1,1,1,...
        x[1] = np.arange(xmax)

        for i in range(1, d):
            x[i + 1] = x[i] * x[1]

        # matrix A:
        #
        # axis 0: the different orders
        # axis 1: all coefficients for the given order

        A = np.reshape(
            [hdr["HIERARCH TNG DRS CAL TH COEFF LL" + str(i)] for i in range(omax * (d + 1))],
            (omax, d + 1),
        )  # slow 30 ms

        # the wavelengths for each order are a simple dot product between the coefficients and pixel-wise data (X)
        wavelengths = np.dot(A, x)

        vacuum_wavelengths = airtovac(wavelengths)
        return vacuum_wavelengths

    def load_old_pipeline_version(self):
        """
        load the data from the old HARPS-N pipeline. This will be mainly used for the comparison with the
        HARPS-TERRA pipeline
        """
        if not self._internal_configs["use_old_pipeline"]:
            raise custom_exceptions.InvalidConfiguration("Can't load data from old pipeline without the config")

        with fits.open(self.file_path) as hdulist:
            # Compute the wavelength solution + BERV correction
            wave_from_file = self.build_HARPS_wavelengths(hdulist[0].header)

            sci_data = hdulist[0].data  # spetra from all orders

            # photon noise + estimate of max value for the rest
            # from ETC calculator the readout noise should be the largest contribution
            # assuming that it is of ~7e- (equal to manual) it should have a maximum contribution
            # of 200
            flux_errors = np.sqrt(250 + np.abs(sci_data, dtype=float))

            # Validate for overflows and missing data
            quality_data = np.zeros(sci_data.shape)
            quality_data[np.where(np.isnan(sci_data))] = NAN_DATA.code
            quality_data[np.where(sci_data > 300000)] = SATURATION.code
            quality_data[np.where(sci_data < -3 * flux_errors)] = ERROR_THRESHOLD.code

        self.spectra = sci_data.astype(np.float64)
        self.wavelengths = wave_from_file
        self.qual_data = quality_data
        self.uncertainties = flux_errors.astype(np.float64)

        self.build_mask(bypass_QualCheck=False)
        self.apply_BERV_correction(self.get_KW_value("BERV"))

    def load_S2D_data(self):
        if self.is_open:
            logger.debug("{} has already been opened", self.__str__())
            return
        super().load_S2D_data()

        if self._internal_configs["use_old_pipeline"]:
            # The return is to ensure that we don't do anything after this point!
            return self.load_old_pipeline_version()

        with fits.open(self.file_path) as hdulist:

            self.wavelengths = hdulist["WAVEDATA_VAC_BARY"].data
            self.qual_data = hdulist["QUALDATA"].data

            SCIDATA_KEY = "SCIDATA"
            ERRDATA_KEY = "ERRDATA"

            # Fixing dtype to avoid problems with the cython interface
            self.spectra = hdulist[SCIDATA_KEY].data.astype(np.float64)
            self.uncertainties = hdulist[ERRDATA_KEY].data.astype(np.float64)

            if self._internal_configs["apply_FluxCorr"]:
                logger.debug("Starting chromatic flux correction")
                keyword = "HIERARCH TNG QC ORDER%d FLUX CORR"
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

    def load_S1D_data(self):
        with fits.open(self.file_path) as hdulist:
            full_data = hdulist[1].data

        self.wavelengths = full_data["wavelength"].reshape((1, self.array_size[1]))
        self.spectra = full_data["flux"].reshape((1, self.array_size[1])).astype(np.float64)
        self.uncertainties = full_data["error"].reshape((1, self.array_size[1])).astype(np.float64)
        self.qual_data = full_data["quality"].reshape((1, self.array_size[1]))
        self.build_mask(bypass_QualCheck=False)

    def check_header_QC(self, header):
        super().check_header_QC(header)
        if self._internal_configs["use_old_pipeline"]:
            logger.warning("No QC checks for data from the old pipeline!")
            return
        fatal_QC_flags = {"HIERARCH TNG QC SCIRED CHECK": 0}

        nonfatal_QC_flags = {
            "HIERARCH TNG QC SCIRED FLUX CORR CHECK": 0,
            "HIERARCH TNG QC SCIRED DRIFT CHECK": 0,
            "HIERARCH TNG QC SCIRED DRIFT FLUX_RATIO CHECK": 0,
            "HIERARCH TNG QC SCIRED DRIFT CHI2 CHECK": 0,
        }

        for flag, bad_value in fatal_QC_flags.items():
            if header[flag] == bad_value:
                msg = f"\tQC flag {flag} is not True"
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

    def close_arrays(self):
        """
        Reset the BERV correction flag if we are using the old pipeline version!
        Returns
        -------

        """
        super().close_arrays()
        if self._internal_configs["use_old_pipeline"]:
            self.is_BERV_corrected = False

    def find_files(self, file_name: UI_PATH):
        """
        Find the CCF and the S2D files, which should be stored inside the same folder
        """
        logger.debug("Searching for the ccf and e2ds files")

        if not isinstance(file_name, Path):
            file_name = Path(file_name)


        search_status = MISSING_DATA("Missing the ccf file")
        ccf_path = None

        if file_name.is_dir():
            logger.debug("Received a folder, searching inside for necessary files")
            # search for e2ds file
            folder_name = file_name

            e2ds_files = file_name.glob("**/*e2ds_A.fits")
            ccf_files = file_name.glob("**/*ccf_*_A.fits")

            for name, elems in [("e2ds_A", e2ds_files), ("ccf", ccf_files)]:
                if len(elems) > 1:
                    msg = f"HARPS data only received folder name and it has more than 1 {name} file in it"
                    raise custom_exceptions.InvalidConfiguration(msg)

                if len(elems) < 1:
                    msg = f"HARPS data only received folder name and it has no {name} file in it"
                    raise custom_exceptions.InvalidConfiguration(msg)

            e2ds_path = e2ds_files[0]
            ccf_path = ccf_files[0]
            search_status = SUCCESS("Found all input files")
        else:
            logger.debug("Received path of E2DS file; searching for CCF with matching name")
            folder_name = file_name.parent
            e2ds_path = file_name
            file_start, *_ = file_name.stem.split("_")

            found_CCF = False
            ccf_files = folder_name.glob("*ccf*_A.fits")

            for file in ccf_files:
                if file_start in file.name:
                    ccf_path = file
                    found_CCF = True

            if found_CCF:
                logger.info("Found CCF file: {}".format(ccf_path))
                search_status = SUCCESS("Found CCF file")
            else:
                logger.critical("Was not able to find CCF file. Marking frame as invalid")
                ccf_path = ""

        return e2ds_path, ccf_path, search_status
