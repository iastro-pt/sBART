import datetime
import glob
import os
from typing import Any, Dict, Optional

import numpy as np
from astropy.coordinates import EarthLocation
from astropy.io import fits
from loguru import logger
from scipy.constants import convert_temperature

from SBART.Masks import Mask
from SBART.utils import custom_exceptions
from SBART.utils.RV_utilities import airtovac
from SBART.utils.status_codes import (
    ERROR_THRESHOLD,
    KW_WARNING,
    MISSING_DATA,
    NAN_DATA,
    SATURATION,
    SUCCESS,
)
from SBART.utils.units import convert_data, kilometer_second, meter_second

from .ESO_PIPELINE import ESO_PIPELINE


class HARPS(ESO_PIPELINE):
    """Interface to handle HARPS data; S1D **not** supported!

    This class also defines 2 sub-Instruments:

    * HARPSpre - Before  2015-05-29
    * HARPSpost - Until the ends of time (hopefully)

    The steps to load the S2D data are described in the HARPS `DRS manual <https://www.eso.org/sci/facilities/lasilla/instruments/harps/doc/DRS.pdf>`_. The summary is:

        - Construct the wavelength solution & correct from BERV
        - Load instrumental drift
        - Construct flux noises:

            - Table 10 of `the user manual <https://www.eso.org/sci/facilities/lasilla/instruments/harps/doc/manual/HARPS-UserManual2.4.pdf>`_ gives max RON of 7.07 for red detector
            - Noise = sqrt(obj + sky + n*dark*expTime + nBinY*ron^2)

    **User parameters:**

     Currently there are no HARPS-specific parameters

    *Note:* Check the **User parameters** of the parent classes for further customization options of SBART

    """

    sub_instruments = {
        "HARPS03": datetime.datetime.strptime("2015-05-29", r"%Y-%m-%d"),
        "HARPS15": datetime.datetime.max,
    }

    _name = "HARPS"
    _default_params = ESO_PIPELINE._default_params

    def __init__(
        self,
        file_path,
        user_configs: Optional[Dict[str, Any]] = None,
        reject_subInstruments=None,
        frameID=None,
        quiet_user_params: bool = True,
    ):
        """HARPS constructor.

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
        logger.info(f"Creating frame from: {file_path}")

        coverage = [350, 700]
        # Note: 46 blue orders and 26 red orders. From Table 2.2 of:
        # https://www.eso.org/sci/facilities/lasilla/instruments/harps/doc/manual/HARPS-UserManual2.4.pdf
        if user_configs["use_old_pipeline"]:
            mat_size = (72, 4096)
            KW_map = {
                "OBJECT": "OBJECT",
                "BJD": "HIERARCH ESO DRS BJD",
                "MJD": "MJD-OBS",
                "ISO-DATE": "DATE-OBS",
                "DRS-VERSION": "HIERARCH ESO DRS VERSION",
                "RA": "RA",
                "DEC": "DEC",
                "MD5-CHECK": "DATASUM",
                "SPEC_TYPE": None,
                "DRS_CCF_MASK": None,
                "DRS_FLUX_CORRECTION_TEMPLATE": None,
            }
        else:
            mat_size = (71, 4096)
            KW_map = {}

        if user_configs["use_old_pipeline"]:
            file_path, self.ccf_path, self.BIS_file, search_status = self.find_files(file_path)

        super().__init__(
            inst_name="HARPS",
            array_size={"S2D": mat_size},
            file_path=file_path,
            KW_identifier="ESO",
            frameID=frameID,
            user_configs=user_configs,
            reject_subInstruments=reject_subInstruments,
            quiet_user_params=quiet_user_params,
            override_KW_map=KW_map,
            override_indicators=("CONTRAST", "FWHM"),
        )

        if user_configs["use_old_pipeline"] and not search_status.is_good_flag:
            self.add_to_status(search_status)

        self.instrument_properties["wavelength_coverage"] = coverage

        if user_configs["use_old_pipeline"]:
            self.instrument_properties["is_drift_corrected"] = False
            self.is_BERV_corrected = False

        self.instrument_properties["resolution"] = 115_000
        self.instrument_properties["EarthLocation"] = EarthLocation.of_site("La Silla Observatory")
        # ? same as for Paranal?
        # https://www.eso.org/sci/facilities/paranal/astroclimate/site.html
        self.instrument_properties["site_pressure"] = 750
        self.is_blaze_corrected = False

    def load_telemetry_info(self, header):
        """Loads (at least) the following keywords:

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
            "relative_humidity": "RHUM",
            "ambient_temperature": "TEMP",
        }

        for name, endKW in ambi_KWs.items():
            self.observation_info[name] = header[f"HIERARCH {self.KW_identifier} TEL AMBI {endKW}"]
            if "temperature" in name:  # store temperature in KELVIN for TELFIT
                self.observation_info[name] = convert_temperature(
                    self.observation_info[name],
                    old_scale="Celsius",
                    new_scale="Kelvin",
                )

        if self.observation_info["relative_humidity"] == 255:
            logger.warning(f"{self.name} has an invalid value in the humidity sensor...")
            self.observation_info["relative_humidity"] = np.nan

        self.observation_info["airmass"] = header["HIERARCH ESO TEL AIRM START"]

    def find_files(self, file_name):
        """Find the CCF and S2D files and BIS files, which should be stored inside the same folder"""
        logger.debug("Searching for the ccf and e2ds files")

        search_status = MISSING_DATA("Missing the ccf file")
        ccf_path = None
        bis_path = None

        if os.path.isdir(file_name):
            logger.debug("Received a folder, searching inside for necessary files")
            # search for e2ds file
            folder_name = file_name

            e2ds_files = glob.glob(os.path.join(folder_name, "*e2ds_A.fits"), recursive=True)
            ccf_files = glob.glob(os.path.join(folder_name, "*ccf*A.fits"), recursive=True)
            bis_files = glob.glob(os.path.join(folder_name, "*bis*A.fits"), recursive=True)

            for name, elems in [
                ("e2ds_A", e2ds_files),
                ("ccf", ccf_files),
                ("bis", bis_files),
            ]:
                if len(elems) > 1:
                    msg = f"HARPS data only received folder name and it has more than 1 {name} file in it"
                    raise custom_exceptions.InvalidConfiguration(msg)

                if len(elems) < 1:
                    msg = f"HARPS data only received folder name and it has no {name} file in it"
                    if name != "bis":
                        # The BIS file is not critical for the run
                        raise custom_exceptions.InvalidConfiguration(msg)
                    logger.critical(msg)

            e2ds_path = e2ds_files[0]
            ccf_path = ccf_files[0]
            bis_path = bis_files[0]
            search_status = SUCCESS("Found all input files")
        else:
            logger.debug("Received path of E2DS file; searching for CCF with matching name")
            folder_name = os.path.dirname(file_name)
            e2ds_path = file_name
            file_start, *_ = os.path.basename(file_name).split("_")

            found_CCF = False
            found_BIS = False
            ccf_files = glob.glob(os.path.join(folder_name, "*ccf*A.fits"), recursive=True)
            bis_files = glob.glob(os.path.join(folder_name, "*bis*A.fits"), recursive=True)
            for file in ccf_files:
                if file_start in file:
                    ccf_path = file
                    found_CCF = True

            for file in bis_files:
                if file_start in file:
                    bis_path = file
                    found_BIS = True
            if found_CCF:
                logger.info(f"Found CCF file: {ccf_path}")
                search_status = SUCCESS("Found CCF file")
            else:
                logger.critical("Was not able to find CCF file. Marking frame as invalid")
                ccf_path = ""

            if not found_BIS:
                bis_path = None

        return e2ds_path, ccf_path, bis_path, search_status

    def build_HARPS_wavelengths(self, hdr):
        """Compute the wavelength solution to this given spectra (EQ 4.1 of DRS manual)
        Convert from air wavelenbgths to vacuum
        """
        # degree of the polynomial
        d = hdr["HIERARCH ESO DRS CAL TH DEG LL"]
        # number of orders
        omax = hdr.get("HIERARCH ESO DRS CAL LOC NBO", self.array_size[0])
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
            [hdr["HIERARCH ESO DRS CAL TH COEFF LL" + str(i)] for i in range(omax * (d + 1))],
            (omax, d + 1),
        )  # slow 30 ms

        # the wavelengths for each order are a simple dot product between the coefficients and pixel-wise data (X)
        wavelengths = np.dot(A, x)

        vacuum_wavelengths = airtovac(wavelengths)
        return vacuum_wavelengths

    def _load_old_DRS_KWs(self, header):
        if not self._internal_configs["use_old_pipeline"]:
            raise custom_exceptions.InvalidConfiguration("Can't load data from old pipeline without the config")

        self.observation_info["MAX_BERV"] = header["HIERARCH ESO DRS BERVMX"] * kilometer_second
        self.observation_info["BERV"] = header["HIERARCH ESO DRS BERV"] * kilometer_second

        # Environmental KWs for telfit (also needs airmassm previously loaded)
        ambi_KWs = {
            "relative_humidity": "AMBI RHUM",
            "ambient_temperature": "AMBI TEMP",
        }

        for name, endKW in ambi_KWs.items():
            self.observation_info[name] = header[f"HIERARCH ESO TEL {endKW}"]
            if "temperature" in name:  # store temperature in KELVIN for TELFIT
                self.observation_info[name] = convert_temperature(
                    self.observation_info[name],
                    old_scale="Celsius",
                    new_scale="Kelvin",
                )

        for order in range(self.N_orders):
            self.observation_info["orderwise_SNRs"].append(header[f"HIERARCH ESO DRS SPE EXT SN{order}"])

        self.observation_info["airmass"] = header["HIERARCH ESO TEL AIRM START"]

        bad_drift = False
        try:
            flag = "HIERARCH ESO DRS DRIFT QC"
            if header[flag].strip() != "PASSED":
                bad_drift = True
                msg = f"QC flag {flag} meets the bad value"
                logger.warning(msg)
                self._status.store_warning(KW_WARNING(msg))
            else:
                # self.logger.info("DRIFT QC has passed")
                drift = header["HIERARCH ESO DRS DRIFT RV USED"] * meter_second
                drift_err = header["HIERARCH ESO DRS DRIFT NOISE"] * meter_second
        except Exception:
            bad_drift = True
            logger.warning("DRIFT KW does not exist")

        if bad_drift:
            logger.warning("Due to previous drift-related problems, setting it to zero [m/s]")
            drift = 0 * meter_second
            drift_err = 0 * meter_second

        self.observation_info["drift"] = drift
        self.observation_info["drift_ERR"] = drift_err
        self.load_ccf_data()

    def load_ccf_data(self) -> None:
        """Load the necessarfy CCF data from the file!"""
        logger.debug("Loading data from the ccf file")
        header = fits.getheader(self.ccf_path)

        self.observation_info["DRS_RV"] = header["HIERARCH ESO DRS CCF RV"] * kilometer_second
        self.observation_info["SPEC_TYPE"] = header["HIERARCH ESO DRS CCF MASK"]

        RV_err = np.sqrt(
            header["HIERARCH ESO DRS CAL TH ERROR"] ** 2
            +
            # hdulist[0].header['HIERARCH ESO DRS DRIFT NOISE']**2   +
            (1000 * header["HIERARCH ESO DRS CCF NOISE"]) ** 2,
        )
        self.observation_info["DRS_RV_ERR"] = RV_err * meter_second

        for key in self.available_indicators:
            full_key = "HIERARCH ESO DRS CCF " + key
            self.observation_info[key] = header[full_key]

        # We are missing error in CONTRAST!
        self.observation_info["FWHM_ERR"] = convert_data(
            2.35 * RV_err * meter_second,
            new_units=kilometer_second,
            as_value=True,
        )
        self.observation_info["BIS SPAN_ERR"] = convert_data(
            np.sqrt(2) * RV_err * meter_second,
            new_units=kilometer_second,
            as_value=True,
        )

        if self.BIS_file is not None:
            head = fits.getheader(self.BIS_file)
            self.observation_info["BIS SPAN"] = head["HIERARCH ESO DRS BIS SPAN"]

    def load_S1D_data(self) -> Mask:
        raise NotImplementedError

    def check_header_QC_old_DRS(self, header):
        logger.info("Currently missing QC checks for the old DRS")

    def load_old_DRS_S2D(self):
        """Loads the spectra

        Returns
        -------

        """
        super().load_S2D_data()

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
        return 1

    def close_arrays(self):
        super().close_arrays()
        if self._internal_configs["use_old_pipeline"]:
            self.is_BERV_corrected = False
