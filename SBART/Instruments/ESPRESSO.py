import datetime
from typing import Any, Dict, Iterable, Optional

import numpy as np
from astropy.coordinates import EarthLocation
from astropy.io import fits
from iCCF import gaussfit
from loguru import logger
from scipy.constants import convert_temperature

from SBART.Base_Models import Frame
from SBART.utils.RV_utilities.CCF_errors import ccffitRV
from SBART.utils.status_codes import ERROR_THRESHOLD, FATAL_KW, KW_WARNING
from SBART.utils.units import kilometer_second
from SBART.utils.UserConfigs import BooleanValue, DefaultValues, UserParam


class ESPRESSO(Frame):
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
    apply_FluxCorr                    False              False         boolean          Apply polynomial flux correction to flux to preserve red-blue balance
    Telluric_Corrected                False              False         boolean          Use S2D data in the format given by R. Allart's correction tool
    ================================ ================ ================ ================ ================

    *Note:* Also check the **User parameters** of the parent classes for further customization options of SBART

    """

    _default_params = Frame._default_params + DefaultValues(
        apply_FluxCorr=UserParam(False, constraint=BooleanValue),
        Telluric_Corrected=UserParam(False, constraint=BooleanValue),
    )

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
        try:
            spec_fmt = user_configs["spectra_format"]
        except (KeyError, TypeError):
            spec_fmt = self.__class__._default_params["spectra_format"].default_value

        coverage = (350, 900)
        if spec_fmt == "S2D":
            mat_size = (170, 9111)
        else:
            mat_size = (1, 443262)

        self.UT_number = None
        KW_map = {
            "OBJECT": "OBJECT",
            "BJD": "HIERARCH ESO QC BJD",
            "MJD": "MJD-OBS",
            "ISO-DATE": "DATE-OBS",
            "DRS-VERSION": "HIERARCH ESO PRO REC1 PIPE ID",
            "MD5-CHECK": "DATAMD5",
            "RA": "RA",
            "DEC": "DEC",
            "SPEC_TYPE": "HIERARCH ESO QC CCF MASK",
        }

        super().__init__(
            inst_name="ESPRESSO",
            array_size=mat_size,
            file_path=file_path,
            frameID=frameID,
            KW_map=KW_map,
            available_indicators=("CONTRAST", "FWHM", "BIS SPAN"),
            user_configs=user_configs,
            reject_subInstruments=reject_subInstruments,
        )

        self.__class__.instrument_properties["wavelength_coverage"] = coverage
        self.__class__.instrument_properties["resolution"] = 140_000
        self.__class__.instrument_properties["EarthLocation"] = EarthLocation.of_site(
            "Cerro Paranal"
        )
        self.__class__.instrument_properties["is_drift_corrected"] = True

        # https://www.eso.org/sci/facilities/paranal/astroclimate/site.html
        self.__class__.instrument_properties["site_pressure"] = 750
        self.is_BERV_corrected = True

        # CHeck for BLAZE correction
        self.is_blaze_corrected = True
        if "BLAZE" in self.file_path:
            # The S2D_BLAZE_A files do not have the blaze correction!
            self.is_blaze_corrected = False

    def load_instrument_specific_KWs(self, header):
        # Find the UT number and load the airmass
        for i in range(1, 5):
            try:
                self.observation_info["airmass"] = header[f"HIERARCH ESO TEL{i} AIRM START"]
                self.UT_number = i
                break
            except KeyError as e:
                if i == 4:
                    msg = "\tCannot find ESO TELx AIRM START key"
                    raise KeyError(msg)

        # Load BERV info + previous RV
        self.observation_info["MAX_BERV"] = header["HIERARCH ESO QC BERVMAX"] * kilometer_second
        self.observation_info["BERV"] = header["HIERARCH ESO QC BERV"] * kilometer_second

        self.observation_info["DRS_RV"] = header["HIERARCH ESO QC CCF RV"] * kilometer_second
        self.observation_info["DRS_RV_ERR"] = (
            header["HIERARCH ESO QC CCF RV ERROR"] * kilometer_second
        )

        # Environmental KWs for telfit (also needs airmassm previously loaded)
        ambi_KWs = {
            "relative_humidity": "AMBI RHUM",
            "ambient_temperature": "AMBI TEMP",
        }

        for name, endKW in ambi_KWs.items():
            self.observation_info[name] = float(header[f"HIERARCH ESO TEL{self.UT_number} {endKW}"])
            if "temperature" in name:  # store temperature in KELVIN for TELFIT
                self.observation_info[name] = convert_temperature(
                    self.observation_info[name], old_scale="Celsius", new_scale="Kelvin"
                )
        for order in range(self.N_orders):
            self.observation_info["orderwise_SNRs"].append(
                header[f"HIERARCH ESO QC ORDER{order + 1} SNR"]
            )

        # Chosen activity indicators
        for key in self.available_indicators:
            full_key = "HIERARCH ESO QC CCF " + key
            self.observation_info[key] = header[full_key]

    def load_S2D_data(self):
        if self.is_open:
            logger.debug("{} has already been opened", self.__str__())
            return
        super().load_S2D_data()

        with fits.open(self.file_path) as hdulist:

            self.wavelengths = hdulist["WAVEDATA_VAC_BARY"].data
            self.qual_data = hdulist["QUALDATA"].data

            SCIDATA_KEY = "SCIDATA"
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
                keyword = "HIERARCH ESO QC ORDER%d FLUX CORR"
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

                # TODO: understand if we want to include the factor in uncertainties or not!
                # self.uncertainties = self.uncertainties / corr_model # maintain the SNR in the corrected spectrum

            else:
                # Disabled the flux correction as we are artifically increasing the SNR of the spectra...
                # Shouldn't we also increase the flux uncertainty?? The DRS does not do it....

                logger.warning("Not applying correction to blue-red flux balance!")
                # / corr_model

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

    def build_mask(self, bypass_QualCheck: bool = False) -> None:
        super().build_mask(bypass_QualCheck=bypass_QualCheck, assess_bad_orders=False)

        if self._internal_configs["spectra_format"] == "S2D":
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

    def CCF_data(self):
        ccf_paths = [i.replace("S2D_BLAZE_A", "CCF_A") for i in self.folder_names]

        RVs = {i: [] for i in range(self.N_epochs)}
        ERRORS = {i: [] for i in range(self.N_epochs)}
        for epo_index, path in enumerate(ccf_paths):
            with fits.open(path) as hdulist:
                sci = hdulist["SCIDATA"].data
                err = hdulist["ERRDATA"].data
                qual = hdulist["QUALDATA"].data
                qual[np.where(qual != 0)] = 1
                qual.astype(bool)

                start = hdulist[0].header["HIERARCH ESO RV START"]
                step = hdulist[0].header["HIERARCH ESO RV STEP"]

            RVd = start + step * np.arange(0, sci.shape[1], 1)

            for order in range(self.N_orders):
                if epo_index == 3 and order in [115, 116] and 0:
                    import matplotlib.pyplot as plt

                    plt.plot(RVd, sci[order])
                    plt.show()
                    print(sci[order], qual[order], err[order])

                if all(sci[order] == 0):
                    ccf_rv, ccf_err = np.nan, np.nan
                else:

                    ccf = sci[order]
                    rv = RVd
                    mean = rv[np.argmin(ccf)]
                    sigma = np.sqrt(sum((rv - mean) ** 2.0) / (len(rv)))
                    guess = [-ccf.ptp(), mean, sigma, 0.5 * (ccf[0] + ccf[-1])]

                    if 0:
                        output_rv, output_err = gaussfit(
                            RVd, sci[order], p0=guess, return_errors=True
                        )
                    if 0:
                        output_rv = gaussfit(RVd, sci[order], p0=guess, return_errors=False)
                        output_err = [0, 0]

                    ccf_rv, ccf_err, _ = ccffitRV(rv, ccf, err[order])

                    if epo_index == 0 and order == 115 and 0:
                        import matplotlib.pyplot as plt

                        plt.plot(rv, gauss(rv, guess), color="red", linestyle="--")

                        plt.plot(rv, gauss(rv, output_rv))
                        plt.plot(rv, sci[order])
                        plt.show()
                        ccf_rv = output_rv[1]
                        ccf_err = output_err[1]
                RVs[epo_index].append(ccf_rv)
                ERRORS[epo_index].append(ccf_err)

        return RVs, ERRORS

    def check_header_QC(self, header):
        super().check_header_QC(header)

        fatal_QC_flags = {"HIERARCH ESO QC SCIRED CHECK": 0}

        nonfatal_QC_flags = {
            "HIERARCH ESO QC SCIRED FLUX CORR CHECK": 0,
            "HIERARCH ESO QC SCIRED DRIFT CHECK": 0,
            "HIERARCH ESO QC SCIRED DRIFT FLUX_RATIO CHECK": 0,
            "HIERARCH ESO QC SCIRED DRIFT CHI2 CHECK": 0,
            "HIERARCH ESO INS{} ADC{} RA": 0,  # related with ADC2 problem
            "HIERARCH ESO INS{} ADC{} dec": 0,  # related with ADC2 problem
            "HIERARCH ESO INS{} ADC{} SENS1": 0,  # related with ADC2 problem
            "HIERARCH ESO INS{} ADC{} TEMP": 0,  # related with ADC2 problem
        }

        for flag, bad_value in fatal_QC_flags.items():
            if header[flag] == bad_value:
                msg = f"\tQC flag {flag} is not True"
                logger.critical(msg)
                self.add_to_status(FATAL_KW(msg.replace("\t", "")))

        for flag, bad_value in nonfatal_QC_flags.items():
            if "ADC" in flag:
                found_UT = False
                for UT_KW in ["", "2", "3", "4"]:
                    try:
                        for ADC in [1, 2]:
                            ADC_KW = flag.format(UT_KW, ADC)
                            if header[ADC_KW] == bad_value:
                                msg = f"QC flag {ADC_KW} has a value of {bad_value}"
                                logger.warning(msg)
                                self._status.store_warning(KW_WARNING(msg))
                            found_UT = True
                    except:
                        pass
                if not found_UT:
                    logger.critical(
                        f"Did not find the entry for the following UT related metric: {flag}"
                    )
            else:
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


def gauss(x, p):
    """A Gaussian function with parameters p = [A, x0, σ, offset]."""
    return p[0] * np.exp(-((x - p[1]) ** 2) / (2 * p[2] ** 2)) + p[3]
