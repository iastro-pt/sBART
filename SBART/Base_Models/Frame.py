import datetime
import os
import time
from typing import Any, Dict, Iterable, List, NoReturn, Optional

import numpy as np
from astropy.io import fits
from loguru import logger

from SBART.Components import Spectral_Modelling, Spectrum
from SBART.Masks import Mask
from SBART.utils import custom_exceptions
from SBART.utils.UserConfigs import (
    BooleanValue,
    DefaultValues,
    Positive_Value_Constraint,
    UserParam,
    ValueFromList,
    ValueInInterval,
)
from SBART.utils.custom_exceptions import FrameError
from SBART.utils.ranges import ranges
from SBART.utils.status_codes import (
    HIGH_CONTAMINATION,
    LOADING_EXTERNAL_DATA,
    LOW_SNR,
    MISSING_DATA,
    MISSING_EXTERNAL_DATA,
    MISSING_FILE,
    NO_VALID_ORDERS,
    NON_COMMON_WAVELENGTH,
    QUAL_DATA,
    Flag,
    Status,
)
from SBART.utils.telluric_utilities.compute_overlaps_blocks import check_if_overlap
from SBART.utils.types import RV_measurement
from SBART.utils.units import kilometer_second


class Frame(Spectrum, Spectral_Modelling):
    """
    Base Class for the different :ref:`Instruments<InstrumentsDescription>`, providing a shared interface to spectral data and
    header information.

    This class defines a set of Keywords, consistent for all s-BART supported Instruments, which can be accessed through the
    proper methods. The internal keywords are initialized to a default value, which the Frame will use if the instrument does
    not provide that metric/value. Furthermore, all RV-related metrics are returned as astropy.Quantity objects (or lists of
    such objects). For such cases, one can use :func:`~SBART.utils.units.convert_data` to convert data to different units and/or
    to floats


    The supported list of keywords, and the default initialization values is:

    ========================= ===========================
        Internal KW name       Default intialization
    ========================= ===========================
        BERV                   np.nan * kilometer_second
        MAX_BERV               np.nan * kilometer_second
        previous_SBART_RV      np.nan * kilometer_second
        previous_SBART_RV_ERR  np.nan * kilometer_second
        DRS_RV_ERR             np.nan * kilometer_second
        DRS_RV_ERR             np.nan * kilometer_second
        drift                  np.nan * kilometer_second
        drift_ERR              np.nan * kilometer_second
        relative_humidity      np.nan
        ambient_temperature    np.nan
        airmass                np.nan
        orderwise_SNRs         []
        OBJECT                 None
        BJD                    None
        MJD                    None
        DRS-VERSION            None
        MD5-CHECK              None
        ISO-DATE               None
        CONTRAST               0
        FWHM                   0
        BIS SPAN               0
        RA                     None
        DEC                    None
        SPEC_TYPE              ""
    ========================= ===========================

    **User parameters:**

    ================================ ================ ================ ================ ================
    Parameter name                      Mandatory      Default Value    Valid Values    Comment
    ================================ ================ ================ ================ ================
    bypass_QualCheck                    False              False         boolean         If True, avoid checking the pixel-wise QC checks
    reject_order_percentage             False               0.25         range [0 ,1]    Smallest percentage of "valid" pixels in a "valid" order
    minimum_order_SNR                   False               20           int/float >= 0  If the order's SNR is below this value, reject the order
    spectra_format                      False               S2D          "S2D" or "S1D"  Indicates where we are using S2D or S1D data. Not all instruments support S1D
    ================================ ================ ================ ================ ================

    *Note:* Also check the **User parameters** of the parent classes for further customization options of SBART

    """

    _object_type = "Frame"
    _name = ""

    instrument_properties = {
        "name": "",
        "array_size": (),
        "wavelength_coverage": (),
        "resolution": None,
        "EarthLocation": None,
        "site_pressure": None,  # pressure in hPa
        "is_drift_corrected": None,  # True if the S2D files are already corrected from the drift
    }
    sub_instruments = {}
    # Dict of options and default values for them. Specific for each instrument

    _default_params = DefaultValues(
        bypass_QualCheck=UserParam(False, constraint=BooleanValue),
        #
        reject_order_percentage=UserParam(
            0.25, constraint=ValueInInterval((0, 1), include_edges=True)
        ),
        # If the SNR is smaller, discard the order:
        minimum_order_SNR=UserParam(20, constraint=Positive_Value_Constraint),
        spectra_format=UserParam("S2D", constraint=ValueFromList(("S1D", "S2D"))),
    )

    def __init__(
        self,
        inst_name: str,
        array_size: tuple,
        file_path: str,
        frameID: int,
        KW_map: Dict[str, str],
        available_indicators: tuple,
        user_configs: Optional[Dict[str, Any]] = None,
        reject_subInstruments: Optional[Iterable[str]] = None,
        need_external_data_load: bool = False,
        init_log: bool = True,
    ):
        """
        The Frame object is initialized with the following set of Keywords:

        Parameters
        ----------
        inst_name
            Name of the instrument
        array_size
            Size of the data that will be loaded from disk. Follow the format [Number order, Number pixels]
        file_path
            Path to the file that is going to be opened
        frameID
            Numerical value that represents the frame's ID inside the :class:`~SBART.data_objects.DataClass.DataClass`
        KW_map
            Dictionary where the keys are names of internal Keywords and the values represent the keyword name on the header of the
            .fits files
        available_indicators
            Names of available activity indicators for the instrument
        user_configs
            User configs information to be loaded in the parent class
        reject_subInstruments
            List of subInstruments to completely reject
        need_external_data_load
            True if the instrument must load data from a file that is not the one specified on the "file_path" argument
        init_log
            If True create a log entry with the filename

        """
        self.array_size = array_size
        self.__class__.instrument_properties["name"] = inst_name
        self.__class__.instrument_properties["array_size"] = array_size

        super().__init__(user_configs=user_configs)

        self.frameID = frameID
        self._status = Status()  # BY DEFAULT IT IS A VALID ONE!

        self.spectral_format = self._internal_configs["spectra_format"]

        self.file_path = file_path.split("\n")[0].strip()
        if init_log:
            logger.info("Creating frame from: {}".format(self.file_path))
        self.inst_name = inst_name

        self.sub_instrument = None

        self.available_indicators = available_indicators

        self._KW_map = KW_map

        # stores the information loaded from the header of the S2D files. THis dict will be the default values in case
        # the instrument does not support them!
        # orderwise SNRs OR values with units -> should not be passed inside the KW_map!!!
        self.observation_info = {
            "BERV": np.nan * kilometer_second,
            "previous_SBART_RV": np.nan * kilometer_second,
            "previous_SBART_RV_ERR": np.nan * kilometer_second,
            "DRS_RV": np.nan * kilometer_second,
            "DRS_RV_ERR": np.nan * kilometer_second,
            "drift": np.nan * kilometer_second,
            "drift_ERR": np.nan * kilometer_second,
            "relative_humidity": np.nan,  # for telfit
            "ambient_temperature": np.nan,  # for telfit
            "airmass": np.nan,
            "orderwise_SNRs": [],
            "OBJECT": None,
            "MAX_BERV": np.nan * kilometer_second,
            "BJD": None,
            "MJD": None,
            "DRS-VERSION": None,
            "MD5-CHECK": None,
            "ISO-DATE": None,
            "CONTRAST": 0,
            "FWHM": 0,
            "BIS SPAN": 0,
            "RA": None,
            "DEC": None,
            "SPEC_TYPE": "",
        }

        self.load_header_info()
        # list of lists Each entry will be a pair of Reason: list<[<start, end>]> wavelenghts. When the S2D array is
        # opened, these elements will be used to mask spectral regions

        # TODO: understand why the typing error goes away when we use dict() instead of {}
        self.wavelengths_to_remove: Dict[Flag, List[List[int]]] = {}

        # Store here the wavelength limits for each order (if we want to impose them)!
        self.wavelengths_to_keep = None

        if reject_subInstruments is not None:
            for bad_subInst in reject_subInstruments:
                if self.is_SubInstrument(bad_subInst):
                    self.add_to_status(MISSING_DATA("Rejected entire subInstrument"))
                    logger.warning("Rejecting subInstruments")

        if need_external_data_load:
            self.add_to_status(LOADING_EXTERNAL_DATA)

    def mark_wavelength_region(self, reason: Flag, wavelength_blocks: List[List[int]]) -> None:
        """Add wavelength regions to be removed whenever the S2D file is opened

        Parameters
        ----------
        reason : Flag
            Flag for the removal type
        wavelength_blocks : List[list]
            List with lists of wavelength limits. [[lambda_0, lambda_1], [lambda_2, lambda_3]]
        """
        self.wavelengths_to_remove[reason] = wavelength_blocks

    def select_wavelength_region(self, order, wavelength_blocks):
        if self.wavelengths_to_keep is None:
            self.wavelengths_to_keep = {}
        self.wavelengths_to_keep[order] = wavelength_blocks

    def finalize_data_load(self) -> NoReturn:
        """
        Called for all Instruments, even those that do not need an external data load.
        Checks if the non-fatal Flag "LOADING_EXTERNAL_DATA" exists in the Status. If so, add the fatal Flag
        "MISSING_EXTERNAL_DATA". Otherwise, does nothing

        Returns
        -------

        """
        if self._status.has_flag(LOADING_EXTERNAL_DATA):
            logger.critical("Frame {} did not load the external data that it needed!", self.name)
            self.add_to_status(MISSING_EXTERNAL_DATA)

    def add_to_status(self, new_flag: Flag) -> NoReturn:
        logger.info("Updating Frame ({}) status to {}", self.fname, new_flag)

        super().add_to_status(new_flag=new_flag)

        if not self.is_valid:
            self.close_arrays()

    def _data_access_checks(self) -> NoReturn:
        super()._data_access_checks()
        if not self.is_open:
            self.load_data()

    @property
    def status(self) -> Status:
        """
        Return the Status of the entire Frame
        Returns
        -------

        """
        return self._status

    ###################################
    #          Cleaning data          #
    ###################################

    def build_mask(self, bypass_QualCheck: bool = False, assess_bad_orders: bool = True) -> None:
        """Build a spectral mask based on the S2D data

        Parameters
        ----------
        bypass_QualCheck : bool, optional
            Do not check the QUAL_DATA array for non-zero values, by default False
        """
        logger.debug("Creating spectral mask")
        self.spectral_mask = Mask(initial_mask=np.zeros(self.array_size, dtype=np.uint16))
        if not bypass_QualCheck:
            zero_indexes = np.where(self.qual_data != 0)
            self.spectral_mask.add_indexes_to_mask(zero_indexes, QUAL_DATA)

        self.spectral_mask.add_indexes_to_mask(np.where(np.isnan(self.spectra)), MISSING_DATA)
        self.spectral_mask.add_indexes_to_mask(np.where(self.spectra == 0), MISSING_DATA)

        self.spectral_mask.add_indexes_to_mask(np.where(np.isnan(self.uncertainties)), MISSING_DATA)

        order_map = {
            i: (np.min(self.wavelengths[i]), np.max(self.wavelengths[i]))
            for i in range(self.N_orders)
        }
        removal_reasons = [i.name for i in self.wavelengths_to_remove.keys()]
        N_point_removed = []
        time_took = []

        logger.debug("Cleaning wavelength regions from {}".format(removal_reasons))

        for removal_reason, wavelengths in self.wavelengths_to_remove.items():
            start_time = time.time()
            nrem = len(wavelengths)

            N_point_removed.append(nrem)
            for wave_pair in wavelengths:

                for order in range(self.N_orders):
                    if check_if_overlap(wave_pair, order_map[order]):
                        indexes = np.where(
                            np.logical_and(
                                self.wavelengths[order] >= wave_pair[0],
                                self.wavelengths[order] <= wave_pair[1],
                            )
                        )
                        self.spectral_mask.add_indexes_to_mask_order(order, indexes, removal_reason)
            time_took.append(time.time() - start_time)
        logger.debug(
            "Removed {} regions ({})", sum(N_point_removed), " + ".join(map(str, N_point_removed))
        )
        logger.debug("Ensuring that we have increasing wavelengths")

        diffs = np.where(np.diff(self.wavelengths, axis=1) < 0)
        if diffs[0].size > 0:
            logger.warning("Found non-increasing wavelengths on {}", self.name)
            self.spectral_mask.add_indexes_to_mask(diffs, QUAL_DATA("Non-increasing wavelengths"))

        logger.debug("Took {} seconds ({})", sum(time_took), " + ".join(map(str, time_took)))

        if assess_bad_orders:
            self.assess_bad_orders()

        if self.wavelengths_to_keep is not None:
            logger.info("Provided desired wavelength region. Rejecting regions outside it")
            for order in range(self.N_orders):
                good_regions = self.wavelengths_to_keep[order]
                if len(good_regions) == 0:  # TODO: ensure that the order is also rejected
                    continue

                inds = np.zeros(self.wavelengths[order].size, dtype=bool)
                for region in good_regions:
                    wavelengths_to_keep = np.where(
                        np.logical_and(
                            self.wavelengths[order] >= region[0],
                            self.wavelengths[order] <= region[1],
                        )
                    )
                    inds[wavelengths_to_keep] = True
                self.spectral_mask.add_indexes_to_mask_order(
                    order, np.where(~inds), NON_COMMON_WAVELENGTH
                )

    def assess_bad_orders(self) -> None:
        """Evaluate the masked points to find those that can always be discarded!"""
        # True in the points to mask
        logger.debug("Rejecting spectral orders")
        entire_mask = self.spectral_mask.get_custom_mask()

        for order, value in enumerate(entire_mask):
            # See if the total amounf of rejected points is larger than
            # 1 - reject_order-percentage of the entire order
            perc = self._internal_configs["reject_order_percentage"]
            if np.sum(value) > (1 - perc) * self.pixels_per_order:
                self._OrderStatus.add_flag_to_order(
                    order, HIGH_CONTAMINATION("Rejection threshold met in order")
                )

        if len(self.bad_orders) > 0:
            logger.info(
                "Frame {} rejected {} orders due for having less than {} valid pixels: {}",
                self.frameID,
                len(self.bad_orders),
                self._internal_configs["reject_order_percentage"],
                ranges(list(self.bad_orders)),
            )

        if self.is_S2D:  # we don't have the SNR for the S1D file!
            bad_SNR = []
            SNRS = self.get_KW_value("orderwise_SNRs")
            for order in range(self.N_orders):
                if SNRS[order] < self._internal_configs["minimum_order_SNR"]:
                    self._OrderStatus.add_flag_to_order(
                        order, LOW_SNR("Minimum SNR not met in order")
                    )
                    bad_SNR.append(order)

            if len(bad_SNR) > 0:
                logger.info(
                    "Frame {} rejected {} orders for having SNR smaller than {}: {}",
                    self.frameID,
                    len(bad_SNR),
                    self._internal_configs["minimum_order_SNR"],
                    ranges(bad_SNR),
                )

        if len(self.bad_orders) == self.N_orders:
            logger.critical("All spectral orders of Frame {} have been rejected", self)
            self.add_to_status(NO_VALID_ORDERS(" Rejected all spectral orders"))
        elif len(self.bad_orders) > 0.8 * self.N_orders:
            logger.warning("Frame {} is rejecting more than 80% of the spectral orders", self)

    ####################################
    #      Sanity Checks               #
    ####################################
    def check_header_QC(self, header: fits.header.Header):
        """Check if the header keywords are in accordance with their default value. Each instrument
        should do this check on its own
        """
        logger.debug("Validating header KeyWords")
        pass

    def find_instrument_type(self):

        obs_date = self.get_KW_value("ISO-DATE")
        obs_date = "-".join(obs_date.split("T")).split(":")[0]
        obs_date = datetime.datetime.strptime(obs_date, r"%Y-%m-%d-%H")

        for key, threshold in self.__class__.sub_instruments.items():
            # If it is not higher tha  the threshold, then it beleongs in this "interval"
            if not obs_date > threshold:
                self.sub_instrument = key
                break
        logger.debug("Frame determined to be from {}", self.sub_instrument)

    #####################################
    #      Handle data management      #
    ####################################

    def load_data(self) -> None:
        if self.is_S1D:
            self.load_S1D_data()
        elif self.is_S2D:
            self.load_S2D_data()

        if not self.is_valid:
            raise FrameError("Frame is no longer valid")

    def load_S1D_data(self):
        """To be overriden by the children classes"""
        logger.debug("Opening the S1D arrays from {}", self.fname)
        if not self.is_valid:
            raise FrameError
        self._spectrum_has_data_on_memory = True

    def load_S2D_data(self):
        """To be overriden by the children classes"""
        logger.debug("Opening the S2D arrays from {}", self.fname)
        if not self.is_valid:
            raise FrameError
        self._spectrum_has_data_on_memory = True

    def load_instrument_specific_KWs(self, header):
        """Load the KW values that can not be loaded in a general fashion (e.g. needs UT number or units)
        To be overriden by the different instruments
        Parameters
        ----------
        header : [type]
            [description]
        """
        return

    def store_previous_SBART_result(self, RV: RV_measurement, RV_err: RV_measurement) -> NoReturn:
        """
        Store, from the outside, RV and uncertainty from a previous SBART application

        Parameters
        ----------
        RV
        RV_err

        Returns
        -------

        """
        if not isinstance(RV, RV_measurement) or not isinstance(RV_err, RV_measurement):
            raise custom_exceptions.InvalidConfiguration(
                "The previous SBART RVs must be astropy quantities!"
            )

        self.observation_info["previous_SBART_RV"] = RV
        self.observation_info["previous_SBART_RV_ERR"] = RV_err

    def load_header_info(self) -> None:
        """Open the header of the fits file and load the necessary keywords. To be overriden by the children classes"""
        try:
            hdu = fits.getheader(self.file_path)
        except FileNotFoundError:
            msg = f"File <{self.file_path}> does not exist"
            self.add_to_status(MISSING_FILE(msg))
            logger.critical(msg)
            return

        for internal_KW, S2D_KW in self._KW_map.items():
            self.observation_info[internal_KW] = hdu[S2D_KW]

        self.load_instrument_specific_KWs(hdu)
        self.check_header_QC(hdu)
        self.find_instrument_type()

    ####################################
    #       Access data
    ####################################

    def get_KW_value(self, KW: str):
        return self.observation_info[KW]

    ####################################
    #       properties of the Frames
    ####################################

    @property
    def is_S1D(self):
        return self.spectral_format == "S1D"

    @property
    def is_S2D(self):
        return self.spectral_format == "S2D"

    @property
    def has_warnings(self):
        return self._status.has_warnings

    def is_Instrument(self, Instrument: str) -> bool:
        return self.inst_name == Instrument

    def is_SubInstrument(self, sub_instrument: str) -> bool:
        """Check if the current instrument is from the given time_block (e.g ESPRESSO18/ESPRESSO19)

        Parameters
        ----------
        sub_instrument : str
            Name of the time block that is going to be checked

        Returns
        -------
        [type]
            Results from the comparison
        """
        return self.sub_instrument == sub_instrument

    @property
    def previous_RV_measurements(self):
        return self.get_KW_value("DRS_RV"), self.get_KW_value("DRS_RV_ERR")

    @property
    def bare_fname(self) -> str:
        """
        Returns the file name without the _S2D (and similar) parts

        The children classes must overload this property. Otherwise, returns the full filename
        Returns
        -------

        """

        return self.fname

    @property
    def fname(self):
        return os.path.basename(self.file_path)

    @property
    def min_pixel_in_order(self) -> int:
        return self._internal_configs["reject_order_percentage"] * self.pixels_per_order

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"Frame of {self.inst_name} : {self.sub_instrument} data ({self.get_KW_value('ISO-DATE')}; ID = {self.frameID})"
