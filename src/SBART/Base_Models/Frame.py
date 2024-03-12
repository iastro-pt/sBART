import datetime
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, NoReturn, Optional, Tuple

import numpy as np
from astropy.io import fits
from loguru import logger

from SBART.Components import Spectral_Modelling, Spectrum, Spectral_Normalization
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


class Frame(Spectrum, Spectral_Modelling, Spectral_Normalization):
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
    spectra_format                      False               S2D          "S2D"            Indicates where we are using S2D or S1D data. Not all instruments support S1D
    use_air_wavelengths                 False             False         boolean         Use air wavelengths, instead of vacuum. Only used in S1D files!
    open_without_BervCorr               False           False           boolean             If True, remove any BERV correction
    ================================ ================ ================ ================ ================

    .. note::
       This class also uses the User parameters defined by the :class:`~SBART.Components.Modelling.Spectral_Modelling`
    class

    .. note::
        Also check the **User parameters** of the parent classes for further customization options of SBART

    """

    _object_type = "Frame"
    _name = ""

    sub_instruments = {}
    # Dict of options and default values for them. Specific for each instrument

    _default_params = DefaultValues(
        bypass_QualCheck=UserParam(False, constraint=BooleanValue),
        open_without_BervCorr=UserParam(
            False,
            constraint=BooleanValue,
            description="Ensure that the Frame is not BERV corrected, independently of correction being applied or not in the official pipeline",
        ),
        apply_FluxCorr=UserParam(False, constraint=ValueFromList((False,))),
        use_air_wavelengths=UserParam(
            False,
            constraint=BooleanValue,
            description="Use air wavelengths, instead of the vacuum ones",
        ),
        apply_FluxBalance_Norm=UserParam(False, constraint=ValueFromList((False,))),
        reject_order_percentage=UserParam(
            0.25, constraint=ValueInInterval((0, 1), include_edges=True)
        ),
        # If the SNR is smaller, discard the order:
        minimum_order_SNR=UserParam(
            5,
            constraint=Positive_Value_Constraint,
            description="SNR threshold under which the spectral order is rejected",
        ),
        bypass_ST_designation=UserParam(
            default_value=None, constraint=ValueFromList((None, "S2D", "S1D"))
        ),
    )

    def __init__(
        self,
        inst_name: str,
        array_size: Dict[str, tuple],
        file_path: Path,
        frameID: int,
        KW_map: Dict[str, str],
        available_indicators: tuple,
        user_configs: Optional[Dict[str, Any]] = None,
        reject_subInstruments: Optional[Iterable[str]] = None,
        need_external_data_load: bool = False,
        init_log: bool = True,
        quiet_user_params: bool = True,
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
        quiet_user_params
            If True, there are no logs for the generation of the user parameters of each Frame
        """
        self.instrument_properties = {
            "name": inst_name,
            "array_sizes": array_size,
            "array_size": None,
            "wavelength_coverage": (),
            "resolution": None,
            "EarthLocation": None,
            "site_pressure": None,  # pressure in hPa
            "is_drift_corrected": None,  # True if the S2D files are already corrected from the drift
        }

        self.frameID = frameID
        self._status = Status()  # BY DEFAULT IT IS A VALID ONE!

        if not isinstance(file_path, (str, Path)):
            raise custom_exceptions.InvalidConfiguration("Invalid path!")

        if not isinstance(file_path, Path):
            file_path = Path(file_path)

        self.file_path: Path = file_path
        if init_log:
            logger.info("Creating frame from: {}".format(self.file_path))
        self.inst_name = inst_name

        self.sub_instrument = None

        self.available_indicators = available_indicators

        self._KW_map = KW_map
        if "UseMolecfit" in user_configs:
            self.spectral_format = "S1D"
        elif "bypass_ST_designation" in user_configs:
            self.spectral_format = user_configs["bypass_ST_designation"]
        else:
            self.spectral_format = self.get_spectral_type()
        self.instrument_properties["array_size"] = self.instrument_properties["array_sizes"][
            self.spectral_format
        ]
        self.array_size = self.instrument_properties["array_size"]
        super().__init__(user_configs=user_configs, quiet_user_params=quiet_user_params)

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
            "EXPTIME": 0,
            "RA": None,
            "DEC": None,
            "SPEC_TYPE": "",
            "DET_BINX": None,
            "DET_BINY": None,
            "seeing": None,
        }

        # Used to allow to reject a wavelength region from one order and keep any overlap that might exist on others
        self._orderwise_wavelength_rejection: Optional[Dict[int, List]] = None

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

    def get_spectral_type(self) -> str:
        """
        Check the filename to see if we are using an S1D or S2D file
        Returns
        -------

        """
        name_lowercase = self.file_path.stem.lower()
        if "s2d" in name_lowercase or "e2ds" in name_lowercase:
            return "S2D"
        elif "s1d" in name_lowercase:
            return "S1D"
        else:
            raise custom_exceptions.InternalError(
                f"{self.name} can't recognize the file that it received!"
            )

    def copy_into_S2D(self, new_S2D_size: Optional[Tuple[int, int]] = None):
        """
        Return a new object which contains the S1D that that has been converted into a S2D

        Parameters
        -----------
        new_S2D_size: Optional[Tuple[int, int]]
            Size of the new S2D size, should be a tuple with two elements: (number orders, pixel in order).
            If it is None, then uses the standard size of S2D files of this instrument. **Default:** None

        Returns
        -------

        """
        if self.is_S2D:
            raise custom_exceptions.InvalidConfiguration("Can't transform S2D file into S2D file")
        logger.warning("Creating a copy of a S1D Frame for transformation into S2D")

        og_shape = (
            self.instrument_properties["array_sizes"]["S2D"]
            if new_S2D_size is None
            else new_S2D_size
        )

        reconstructed_S2D = np.zeros(og_shape)
        reconstructed_wavelengths = np.zeros(og_shape)
        reconstructed_uncertainties = np.zeros(og_shape)

        order_number = 0
        order_size = reconstructed_wavelengths[0].size
        to_break = False
        wavelengths, flux, uncertainties, _ = self.get_data_from_full_spectrum()
        wavelengths = wavelengths[0]
        flux = flux[0]
        uncertainties = uncertainties[0]

        while not to_break:
            start_order = order_size * order_number
            end_order = start_order + order_size
            if end_order >= wavelengths.size:
                to_break = True
                end_order = wavelengths.size

            slice_size = end_order - start_order
            reconstructed_wavelengths[order_number] = np.pad(
                wavelengths[start_order:end_order], (0, order_size - slice_size), constant_values=0
            )
            reconstructed_S2D[order_number] = np.pad(
                flux[start_order:end_order], (0, order_size - slice_size), constant_values=0
            )
            reconstructed_uncertainties[order_number] = np.pad(
                uncertainties[start_order:end_order],
                (0, order_size - slice_size),
                constant_values=0,
            )
            order_number += 1

        # The "new" orders that don't have any information will have a flux of zero. Thus, they will be deemed to
        # be invalid during the mask creation process (that is re-launched after this routine is done)

        # Ensure that we don't lose information due to the SNR cut
        user_configs = self._internal_configs._user_configs
        user_configs["minimum_order_SNR"] = 0

        inst_properties = self.instrument_properties["array_sizes"]
        if new_S2D_size is not None:
            inst_properties["S2D"] = new_S2D_size

        new_frame = Frame(
            inst_name=self.inst_name,
            array_size=inst_properties,
            file_path=self.file_path,
            frameID=self.frameID,
            KW_map=self._KW_map,
            available_indicators=self.available_indicators,
            user_configs=self._internal_configs._user_configs,
        )
        new_frame.wavelengths = reconstructed_wavelengths
        new_frame.spectra = reconstructed_S2D
        new_frame.uncertainties = reconstructed_uncertainties
        for key in ["observation_info", "instrument_properties"]:
            setattr(new_frame, key, getattr(self, key))

        new_frame._spectrum_has_data_on_memory = True  # to avoid new data loads!
        new_frame._never_close = True  # ensure that we don't lose the transformation
        new_frame.spectral_format = "S2D"
        new_frame.instrument_properties["array_size"] = new_S2D_size
        new_frame.array_size = new_S2D_size
        new_frame.sub_instrument = self.sub_instrument
        new_frame.is_blaze_corrected = self.is_blaze_corrected
        new_frame.observation_info["orderwise_SNRs"] = [1 for _ in range(new_S2D_size[0])]
        new_frame.regenerate_order_status()
        return new_frame

    def import_KW_from_outside(self, KW, value, optional: bool):
        """
        Allow to manually override frame parameters from the outside
        """
        if KW not in self.observation_info:
            logger.critical(
                "Keyword <{}> is not supported by the Frames. Couldn't load it from the outside",
                KW,
            )

        if not np.isfinite(value):
            if not optional:
                logger.critical(
                    "Loaded mandatory keyword <{}> with a non-finite value for frame {}",
                    KW,
                    self.fname,
                )
                raise FrameError
            logger.critical(
                "Loaded keyword <{}> has a non-finite value for frame {}",
                KW,
                self.fname,
            )
        self.observation_info[KW] = value

    def reject_wavelength_region_from_order(self, order, region):
        """
        Flag a wavelength region from  an order to be marked as invalid during the creation of the stellar mask
        """
        if not isinstance(region, (Iterable,)):
            raise custom_exceptions.InvalidConfiguration(
                "The rejection region must be a list of lists"
            )

        if self._orderwise_wavelength_rejection is None:
            self._orderwise_wavelength_rejection = {}
        self._orderwise_wavelength_rejection[order] = region

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

    def finalize_data_load(self, bad_flag: Optional[Flag] = None) -> NoReturn:
        """
        Called for all Instruments, even those that do not need an external data load.
        Checks if the non-fatal Flag "LOADING_EXTERNAL_DATA" exists in the Status. If so, add the fatal Flag
        "MISSING_EXTERNAL_DATA". Otherwise, does nothing

        Returns
        -------

        """
        if self._status.has_flag(LOADING_EXTERNAL_DATA):
            logger.critical(f"Frame {self.name} did not load the external data that it needed!")

            self._status.delete_flag(LOADING_EXTERNAL_DATA)
            if bad_flag is None:
                self.add_to_status(MISSING_EXTERNAL_DATA)
            else:
                self.add_to_status(bad_flag)

    def finalized_external_data_load(self):
        """Tuns an invalid CARMENES::KOBE frame into a valid one (assuming that the only problem is missing the SHAQ loads)

        If the status of the frame is different than MISSING_SHAQ_DATA (meaning that something went bad with the data load)
        Returns
        -------
        NoReturn
        """
        logger.info("Finalizing external data loading")
        if not self.is_valid:
            logger.warning("Frame has already been rejected.")
        else:
            logger.info("{} is a valid observation. Finishing external data load", self)
            self._status.delete_flag(LOADING_EXTERNAL_DATA)

    def finalized_external_data_load(self):
        """Tuns an invalid CARMENES::KOBE frame into a valid one (assuming that the only problem is missing the SHAQ loads)

        If the status of the frame is different than MISSING_SHAQ_DATA (meaning that something went bad with the data load)
        Returns
        -------
        NoReturn
        """
        logger.info("Finalizing external data loading")
        if not self.is_valid:
            logger.warning("Frame has already been rejected.")
        else:
            logger.info("{} is a valid observation. Finishing external data load", self)
            self._status.delete_flag(LOADING_EXTERNAL_DATA)

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
        self.spectral_mask = Mask(
            initial_mask=np.zeros(self.instrument_properties["array_size"], dtype=np.uint16)
        )
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
        if self._orderwise_wavelength_rejection is not None:
            logger.info("Rejecting spectral chunks from individual orders")
            for order, region in self._orderwise_wavelength_rejection.items():
                for subregion in region:
                    indexes = np.where(
                        np.logical_and(
                            self.wavelengths[order] >= subregion[0],
                            self.wavelengths[order] <= subregion[1],
                        )
                    )
                    self.spectral_mask.add_indexes_to_mask_order(
                        order, indexes, NON_COMMON_WAVELENGTH
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

        This function will check for two things:
        1. Fatal keywords - will mark the Frame as invalid
        2. Warning Keywords - the frame is still valid, but it has a warning issued in the logs

        If any of those conditions is met, make sure that the flags meet the following naming conditions
        (so that we can filter by them later on):

        For fatal flags
        ```
        msg = f"QC flag {flag} has taken the bad value of {bad_value}"
        self.add_to_status(FATAL_KW(msg))
        ```

        For warnings:
        ```
        msg = f"QC flag {flag} meets the bad value"
        self._status.store_warning(KW_WARNING(msg))
        ```
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
    def get_S1D_name(self) -> str:
        """
        Build the S1D name that should be associated with this Frame.
        If it is already a S1D, returns the actual name.
        If it is not, remove "blaze" from the filename and replaces "S2D" with "S1D"

        Returns
        -------

        """
        # TODO: this will not work for non-ESPRESSO files

        if self.is_S1D:
            return self.fname
        name = self.fname
        return name.replace("BLAZE_", "").replace("S2D", "S1D")

    def load_data(self) -> None:
        if self.is_S1D:
            self.load_S1D_data()
        elif self.is_S2D:
            self.load_S2D_data()

        if not self.is_valid:
            raise FrameError("Frame is no longer valid")

        BERV_value = self.get_KW_value("BERV")
        if not self._internal_configs["open_without_BervCorr"]:
            self.apply_BERV_correction(BERV_value)
        else:
            logger.warning(f"Opening {self.name} without the BERV correction")
            self.remove_BERV_correction(BERV_value)

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

    @property
    def spectrum_information(self):
        return {
            **{
                "subInstrument": self.sub_instrument,
                "filename": self.bare_fname,
                "is_S2D": self.is_S2D,
                "is_S1D": self.is_S1D,
            },
            **super().spectrum_information,
        }

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"Frame of {self.inst_name} : {self.sub_instrument} data ({self.get_KW_value('ISO-DATE')}; ID = {self.frameID})"
