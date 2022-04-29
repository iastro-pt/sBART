import json
from pathlib import Path
from typing import Iterable, List, NoReturn, Optional, Type, Union

import numpy as np
from loguru import logger
from tabletexifier import Table

from SBART import __version__
from SBART.Base_Models.BASE import BASE
from SBART.Base_Models.Frame import Frame
from SBART.Quality_Control.activity_indicators import Indicators
from SBART.data_objects.MetaData import MetaData
from SBART.data_objects.Target import Target
from SBART.template_creation.StellarModel import StellarModel
from SBART.template_creation.TelluricModel import TelluricModel
from SBART.utils.custom_exceptions import FrameError, InvalidConfiguration, NoDataError
from SBART.utils.paths_tools.Load_RVoutputs import find_RVoutputs
from SBART.utils.shift_spectra import apply_RVshift
from SBART.utils.spectral_conditions import ConditionModel as CondModel
from SBART.utils.status_codes import (  # for entire frame; for individual pixels
    ACTIVITY_LINE,
    TELLURIC,
    Status,
)
from SBART.utils.types import UI_PATH
from SBART.utils.units import kilometer_second


class DataClass(BASE):
    """
    The user-facing object that handles the loading and data access to the spectral data, independently of the instrument.

    .. note::

         To use this class in SBART RV extraction routines, we place it in shared memory, allowing all processes to easily access
          it. This is done with a `proxyObject <https://docs.python.org/3.8/library/multiprocessing.html>`_.

          SBART already provides a DataClass object that is wrapped by a proxyObject:

        .. code-block:: python

            from SBART.data_objects import DataClassManager
            manager = DataClassManager()
            manager.start()
            data_object = manager.DataClass(*args, **kwargs)

        This *data_object* has all the functions that the DataClass object implements!
    """

    def __init__(
        self,
        path: Iterable[UI_PATH],
        instrument: Type[Frame],
        instrument_options: dict,
        reject_subInstruments: Optional[Iterable[str]] = None,
        target_name: str = None,
    ):
        super().__init__()

        self._inst_type = instrument
        self.input_file = path

        # Hold all of the frames
        self.observations = []

        self.metaData = MetaData()

        if reject_subInstruments is not None:
            logger.warning("Rejecting subInstruments: {}".format(reject_subInstruments))

        OBS_list = []
        if isinstance(path, (str, Path)):
            logger.info("DataClass loading data from {}", self.input_file)
            with open(path) as input_file:
                for line in input_file:
                    OBS_list.append(line)

        elif isinstance(path, (list, tuple, np.ndarray)):
            logger.info("DataClass opening {} files from a list/tuple", len(path))
            OBS_list = path
        else:
            raise TypeError()

        for frameID, filepath in enumerate(OBS_list):
            self.observations.append(
                self._inst_type(
                    filepath.split("\n")[0],
                    instrument_options,
                    reject_subInstruments,
                    frameID=frameID,
                )
            )

        N_files = len(self.observations)
        logger.debug("Selected {} observations from disk", N_files)
        if N_files == 0:
            raise InvalidConfiguration("Input has zero files")

        self.frameID_map = {subInst: [] for subInst in self._inst_type.sub_instruments}
        self._build_frameID_map()

        self._collect_MetaData()

        # TODO: find a better way of doing this!
        self.Target = Target(
            self.collect_KW_observations(
                "OBJECT", self._inst_type.sub_instruments, include_invalid=False
            ),
            original_name=target_name,
        )

        self._validate_loaded_observations()
        self.show_loadedData_table()

        self._applied_telluric_removal = False

        self.StellarModel = None

        for frame in self.observations:
            frame.finalize_data_load()

    ########################
    #    Operate on Data   #
    ########################

    def load_previous_SBART_results(
        self, LoadingPath_previousRun: UI_PATH, use_merged_cube: bool = False
    ):
        """
        Load the results from a previous application of SBART, storing the RV and uncertainty inside the corresponding
        Frame object

        Parameters
        ----------
        LoadingPath_previousRun
        use_merged_cube

        Returns
        -------

        Raises
        -------
        SBART.utils.custom_exceptions.InvalidConfiguration
            If the loaded data uses a different frameID scheme than the one currently in use or if we couldn't find the
            RV outputs on disk
        """

        logger.info("Loading RVs from previous SBART run as the starting-RVs")
        try:
            RV_holder = find_RVoutputs(LoadingPath_previousRun)
        except FileNotFoundError:
            raise InvalidConfiguration("RV outputs couldn't be found on the provided path")

        for frameID, frame in enumerate(self.observations):

            cube = RV_holder.get_RV_cube(frame.sub_instrument, merged=use_merged_cube)

            previous_filename = cube.cached_info["bare_filename"][frameID]
            if previous_filename != frame.bare_fname:
                raise InvalidConfiguration("Loading RVs from cube with different frameID layouts")

    def remove_activity_lines(self, lines: Indicators) -> None:
        """Find the wavelength windows in which activity-related lines are expected to appear,
        for all valid observations

        Parameters
        ----------
        lines : Indicators
            Object with the wavelength blocks "original" positions.

        # TODO: also allow to use the previous SBART RVs for this!!!!
        """
        logger.info("Computing activity windows for each RV measurements")
        for frameID in self.get_valid_frameIDS():
            frame = self.get_frame_by_ID(frameID)
            blocked_regions = lines.compute_forbidden_wavelengths(frame.previous_RV_measurements[0])
            frame.mark_wavelength_region(reason=ACTIVITY_LINE, wavelength_blocks=blocked_regions)

    def remove_telluric_features(self, Telluric_Template: TelluricModel) -> None:

        for subInstrument in self.get_subInstruments_with_valid_frames():
            valid_frameIDS = self.get_frameIDs_from_subInst(subInstrument)

            subInst_template = Telluric_Template.request_data(subInstrument)

            for frameID in valid_frameIDS:
                if subInst_template.for_feature_removal:
                    self.get_frame_by_ID(frameID).mark_wavelength_region(
                        reason=TELLURIC,
                        wavelength_blocks=subInst_template.contaminated_regions,
                    )
                elif subInst_template.for_feature_correction:
                    (
                        wavelengths,
                        model,
                        model_uncertainty,
                    ) = subInst_template.fit_telluric_model_to_frame(self.get_frame_by_ID(frameID))

                    self.get_frame_by_ID(frameID).apply_telluric_correction(
                        wavelengths=wavelengths, model=model, model_uncertainty=model_uncertainty
                    )

        self._applied_telluric_removal = True

    def ingest_StellarModel(self, Stellar_Model: StellarModel) -> None:
        logger.debug("Ingesting StellarModel into the DataClass")
        if self.StellarModel is not None:
            logger.warning(
                "Stellar template has already been ingested. Switching old template by the new one"
            )

        logger.warning("Currently there is no check for same target in S2D data and template!")
        self.StellarModel = Stellar_Model

    def select_common_wavelengths(self, wave_analysis_path, subInst):
        raise NotImplementedError
        with open(wave_analysis_path) as file:
            loaded_json = json.load(file)

        common_waves = {}
        for key, value in loaded_json.items():
            common_waves[int(key)] = value
        for frameID in self.get_frameIDs_from_subInst(subInst):
            frame = self.get_frame_by_ID(frameID)
            obs_rv = frame.get_KW_value("previousRV").to(kilometer_second).value

            for order, elements in common_waves.items():
                good_regions = [apply_RVshift(elem, obs_rv) for elem in elements]
                frame.select_wavelength_region(order, good_regions)

    def reject_observations(self, conditions: CondModel) -> None:
        """Apply the conditions to evaluate if the VALID frame meets the
        specified conditions or not!

        Parameters
        ----------
        conditions:
            Conditions set by the user
        """

        if conditions is None:
            return

        blocked_frames = 0

        for frameID in self.get_valid_frameIDS():
            frame = self.get_frame_by_ID(frameID)
            keep, flags = conditions.evaluate(frame)

            if not keep:
                for flag in flags:
                    frame.add_to_status(flag)
                blocked_frames += 1

        logger.info(
            "User conditions removed {} / {} frames",
            blocked_frames,
            len(self.observations),
        )

        logger.info("Updated observation on disk!")
        self.show_loadedData_table()

        if len(self.get_valid_frameIDS()) == 0:
            raise NoDataError("All observations have been blocked")

    ########################
    #    Sanity Control    #
    ########################
    def _validate_loaded_observations(self) -> None:
        """Check if the same DRS version is used across all loaded files!"""

        for equal_KW in ["DRS-VERSION", "SPEC_TYPE"]:
            collected_KW = set(
                self.collect_KW_observations(equal_KW, self._inst_type.sub_instruments)
            )

            if len(collected_KW) != 1:
                logger.warning("Different values for the KW value <{}>: {}", equal_KW, collected_KW)
            else:
                logger.info("Loaded data from KW : {}", equal_KW, collected_KW)

    def _collect_MetaData(self) -> None:
        """Collect information from the individual (valid) observations to store inside the MetaData object"""

        logger.info("Collecting MetaData from the observations")
        for subInst in self.get_subInstruments_with_valid_frames():
            meta_search = {
                "DRS-VERSION": [],
                "MD5-CHECK": [],
            }

            for key in meta_search:
                meta_search[key] = self.collect_KW_observations(key, [subInst])

            count = len(self.get_frameIDs_from_subInst(subInst))

            for key, items in meta_search.items():
                self.metaData.add_info(key, items, subInst)

            self.metaData.add_info("N_OBS", [count], subInst)
            self.metaData.add_info("SBART_VERSION", [__version__], subInst)

    ########################
    #    access data       #
    ########################

    def get_Target(self) -> Target:
        return self.Target

    def load_frame_by_ID(self, frameID: int) -> int:
        frame = self.get_frame_by_ID(frameID)
        frame.load_data()
        return 1

    def load_all_from_subInst(self, subInst: str) -> int:
        """Load all valid frames from a given subInstrument

        Parameters
        ----------
        subInst : str
            subInstrument name

        Returns
        -------
        int
            [description]
        """
        logger.debug("Opening all frames from {}", subInst)
        IDs_from_subInst = self.get_frameIDs_from_subInst(
            subInstrument=subInst, include_invalid=False
        )
        for frameID in IDs_from_subInst:
            try:
                self.load_frame_by_ID(frameID)
            except FrameError:
                logger.warning("Trying to open invalid frame")
        return 1

    def close_frame_by_ID(self, frameID: int) -> int:
        frame = self.get_frame_by_ID(frameID)
        frame.close_arrays()
        return 1

    def min_pixel_in_order(self) -> int:
        return self.observations[0].min_pixel_in_order

    def get_subInst_from_frameID(self, frameID: int) -> str:
        return self.get_frame_by_ID(frameID).sub_instrument

    def get_frame_OBS_order(self, frameID: int, order: int, include_invalid: bool = False):
        """
        Request the data from one spectral order.

        Parameters
        ----------
        frameID
            FrameID of the observation
        order
            NUmber of order
        include_invalid
            Orders can also be flagged as "invalid". If False, the frame will raise an error when requesting the data

        Returns
        -------

        """
        frame = self.get_frame_by_ID(frameID)
        return frame.get_data_from_spectral_order(order, include_invalid)

    def get_frame_arrays_by_ID(self, frameID: int):
        """
        Access data from the entire spectral range (i.e. all orders come as a matrix)
        Parameters
        ----------
        frameID

        Returns
        -------

        """
        frame = self.get_frame_by_ID(frameID)
        return frame.get_data_from_full_spectrum()

    def get_KW_from_frameID(self, KW: str, frameID: int):
        frame = self.get_frame_by_ID(frameID)
        return frame.get_KW_value(KW)

    def get_filename_from_frameID(self, frameID: int, full_path: bool = False) -> str:
        frame = self.get_frame_by_ID(frameID)
        if full_path:
            return frame.file_path
        return frame.fname

    def get_status_by_frameID(self, frameID: int) -> Status:
        frame = self.get_frame_by_ID(frameID)
        return frame.status

    def collect_KW_observations(
        self,
        KW: str,
        subInstruments: Union[tuple, list],
        include_invalid: bool = False,
        conditions: CondModel = None,
    ) -> list:
        """
        Parse through the loaded observations and retrieve a specific KW from
        all of them. There is no sort of the files. The output will follow the
        order of the files loaded in memory!

        Parameters
        ----------
        KW : str
            KW from the Frame.observation_info dictionary
        subInstruments : Union[tuple, list]
            List of the subInstruments for which we want to retrieve the KW. If
            there are multiple entries, the output will follow the order of this
            list
        include_invalid : bool, optional
            If True, also retrieve the KWs of Frames deemed to not be valid. By
            default False

        Returns
        -------
        list
            List of the KW
        """
        output = []

        for subInst in subInstruments:
            try:
                available_frameIDs = self.get_frameIDs_from_subInst(
                    subInst, include_invalid=include_invalid
                )
            except NoDataError:
                continue

            for frameID in available_frameIDs:
                if conditions is not None:
                    keep, flags = conditions.evaluate(self.get_frame_by_ID(frameID))
                    if not keep:
                        output.append(None)
                        continue
                output.append(self.get_frame_by_ID(frameID).get_KW_value(KW))

        return output

    def collect_RV_information(
        self,
        KW,
        subInst: str,
        frameIDs=None,
        include_invalid: bool = False,
        units=None,
        as_value: bool = True,
    ) -> list:
        """Return the RV measurements (or BERV) from the observations of a given sub-Instrument

        Parameters
        ----------
        KW : KW to request
            previousRV, previousRV_ERR or BERV
        subInst : str
            Name of the sub-Instrument
        frameIDs : None | list
            If not None, retrive only for the frameIDs inside the list. If they are not from subInst, an exception will be raised
        include_invalid : bool, optional
            Include invalid observations, by default False
        units : [type], optional
            units to convert the measurements, by default None
        as_value : bool, optional
            provide a value instead of "units", by default True

        Returns
        -------
        list
            [description]

        Raises
        ------
        InvalidConfiguration
            [description]
        """
        if KW not in ["BERV", "DRS_RV", "DRS_RV_ERR", "previous_SBART_RV", "previous_SBART_RV_ERR"]:
            msg = "Asking for a non-RV KW: {}".format(KW)
            logger.critical(msg)
            raise InvalidConfiguration(msg)

        if frameIDs is None:
            values = self.collect_KW_observations(KW, [subInst], include_invalid=include_invalid)
        else:
            values = []
            for frameID in frameIDs:
                frame = self.get_frame_by_ID(frameID)
                if not frame.is_SubInstrument(subInst):
                    msg = "Frame is not from the selected subInstrument"
                    logger.critical(msg)
                    raise InvalidConfiguration(msg)
                if (not frame.is_valid) and (not include_invalid):
                    msg = "Requesting invalid frame whilst the include_invalid argument is False"
                    logger.critical(msg)
                    raise InvalidConfiguration(msg)
                values.append(frame.get_KW_value(KW))

        if units is not None:
            values = [i.to(units) for i in values]
        if as_value:
            values = [i.value for i in values]

        return values

    def get_valid_frameIDS(self) -> List[int]:
        """Get list of all available frame IDs (across all subInstruments)

        Returns
        -------
        list
            [description]
        """
        out = []
        for subInst in self._inst_type.sub_instruments:
            try:
                out.extend(self.get_frameIDs_from_subInst(subInst, include_invalid=False))
            except NoDataError:
                continue

        return out

    def get_frameIDs_from_subInst(
        self, subInstrument: str, include_invalid: bool = False
    ) -> List[int]:
        """Get all frameIDs associated with a given instrument. By default, only returns the valid ones

        Parameters
        ----------
        subInstrument : str
            SubInstrument name
        include_invalid : bool, optional
            If True, also return IDs associated with the invalid frames, by default False

        Returns
        -------
        list
            [description]
        """

        frameIDS = [
            i
            for i in self.frameID_map[subInstrument]
            if self.get_frame_by_ID(i).is_valid or include_invalid
        ]
        if len(frameIDS) == 0:
            msg = f"There is no available observation in {subInstrument}"
            raise NoDataError(msg)

        return list(frameIDS)

    def get_frame_by_ID(self, frameID: int) -> Frame:
        """
        Return the frame object that is associated with a given ID

        Parameters
        ----------
        frameID

        Returns
        -------

        """
        return self.observations[frameID]

    def get_subInstruments_with_valid_frames(self) -> list:
        """Find all subInstruments that have at least one valid observation

        Returns
        -------
        list
            SubInstruments that have at least one valid observation
        """
        out = []
        for subInst in self._inst_type.sub_instruments:
            try:
                _ = self.get_frameIDs_from_subInst(subInst, include_invalid=False)
                out.append(subInst)
            except NoDataError:
                continue

        return out

    def get_available_subInstruments(self) -> list:
        return self.frameID_map.keys()

    def get_instrument_information(self) -> dict:
        return self._inst_type.instrument_properties

    def get_stellar_template(self, subInst: str):
        return self.StellarModel.request_data(subInst)

    def get_stellar_model(self) -> StellarModel:
        return self.StellarModel

    def get_metaData(self) -> MetaData:
        return self.metaData

    ########################
    #    Data storage      #
    ########################

    def get_internalPaths(self):
        return self._internalPaths

    def trigger_data_storage(self) -> None:
        output_path = self._internalPaths.root_storage_path
        logger.debug("DataClass storing Data to {}", output_path)
        self.metaData.store_json(output_path)
        logger.debug("DataClass finished data storage")

    def generate_root_path(self, storage_path: Path) -> NoReturn:
        """
        Triggers the parent routine and also generates the root path for the Frames

        Parameters
        ----------
        storage_path

        Returns
        -------

        """
        super().generate_root_path(storage_path)

        # The Frames don't store data inside the Iteration folder!
        frame_root_path = storage_path.parent.parent
        for frame in self.observations:
            frame.generate_root_path(frame_root_path)

    ########################
    #          MISC        #
    ########################

    def _build_frameID_map(self) -> None:
        """Populate the self.frameID_map with the frameIDs available for each subInstrument"""
        for frame_ID, frame in enumerate(self.observations):
            self.frameID_map[frame.sub_instrument].append(frame_ID)

    def show_loadedData_table(self) -> Table:
        """Compute the number of observations available in each subInstrument. Find the number of valid (
        with and without warnings) and the invalid ones!
        """
        tab = Table(["subInstrument", "Total OBS", "Valid OBS [warnings]", "INVALID OBS"])
        total_warnings = 0
        total = 0
        total_valid = 0
        total_invalid = 0
        for subinst, frames in self.frameID_map.items():
            row = [subinst, len(frames)]
            valid = 0
            with_warnings = 0
            invalid = 0
            for frameID in frames:
                frame = self.get_frame_by_ID(frameID)

                if frame.is_valid:
                    valid += 1
                    if frame.has_warnings:
                        with_warnings += 1
                else:
                    invalid += 1
            total_valid += valid
            total_warnings += with_warnings
            total_invalid += invalid
            total += len(frames)
            row.extend([f"{valid} [{with_warnings}]", invalid])
            tab.add_row(row)

        row = ["Total", total, f"{total_valid} [{total_warnings}]", total_invalid]
        tab.add_row(row)
        logger.info(tab)

        return tab

    def __repr__(self):
        return (
            f"Data Class from {self._inst_type.instrument_properties['name']} holding "
            + ", ".join([f"{len(IDS)} OBS from {name}" for name, IDS in self.frameID_map.items()])
        )
