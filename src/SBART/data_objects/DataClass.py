import ujson as json
from pathlib import Path
from typing import Iterable, List, NoReturn, Optional, Type, Union, Dict, Any, Tuple

import numpy as np
from loguru import logger
from tabletexifier import Table

from SBART import __version__
from SBART.utils.BASE import BASE
from SBART.Base_Models.Frame import Frame
from SBART.Quality_Control.activity_indicators import Indicators
from SBART.data_objects.MetaData import MetaData
from SBART.data_objects.RV_outputs import RV_holder
from SBART.data_objects.Target import Target
from SBART.template_creation.StellarModel import StellarModel
from SBART.template_creation.TelluricModel import TelluricModel
from SBART.utils.custom_exceptions import FrameError, InvalidConfiguration, NoDataError
from SBART.utils.shift_spectra import apply_RVshift
from SBART.utils.spectral_conditions import ConditionModel as CondModel
from SBART.utils.status_codes import (  # for entire frame; for individual pixels
    ACTIVITY_LINE,
    TELLURIC,
    Status,
    SIGMA_CLIP_REJECTION,
)
from SBART.utils.types import UI_PATH
from SBART.utils.units import kilometer_second, meter_second
from SBART.utils import custom_exceptions


class DataClass(BASE):
    """
    The user-facing object that handles the loading and data access to the spectral data, independently of the instrument.
    Furthermore, this must be launched as a proxyObject (insert docs here) in order to avoid problems with data syncronization
    and optimize the speed of the code.

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
        input_files: Iterable[UI_PATH],
        storage_path: UI_PATH,
        instrument: Type[Frame],
        instrument_options: dict,
        reject_subInstruments: Optional[Iterable[str]] = None,
        target_name: str = None,
        sigma_clip_RVs: Optional[float] = None,
    ):
        """
        Parameters
        =============
        input_files:
            Either a path to a txt file, or a list of S2d files
        storage_path:
            Root path of the SBART outputs
        instrument:
            Instrument that we will be loading data from. Must be an object of type SBART.Instruments
        instrument_options
        reject_subInstruments
            Iterable with names of subInstruments to automatically reject
        target_name:
            Original name of the target. To be deprecated
        sigma_clip_RVs:
            If it is a positive integer, reject frames with a sigma clip on the DRS RVs
        """
        super().__init__()
        self.sigma_clip_RVs = sigma_clip_RVs

        self._inst_type = instrument
        self.input_file = input_files

        # Hold all of the frames
        self.observations: Iterable[Frame] = []

        self.metaData = MetaData()

        if reject_subInstruments is not None:
            logger.warning("Rejecting subInstruments: {}".format(reject_subInstruments))

        OBS_list = []
        if isinstance(input_files, (str, Path)):
            logger.info("DataClass loading data from {}", self.input_file)
            with open(input_files) as input_file:
                for line in input_file:
                    OBS_list.append(Path(line.split("\n")[0]))

        elif isinstance(input_files, Iterable):
            logger.info("DataClass opening {} files from a list/tuple", len(input_files))

            OBS_list = [Path(i) if isinstance(i, str) else i for i in input_files]
        else:
            raise TypeError()

        for frameID, filepath in enumerate(OBS_list):
            self.observations.append(
                self._inst_type(
                    filepath,
                    instrument_options,
                    reject_subInstruments,
                    frameID=frameID,
                    quiet_user_params=frameID != 0,  # Only the first frame will output logs
                )
            )

        self.generate_root_path(storage_path)

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

        self.load_instrument_extra_information()

        for frame in self.observations:
            frame.initialize_modelling_interface()
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
            RV_RESULTS = RV_holder.load_from_disk(LoadingPath_previousRun)
        except FileNotFoundError:
            raise InvalidConfiguration("RV outputs couldn't be found on the provided path")

        for ID_index, frameID in enumerate(self.get_valid_frameIDS()):
            frame = self.get_frame_by_ID(frameID)
            cube = RV_RESULTS.get_RV_cube(frame.sub_instrument, merged=use_merged_cube)
            _, sbart_rv, sbart_uncert = cube.get_RV_from_ID(
                frameID=frameID,
                which="SBART",
                apply_SA_corr=False,
                as_value=False,
                units=None,
                apply_drift_corr=False,
            )

            previous_filename = cube.cached_info["date_folders"][cube.frameIDs.index(frameID)]

            if previous_filename != frame.file_path:
                msg = (
                    "Loading RVs from cube with different frameID layouts of {} ({} vs {})".format(
                        frame.sub_instrument, previous_filename, frame.file_path
                    )
                )
                logger.critical(msg)
                raise InvalidConfiguration(msg)

            cube_ids = cube.frameIDs

            frame.store_previous_SBART_result(RV=sbart_rv, RV_err=sbart_uncert)

    def reject_order_region_from_frame(self, frameID: int, order: int, region):
        frame = self.get_frame_by_ID(frameID)
        frame.reject_wavelength_region_from_order(order, region)
        frame.close_arrays()  # ensure that the arrays are closed and that the next call will load data

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

    def replace_frames_with_S2D_version(self, new_shape: Optional[Tuple[int, int]] = None):
        """
        In-place substitution of all frames with their S2D-compatible shapes!
        Returns
        -------

        """
        logger.warning("Transforming the frames to have a S2D-compatible shape")
        for index, frame in enumerate(self.observations):
            s2d_frame = frame.copy_into_S2D(new_S2D_size=new_shape)
            s2d_frame.build_mask()
            self.observations[index] = s2d_frame
            del frame

    def ingest_StellarModel(self, Stellar_Model: StellarModel) -> None:
        logger.debug("Ingesting StellarModel into the DataClass")
        if self.StellarModel is not None:
            logger.warning(
                "Stellar template has already been ingested. Switching old template by the new one"
            )

        # Empty update just to ensure initialization of the modelling interfaces
        Stellar_Model.update_interpol_properties({})

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

        if self.sigma_clip_RVs is not None:
            logger.info(
                f"Rejecting frames that are more than {self.sigma_clip_RVs} sigma away from mean RV"
            )

            for subInstrument in self.get_subInstruments_with_valid_frames():
                RV = self.collect_RV_information(
                    KW="DRS_RV", subInst=subInstrument, include_invalid=False, as_value=True
                )
                err = self.collect_RV_information(
                    KW="DRS_RV_ERR", subInst=subInstrument, include_invalid=False, as_value=True
                )
                mean_RV = np.median(RV)
                metric = np.std(
                    RV
                )  # using the same sigma clip as https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.sigmaclip.html

                bounds = [
                    mean_RV - self.sigma_clip_RVs * metric,
                    mean_RV + self.sigma_clip_RVs * metric,
                ]

                bad_indexes = np.where(np.logical_or(RV < bounds[0], RV > bounds[1]))
                valid_frameIDs = self.get_frameIDs_from_subInst(subInstrument)
                bad_IDS = np.asarray(valid_frameIDs)[bad_indexes]

                for frameID_to_reject in bad_IDS:
                    bad_frame = self.get_frame_by_ID(frameID_to_reject)
                    logger.warning(f"{bad_frame} rejected due to sigma clipping of DRS RVs")
                    bad_frame.add_to_status(SIGMA_CLIP_REJECTION)

                logger.info(f"Sigma clip rejected {len(bad_IDS)} frames of {subInstrument}")

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

    def set_all_as_Zscore_frames(self) -> NoReturn:
        logger.warning("Setting all frames as a zero-mean unit variance")
        for frameID in self.get_valid_frameIDS():
            frame = self.get_frame_by_ID(frameID)
            frame.set_frame_as_Zscore()

    def normalize_all(self) -> NoReturn:
        """
        Launch the normalization for all (valid) frames
        Returns
        -------

        """
        logger.info("Normalizing all frames")
        for frameID in self.get_valid_frameIDS():
            frame = self.get_frame_by_ID(frameID)
            frame._never_close = True

        for subInst in self.get_subInstruments_with_valid_frames():
            self.normalize_all_from_subInst(subInst)

    def update_uncertainties_of_frame(self, frameID: int, new_uncerts) -> NoReturn:
        """
        Update the flux uncertainties of a given frame

        .. warning::
            This will change your measured data, be careful with this!!!
        """
        logger.info(f"Setting uncertainties of frame (ID = {frameID}")

        frame = self.get_frame_by_ID(frameID)
        frame.update_uncertainties(new_uncerts)
        frame._never_close = True
        logger.warning("This frame will never close until SBART finished!")

    def normalize_all_from_subInst(self, subInst: str) -> NoReturn:
        """
        Normalizing all (valid) frames from a given subInstrument
        Parameters
        ----------
        subInst

        Returns
        -------

        """
        logger.debug(f"Normalizing all frames from {subInst}")
        for fId in self.get_frameIDs_from_subInst(subInst):
            frame = self.get_frame_by_ID(fId)
            frame.normalize_spectra()

    def scale_up_all_observations(self, factor: float) -> NoReturn:
        """
        Multiply the flux and uncertainties by a given flux level (avoid possible SNR issues)
        Parameters
        ----------
        factor

        Returns
        -------

        """

        logger.warning(f"Scaling up all spectra by a factor of {factor}")
        for fId in self.get_valid_frameIDS():
            frame = self.get_frame_by_ID(fId)
            frame.scale_spectra(factor)

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

    def update_interpol_properties_of_all_frames(self, new_properties: Dict[str, Any]):
        if not isinstance(new_properties, dict):
            raise custom_exceptions.InvalidConfiguration(
                "The interpolation properties must be passed as a dictionary"
            )

        for frame in self.observations:
            frame.set_interpolation_properties(new_properties)

    def update_interpol_properties_of_stellar_model(self, new_properties: Dict[str, Any]):
        if not isinstance(new_properties, dict):
            raise custom_exceptions.InvalidConfiguration(
                "The interpolation properties must be passed as a dictionary"
            )

        if self.StellarModel is None:
            raise custom_exceptions.NoDataError("The Stellar Model wasn't ingested")
        self.StellarModel.update_interpol_properties(new_properties)

    def update_frame_interpol_properties(self, frameID, new_properties) -> NoReturn:
        """
        Allow to update the interpolation settings from the outside, so that any object can configure
        the interpolation as it wishes
        """

        frame = self.get_frame_by_ID(frameID)
        frame.set_interpolation_properties(new_properties)

    def interpolate_frame_order(
        self, frameID, order, new_wavelengths, shift_RV_by, RV_shift_mode, include_invalid=False
    ):
        """
        Interpolate a given order to a new wavelength solution
        """
        frame = self.get_frame_by_ID(frameID)
        return frame.interpolate_spectrum_to_wavelength(
            order=order,
            new_wavelengths=new_wavelengths,
            shift_RV_by=shift_RV_by,
            RV_shift_mode=RV_shift_mode,
            include_invalid=include_invalid,
        )

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
        return_frameIDs: bool = False,
    ) -> Union[list, Tuple[List[float], List[int]]]:
        """
        Parse through the loaded observations and retrieve a specific KW from
        all of them. There is no sort of the files. The output will follow the
        order of the files loaded in memory!

        Parameters
        ----------
        return_frameIDs
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
        all_frameIDs = []

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
                all_frameIDs.append(frameID)

        if return_frameIDs:
            return output, available_frameIDs

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
        return self.observations[0].instrument_properties

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

        for frame in self.observations:
            frame.trigger_data_storage()

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
        frame_root_path = self._internalPaths.root_storage_path
        for frame in self.observations:
            frame.generate_root_path(frame_root_path)

    ########################
    #          MISC        #
    ########################
    def has_instrument_data(self, instrument: str) -> bool:
        """
        Check if the first loaded frame is of a given Instrument

        .. warning::
            in the off-chance that we mix data this will give problems...
        """
        return self.observations[0].is_Instrument(instrument)

    def has_instrument_data(self, instrument: str) -> bool:
        """
        Check if the first loaded frame is of a given Instrument

        .. warning::
            in the off-chance that we mix data this will give problems...
        """
        return self.observations[0].is_Instrument(instrument)

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

    def load_instrument_extra_information(self):
        """
        See if the given instrument is one of the ones that has extra information to load.
        If so, then
        """
        info_load_map = {"CARMENES": self.load_CARMENES_extra_information}

        logger.info("Checking if the instrument has extra data to load")
        for key, load_func in info_load_map.items():
            if self.has_instrument_data(key):
                logger.info(f"Dataclass has {key} data. Extra loading is being triggered")
                load_func()
                return

        logger.info("Current instrument does not need to load anything from the outside")

    def load_CARMENES_extra_information(self) -> None:
        """CARMENES pipeline does not give RVs, we have to do an external load of the information

        Parameters
        ----------
        shaq_folder : str
            Path to the main folder of shaq-outputs. where all the KOBE-*** targets live
        """

        name_to_search = self.Target.true_name

        if self.observations[0]._internal_configs["is_KOBE_data"]:
            if "KOBE-" not in name_to_search:
                name_to_search = "KOBE-" + name_to_search  # temporary fix for naming problem!
        else:
            logger.info(f"Not loading KOBE data, searching for {name_to_search} dat file with Rvs")

        shaq_folder = Path(self.observations[0]._internal_configs["shaq_output_folder"])
        override_BERV = self.observations[0]._internal_configs["override_BERV"]

        if shaq_folder.name.endswith("dat"):
            logger.info("Received the previous RV file, not searching for outputs")
            shaqfile = shaq_folder
        else:
            logger.info("Searching for outputs of previous RV extraction")
            shaqfile = shaq_folder / name_to_search / f"{name_to_search}_RVs.dat"

        if shaqfile.exists():
            logger.info("Loading extra CARMENES data from {}", shaqfile)
        else:
            logger.critical(f"RV file does not exist on {shaqfile}")
            raise custom_exceptions.InvalidConfiguration("Missing RV file for data")

        number_loads = 0
        locs = []
        loaded_BJDs = [frame.get_KW_value("BJD") for frame in self.observations]
        with open(shaqfile) as file:
            for line in file:
                if "#" in line:  # header or other "BAD" files
                    continue
                # TODO: implement a more thorough check in here, to mark the "bad" frames as invalid!
                ll = line.strip().split()
                if len(ll) == 0:
                    logger.warning(f"shaq RV from {name_to_search} has empty line")
                    continue
                bjd = round(float(ll[1]) - 2400000.0, 7)  # we have the full bjd date

                try:
                    index = loaded_BJDs.index(
                        bjd
                    )  # to make sure that everything is loaded in the same order
                    locs.append(index)
                except ValueError:
                    logger.warning("RV shaq has entry that does not exist in the S2D files")
                    continue

                self.observations[index].import_KW_from_outside(
                    "DRS_RV", float(ll[5]) * kilometer_second, optional=False
                )
                self.observations[index].import_KW_from_outside(
                    "DRS_RV_ERR", float(ll[3]) * kilometer_second, optional=False
                )
                if override_BERV:
                    self.observations[index].import_KW_from_outside(
                        "BERV", float(ll[10]) * kilometer_second, optional=False
                    )

                self.observations[index].import_KW_from_outside(
                    "FWHM", float(ll[11]), optional=True
                )
                self.observations[index].import_KW_from_outside(
                    "BIS SPAN", float(ll[13]), optional=True
                )

                drift_val = np.nan_to_num(float(ll[7])) * meter_second
                drift_err = np.nan_to_num(float(ll[8])) * meter_second
                self.observations[index].import_KW_from_outside("drift", drift_val, optional=False)
                self.observations[index].import_KW_from_outside(
                    "drift_ERR", drift_err, optional=False
                )

                number_loads += 1

                self.observations[index].finalized_external_data_load()
        if number_loads < len(self.observations):
            msg = "RV shaq outputs does not have value for all S2D files of {} ({}/{})".format(
                name_to_search, number_loads, len(self.observations)
            )
            logger.critical(msg)

    def __repr__(self):
        return (
            f"Data Class from {self._inst_type.instrument_properties['name']} holding "
            + ", ".join([f"{len(IDS)} OBS from {name}" for name, IDS in self.frameID_map.items()])
        )
