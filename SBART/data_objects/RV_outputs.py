"""
Store the :py:class:`~SBART.data_objects.RV_cube` of all the available sub-Instruments (for which SBART
computed RVs)
"""
import glob
from pathlib import Path
from typing import List, NoReturn, Optional, Union

import numpy as np
from loguru import logger
from tabletexifier import Table

from SBART import __version__
from SBART.utils.BASE import BASE
from SBART.data_objects.RV_cube import RV_cube
from SBART.utils.custom_exceptions import InvalidConfiguration, NoComputedRVsError
from SBART.utils.paths_tools import build_filename, find_latest_version, ensure_path_from_input


class RV_holder(BASE):
    """
    Manages the :py:class:`~SBART.data_objects.RV_cube` that the :py:mod:`~SBART.rv_calculation` routines produce.

    The goal of this class is to provide a simple user-interface to access the RV outputs of all
    sub-Instruments, and provide centralized trigger for data storage routines.

    .. note::
        This class is never created by the user, but it can be retrieved/returned from some functions.

    .. note::
        It is possible to load the RV results from disk by using:
        :py:func:`~SBART.outside_tools.Load_RVoutputs.find_RVoutputs`
    """

    _valid_keys = [
        "BJD",
        "MJD",
        "RVc",
        "RVc_ERR",
        "OBJ",
        "SA",
        "DRIFT",
        "DRIFT_ERR",
        "full_path",
        "filename",
        "frameIDs",
        "DLW",
        "DLW_ERR"
    ]

    def __init__(self, subInsts: List[str], output_keys: List[str], storage_path: Path):
        super().__init__(
            user_configs={},
            needed_folders={
                "individual_subInst": "individual_subInst",
                "merged_subInst": "merged_subInst",
            },
            root_level_path=storage_path,
        )

        self.output_keys = None

        self._individual_cubes = {subInst: None for subInst in subInsts}

        self._merged_cubes = {subInst: None for subInst in subInsts}
        self.update_output_keys(output_keys)

    def update_output_keys(self, keys):
        self.output_keys = keys
        self._process_output_keys()

    def remove_subInstrument_data(self, subInst) -> None:
        self._individual_cubes[subInst] = None
        self._merged_cubes[subInst] = None

    def add_RV_cube(self, subInst: str, RV_cube, is_merged: bool = False) -> NoReturn:
        """Store a new RV cube inside the holder

        Parameters
        ----------
        subInst : str
            sub-Instrument that generated the data
        RV_cube : :py:class:`~SBART.data_objects.RV_cube`
            RV cube with the data
        is_merged : bool, optional
            if True, the RVs were computed by merging the skipped orders of each sub-Inst, by default False
        """
        self._validate_subInst(subInst)
        logger.debug(
            "Adding new RV cube to the RV_holder: {}; Is merged: {}",
            RV_cube.name,
            is_merged,
        )

        if is_merged:
            self._merged_cubes[subInst] = RV_cube
        else:
            self._individual_cubes[subInst] = RV_cube

    def get_orders_to_skip(self, subInst: str) -> List[int]:
        """Retrieve the orders that were skipped for the calculation of RVs
        Parameters
        ----------
        subInst : str
            Name of the sub-Instrument, or 'merged'. Tis refers to how we selected the orders to be
        discarded. If sub-Instrument, then we forced the same orders inside each individual dataSet. If
        'merged' the orders were forced to be the same across the entire data that was loaded!

        Returns
        -------
        List[int]
            List of orders that was skipped

        Raises
        ------
        InvalidConfiguration
            [description]
        NoComputedRVsError
            [description]
        """

        if subInst == "merged":
            # in the merged cubes the orders to skip are equal in all of them...
            selected_cube = self.get_RV_cube(list(self._merged_cubes.keys())[0], merged=True)
        else:
            self._validate_subInst(subInst)
            selected_cube = self.get_RV_cube(subInst, merged=False)

        return selected_cube.problematic_orders

    def get_stored_subInsts(self, merged: bool) -> List[str]:
        """
        Get a list of the subInsts that have available RV cubes inside this object

        Parameters
        ----------
        merged: bool
            If True, check for the "viable" instruments in the "merged" category

        Returns
        -------
        subInst_list
            List of the subInstruments that have RVs stored in this object
        """

        if merged:
            selected_cube = self._merged_cubes
        else:
            selected_cube = self._individual_cubes
        return list(selected_cube.keys())

    def get_RV_cube(self, subInst: str, merged: bool):
        """
        Return a RV cube for the given subInstrument. The selected cybe is the one that was created when considering (or not)
        a merged set of orders to skip for the RV extraction

        Parameters
        ----------
        subInst : str
            desired SubInstrument
        merged : bool
            Return the cube that was created when considering (if True) or not (if False) a combined set of orders to skip (across all
            subInstruments)

        Returns
        -------
        cube: :py:class:`~SBART.data_objects.RV_cube`
            RV_cube object for the given configuration

        Raises
        ------
        InvalidConfiguration
            If the subInst has no valid data
        NoComputedRVsError
            If there is no RV cube stored for the configurations requested by the user
        """
        self._validate_subInst(subInst)
        if merged:
            selected_cube = self._merged_cubes[subInst]
        else:
            selected_cube = self._individual_cubes[subInst]

        if selected_cube is None:
            raise NoComputedRVsError()

        return selected_cube

    def _validate_subInst(self, subInst: str) -> None:
        """Check if the provided subInstrument is among the valid ones!

        Parameters
        ----------
        subInst : str
            subInstrument name that will be accessed

        Raises
        ------
        InvalidConfiguration
            If the name is invalid
        """
        valid_options = list(self._individual_cubes.keys())
        if subInst not in valid_options:
            msg = "sub-Instrument <{}> is not a valid value. Select one from <{}>".format(
                subInst, valid_options
            )
            logger.critical(msg)
            raise InvalidConfiguration(msg)

    #########################
    #     Handle outputs    #
    #########################

    def store_computed_RVs_to_disk(self, dataClassProxy, which_subInst: str) -> None:
        storage_path = self._internalPaths.root_storage_path
        logger.debug("Storing the RV products to disk: {}", storage_path)

        header = self._generate_output_header()

        for set_index, cube in enumerate(
            [self._individual_cubes[which_subInst], self._merged_cubes[which_subInst]]
        ):
            full_table = Table(header=header, table_style="NoLines")

            if cube is None:
                continue

            cube.export_results(
                header=header,
                keys=self.output_keys,
                dataClassProxy=dataClassProxy,
            )

    def store_complete_timeseries(self) -> None:
        """Store, inside a single txt file, the RV timeseries from all loaded observations!

        Parameters
        ----------
        storage_path : str
            [description]
        """

        header = self._generate_output_header()

        has_individual, has_merged = False, False

        if list(self._individual_cubes.values())[0] is not None:
            has_individual = True
        if list(self._merged_cubes.values())[0] is not None:
            has_merged = True

        cube_folder = ["individual_subInst", "merged_subInst"]
        for set_index, set_of_cubes in enumerate(
            [self._individual_cubes.values(), self._merged_cubes.values()]
        ):
            full_table = Table(header=header, table_style="NoLines")

            if not [has_individual, has_merged][set_index]:
                continue

            for cube in set_of_cubes:
                try:
                    data_block = cube.build_datablock()
                except InvalidConfiguration:
                    continue

                selected_key = cube.time_key
                if self.output_keys[0] != selected_key:
                    logger.warning("User asking for time-key <{}> but we must use <{}>",
                                   self.output_keys[0],
                                   selected_key)
                    self.output_keys[0] = selected_key

                sorted_indexes = np.argsort(data_block[selected_key])

                for sort_index in sorted_indexes:
                    row = []
                    for key in self.output_keys:
                        row.append(data_block[key][sort_index])
                    full_table.add_row(row)

            full_table.write_to_file(
                build_filename(
                    self._internalPaths.get_path_to(cube_folder[set_index], as_posix=False),
                    "full_RVlist",
                    "txt",
                ),
                mode="w",
                write_LaTeX=False,
            )

    def _process_output_keys(self) -> None:
        """Check if Mandatory keys are missing (and add them in that case) and check
        if all keys are valid

        Raises
        ------
        InvalidConfiguration
            If we find a key that is not supported
        """
        logger.debug(f"Validating keys for outputs: {self.output_keys}")

        if self.output_keys is None:
            self.output_keys = []

        time_keys = ["BJD", "MJD"]
        if len(self.output_keys) == 0:
            logger.warning("Output keys is an empty list... Manually creating the output array")
            self.output_keys.append(time_keys[0])

        if self.output_keys[0] not in time_keys:
            logger.warning("Missing time-related key in the selected outputs. Adding it")

            # TODO: do we want to search for the "optimal" one?
            self.output_keys.insert(0, time_keys[0])

        for key_index, key in enumerate(["RVc", "RVc_ERR", "DLW", "DLW_ERR"]):
            if key not in self.output_keys:
                logger.warning(
                    "Mandatory key <{}> not present in the selected outputs. Adding it",
                    key,
                )
                self.output_keys.insert(key_index + 1, key)

        for key in self.output_keys:
            if key not in self.__class__._valid_keys:
                logger.critical(
                    "Output key <{}> is not supported. Can only select from: {}",
                    key,
                    self.__class__._valid_keys,
                )
                raise InvalidConfiguration()

    def _generate_output_header(self) -> List[str]:
        """Generate the header for th output files

        Returns
        -------
        List[str]
            List with the names for the header
        """

        header = []
        for key in self.output_keys:
            entry = key
            if key in ["SA", "DRIFT"]:
                entry += " [m/s] "
            elif key in ["RVc", "RVc_ERR"]:
                entry += " [Km/s] "
            header.append(entry)
        return header

    def generate_new_cube(self, dataClassProxy, subInst, is_merged, has_orderwise_rvs:bool):
        cube_IDS = dataClassProxy.get_frameIDs_from_subInst(subInst)

        cube = RV_cube(subInst, cube_IDS, dataClassProxy.get_instrument_information(), has_orderwise_rvs=has_orderwise_rvs)
        fold_name = "merged_subInst" if is_merged else "individual_subInst"
        cube_root_folder = self._internalPaths.get_path_to(fold_name, as_posix=False)
        cube_root_folder = cube_root_folder / subInst
        cube.generate_root_path(cube_root_folder)
        return cube

    def trigger_data_storage(self, *args, **kwargs) -> NoReturn:
        super().trigger_data_storage(*args, **kwargs)
        for key, cube in self._individual_cubes.items():
            if cube is None:
                continue
            cube.trigger_data_storage()

        for key, cube in self._merged_cubes.items():
            if cube is None:
                continue
            cube.trigger_data_storage()

    @classmethod
    def load_from_disk(
        cls,
        high_level_path: Union[Path, str],
        load_full_flags=False,
        load_work_pkgs=False,
        SBART_version: Optional[str] = None,
        only_load_type: Optional[str] = None
    ):
        high_level_path = ensure_path_from_input(high_level_path,
                                                 ensure_existence=True
                                                 )
        logger.info("Loading RV outputs from {}", high_level_path)

        most_recent_version = find_latest_version(high_level_path)

        SBART_version = SBART_version if SBART_version is not None else most_recent_version

        logger.info("Loading data from SBART version {}", SBART_version)

        if __version__ != SBART_version:
            logger.warning("Current SBART version is {}", __version__)

        # Oh god, my eyes, this is uglyyyyyyyy
        available_paths = [
            Path(path).stem
            for path in glob.glob((high_level_path / "*_subInst/*").as_posix())
            if Path(path).is_dir()
        ]

        available_subInsts = set(available_paths)

        logger.debug(
            "Found {} subInstruments: {}".format(len(available_subInsts), available_subInsts)
        )

        new_holder = RV_holder(
            subInsts=available_subInsts, output_keys=[], storage_path=high_level_path
        )

        for path in high_level_path.iterdir():
            is_merged = "merged" in path.stem
            if only_load_type is not None and path.stem != only_load_type:
                logger.info(f"Only loading rv cubes of the {only_load_type} type.")
                continue
            logger.debug("Loading <{}> data", "merged" if is_merged else "individual")
            if not path.is_dir():
                logger.warning("Found file inside loading folder. Skipping it")
                continue

            if not is_merged and "individual" not in path.stem:
                logger.warning("Found unknown folder inside loading folder. Skipping it!")
                continue

            for subInst in available_subInsts:
                cube_path = path / subInst
                logger.info("Loading RV cube from {}; Path: {}", subInst, cube_path)
                new_cube = RV_cube.load_cube_from_disk(
                    subInst_path=cube_path,
                    load_full_flag=load_full_flags,
                    load_work_pkgs=load_work_pkgs,
                    SBART_version=SBART_version,
                )
                new_holder.add_RV_cube(subInst, RV_cube=new_cube, is_merged=is_merged)

        return new_holder
