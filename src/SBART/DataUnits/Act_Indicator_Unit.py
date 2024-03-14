from typing import NoReturn, List

import numpy as np
import ujson as json
from pathlib import Path

from loguru import logger
from matplotlib import pyplot as plt

from SBART.Base_Models.UnitModel import UnitModel
from SBART.utils import custom_exceptions
from SBART.utils.json_ready_converter import json_ready_converter
from SBART.utils.paths_tools import build_filename


class ActIndicators_Unit(UnitModel):
    _content_name = "Indicators"
    _name = UnitModel._name + _content_name

    def __init__(
        self,
        available_inds: List[str],
        list_of_fIDS: List[int],
        number_OBS: int,
        tot_number_orders: int,
        load_from_disk: bool = False,
    ):
        """
        Parameters
        ----------
        """
        super().__init__(0, 0)
        self._list_of_fIDs = list_of_fIDS

        if not load_from_disk:
            self.combined_holder = {i: [np.nan for _ in list_of_fIDS] for i in available_inds}
            self.combined_error_holder = {i: [np.nan for _ in list_of_fIDS] for i in available_inds}

            self.indicators_holder = {
                i: np.zeros((number_OBS, tot_number_orders)) + np.nan for i in available_inds
            }
            self.indicators_errors_holder = {
                i: np.zeros((number_OBS, tot_number_orders)) + np.nan for i in available_inds
            }
        else:
            self.combined_holder = {}
            self.combined_error_holder = {}
            self.indicators_holder = {}
            self.indicators_errors_holder = {}

    def store_orderwise_indicators(self, frameID, order, ind_name, ind_value, ind_err):
        index = self._list_of_fIDs.index(frameID)

        self.indicators_holder[ind_name][index, order] = ind_value
        self.indicators_errors_holder[ind_name][index, order] = ind_err

    def store_combined_indicators(self, frameID, ind_name, ind_value, ind_err):
        index = self._list_of_fIDs.index(frameID)
        self.combined_holder[ind_name][index] = ind_value
        self.combined_error_holder[ind_name][index] = ind_err

    def get_orderwise_measurement_of_frame(self, frameID, ind_name):
        index = self._list_of_fIDs.index(frameID)
        return self.indicators_holder[ind_name][index], self.indicators_errors_holder[ind_name][
            index
        ]

    def get_combined_measurement_of_frame(self, frameID, ind_name):
        index = self._list_of_fIDs.index(frameID)
        return self.combined_holder[ind_name][index], self.combined_error_holder[ind_name][index]

    def get_combined_measurements(self, ind_name):
        return self.combined_holder[ind_name], self.combined_error_holder[ind_name]

    def get_all_orderwise_indicator(self, ind_name):
        return self.indicators_holder[ind_name], self.indicators_errors_holder[ind_name]

    ###
    # Disk IO operations
    ###
    def get_storage_filename(self):
        return build_filename(
            og_path=self._internalPaths.root_storage_path,
            filename=f"RV_step_chi_squared_eval",
            fmt="json",
        )

    def trigger_data_storage(self):
        # Store order-wise measurements!

        for ind_name, order_wise_values in self.indicators_holder.items():
            np.save(
                file=self._internalPaths.root_storage_path / f"{ind_name}.npy",
                arr=order_wise_values,
            )

        for ind_name, order_wise_values in self.indicators_errors_holder.items():
            np.save(
                file=self._internalPaths.root_storage_path / f"{ind_name}_ERR.npy",
                arr=order_wise_values,
            )

        # Store combined measurements:
        data = {}
        for name, combined in self.combined_holder.items():
            data[name] = json_ready_converter(combined)

        for name, combined in self.combined_error_holder.items():
            data[name + "_ERR"] = json_ready_converter(combined)

        with open(
            self._internalPaths.root_storage_path / f"combined_measurments.json", mode="w"
        ) as handle:
            json.dump(data, handle, indent=4)

    @classmethod
    def load_from_disk(cls, rv_cube_fpath: Path):
        """
        Parameters
        ----------
        rv_cube_fpath: path to the RV cube folder. Internally append the folder name from the corresponding data unit
        Returns
        -------
        """
        super().load_from_disk(rv_cube_fpath)
        new_unit = ActIndicators_Unit(
            available_inds=[], load_from_disk=True, number_OBS=0, tot_number_orders=0
        )
        new_unit.generate_root_path(rv_cube_fpath)
        try:
            logger.info("Searching for combined measurements")
            with open(
                new_unit._internalPaths.root_storage_path / f"combined_measurments.json"
            ) as handle:
                combined_measures = json.load(handle)

                for item, data in combined_measures.items():
                    if "_ERR" in item:
                        new_unit.combined_error_holder[item.split("_ERR")[0]] = data
                    else:
                        new_unit.combined_error_holder[item] = data
        except FileNotFoundError:
            logger.critical(
                f"Couldn't find the .json file on {new_unit._internalPaths.root_storage_path}"
            )

        logger.info("Searching for order-wise data")
        npy_files = rv_cube_fpath.glob("*.npy")
        loaded = 0

        for nopy_path in npy_files:
            arr = np.load(nopy_path)
            name = nopy_path.name
            if "_ERR" in name:
                new_unit.indicators_errors_holder[name.split("_ERR")[0]] = arr
            else:
                new_unit.indicators_errors_holder[name] = arr
            loaded += 1

        if loaded % 2 != 0:
            raise custom_exceptions.InvalidConfiguration(
                "Should have loaded an odd number of npy files on the disk folder..."
            )
        logger.debug("Finished loading all npy files from disk")
        return new_unit

    def generate_root_path(self, storage_path: Path) -> NoReturn:
        if isinstance(storage_path, str):
            storage_path = Path(storage_path)
        storage_path /= self._content_name
        super().generate_root_path(storage_path)
