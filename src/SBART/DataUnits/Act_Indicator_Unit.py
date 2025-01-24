from __future__ import annotations

from pathlib import Path
from typing import List, NoReturn

import numpy as np
import ujson as json
from loguru import logger

from SBART.Base_Models.UnitModel import UnitModel
from SBART.Components.OrderWiseRVs import OrderWiseRVs
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
        tot_number_orders: int,
        load_from_disk: bool = False,
    ):
        super().__init__(0, 0)
        self._list_of_fIDs = list_of_fIDS
        self.subinst = "none"
        self.combined_holder = {}
        self.combined_error_holder = {}
        self.indicators_holder: None | dict[str, OrderWiseRVs] = {}
        self.load = True
        if not load_from_disk:
            self.load = False
            self.combined_holder = {i: [np.nan for _ in list_of_fIDS] for i in available_inds}
            self.combined_error_holder = {i: [np.nan for _ in list_of_fIDS] for i in available_inds}

            self.indicators_holder: dict[str, OrderWiseRVs] = {
                i: OrderWiseRVs(frameIDs=list_of_fIDS, N_orders=tot_number_orders) for i in available_inds
            }

    def store_orderwise_indicators(self, frameID, order, ind_name, ind_value, ind_err, status) -> None:
        self.indicators_holder[ind_name].store_order_data(
            frameID=frameID, order=order, RV=ind_value, error=ind_err, status=status
        )

    def store_combined_indicators(self, frameID, ind_name, ind_value, ind_err):
        index = self._list_of_fIDs.index(frameID)
        self.combined_holder[ind_name][index] = ind_value
        self.combined_error_holder[ind_name][index] = ind_err

    def get_orderwise_measurement_of_frame(self, frameID, ind_name):
        return self.indicators_holder[ind_name].get_orderwise_values(frameID=frameID)[:2]

    def get_combined_measurement_of_frame(self, frameID, ind_name):
        index = self._list_of_fIDs.index(frameID)
        return (
            self.combined_holder[ind_name][index],
            self.combined_error_holder[ind_name][index],
        )

    def get_combined_measurements(self, ind_name):
        return self.combined_holder[ind_name], self.combined_error_holder[ind_name]

    def get_all_orderwise_indicator(self, ind_name):
        return self.indicators_holder[ind_name].data[:2]

    def merge_with(self, new_unit: ActIndicators_Unit) -> None:
        for row_index, frame in enumerate(new_unit._list_of_fIDs):
            added_to_framelist = False
            for indicator in self.combined_holder:
                all_rv, all_err, status = new_unit.indicators_holder[indicator].data

                ord_sta = status.get_status_from_order(frameID=frame, all_orders=True)
                ord_val = all_rv[row_index]
                ord_err = all_err[row_index]
                if frame in self._list_of_fIDs:
                    # Known frame being updated
                    self.indicators_holder[indicator].reset_epoch(frameID=frame)
                    orders = list(range(len(ord_err)))
                    for o, r, e, s in zip(orders, ord_val, ord_err, [i.all_flags for i in ord_sta]):
                        self.indicators_holder[indicator].store_order_data(
                            frameID=frame, order=o, RV=r, error=e, status=s
                        )

                else:
                    if not added_to_framelist:
                        self._list_of_fIDs.append(frame)
                        added_to_framelist = True
                    self.indicators_holder[indicator].add_new_epochs(1, [frame])
                    orders = list(range(len(ord_err)))
                    for o, r, e, s in zip(orders, ord_val, ord_err, [i.all_flags for i in ord_sta]):
                        self.indicators_holder[indicator].store_order_data(
                            frameID=frame, order=o, RV=r, error=e, status=s
                        )

    ###
    # Disk IO operations
    ###
    def get_storage_filename(self):
        return build_filename(
            og_path=self._internalPaths.root_storage_path,
            filename="RV_step_chi_squared_eval",
            fmt="json",
        )

    def trigger_data_storage(self):
        # Store order-wise measurements!

        for ind_name, holder in self.indicators_holder.items():
            holder.store_to_disk(
                self._internalPaths.root_storage_path, extra_identifier=ind_name, associated_subInst=self.subinst
            )

        # Store combined measurements:
        data = {}
        for name, combined in self.combined_holder.items():
            data[name] = json_ready_converter(combined)

        for name, combined in self.combined_error_holder.items():
            data[name + "_ERR"] = json_ready_converter(combined)

        with open(
            self._internalPaths.root_storage_path / "combined_measurments.json",
            mode="w",
        ) as handle:
            json.dump(data, handle, indent=4)

    @classmethod
    def load_from_disk(cls, rv_cube_fpath: Path):
        """Parameters
        ----------
        rv_cube_fpath: path to the RV cube folder. Internally append the folder name from the corresponding data unit

        Returns
        -------

        """
        super().load_from_disk(rv_cube_fpath)
        new_unit = ActIndicators_Unit(available_inds=[], list_of_fIDS=[], load_from_disk=True, tot_number_orders=0)
        new_unit.generate_root_path(rv_cube_fpath)
        try:
            logger.info("Searching for combined measurements")
            with open(new_unit._internalPaths.root_storage_path / "combined_measurments.json") as handle:
                combined_measures = json.load(handle)
                for item, data in combined_measures.items():
                    if "_ERR" in item:
                        new_unit.combined_error_holder[item.split("_ERR")[0]] = data
                    else:
                        new_unit.combined_holder[item] = data
                # breakpoint()
        except FileNotFoundError:
            logger.critical(f"Couldn't find the .json file on {new_unit._internalPaths.root_storage_path}")

        logger.info("Searching for order-wise data")

        for indicator_path in rv_cube_fpath.glob("Indicators/*.fits"):
            name = indicator_path.stem.split("_")[1]
            new_unit.indicators_holder[name] = OrderWiseRVs.load_from_disk(subInst_path=".", full_path=indicator_path)
        new_unit._list_of_fIDs = new_unit.indicators_holder[name].frameIDs
        return new_unit

    def generate_root_path(self, storage_path: Path) -> NoReturn:
        if isinstance(storage_path, str):
            storage_path = Path(storage_path)
        storage_path /= self._content_name
        super().generate_root_path(storage_path)
