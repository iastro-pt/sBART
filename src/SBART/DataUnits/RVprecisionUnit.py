from pathlib import Path
from typing import Any, Dict, List, NoReturn

import ujson as json
from loguru import logger
from tabletexifier import Table

from SBART.Base_Models.UnitModel import UnitModel
from SBART.utils import custom_exceptions
from SBART.utils.json_ready_converter import json_ready_converter
from SBART.utils.paths_tools import build_filename


class RV_Precision_Unit(UnitModel):
    _content_name = "RV_precision"
    _name = UnitModel._name + _content_name

    def __init__(self):
        """Parameters
        ----------

        """
        super().__init__(0, 0)
        self._optim_table: List[Table] = []
        self.RVcontent_info = {}

    def store_RVcontent(
        self,
        frameID: int,
        order: int,
        pred_rv: float,
        pred_err: float,
        quality: float,
        pix_sum_in_template: int,
    ):
        """Store predicted RV content from one order

        Args:
            frameID (int): FrameID
            order (int): Spectral order
            pred_rv (float): Predicted RV, km/s
            pred_err (float): Expected precision, km/s
            quality (float): Qaulity factor
            pix_sum_in_template (int): number of pixels in the template

        """
        if frameID not in self.RVcontent_info:
            self.RVcontent_info[frameID] = {}

        self.RVcontent_info[frameID][order] = {
            "pix_sum_in_template": pix_sum_in_template,
            "quality": quality,
            "pre_rv": pred_rv,
            "pred_err": pred_err,
        }

    def get_RVcontent_frameID_information(self, frameID: int) -> Dict[int, Dict[str, List[Any]]]:
        """Retrieve the RV from one frameID

        Args:
            frameID (int): FrameID

        Raises:
            custom_exceptions.NoDataError: _description_

        Returns:
            Dict[str, List[Any]]: RV information from each order (keys), each providing a dictionary with the following keys:
            "pix_sum_in_template", "quality", "pred_rv", "pred_err"

        """
        try:
            return self.RVcontent_info[frameID]
        except KeyError as exc:
            raise custom_exceptions.NoDataError(f"There is no information from {frameID=}") from exc

    def get_RVcontent_order_information(self, frameID: int, order: int) -> Dict[str, List[Any]]:
        """Retrieve RV content information from one order

        Args:
            frameID (int): frameID
            order (int): order

        Raises:
            custom_exceptions.NoDataError: The order doesn't have any information

        Returns:
            Dict[str, Any]: Dictionary with the following keys:
            "pix_sum_in_template", "quality", "pred_rv", "pred_err"

        """
        try:
            return self.get_RVcontent_frameID_information(frameID)[order]
        except KeyError as exc:
            raise custom_exceptions.NoDataError(f"There is no information order {order=}") from exc

    def ingest_table(self, optimized_table):
        self._optim_table.append(optimized_table)

    ###
    # Disk IO operations
    ###
    def get_storage_filename(self):
        return build_filename(
            og_path=self._internalPaths.root_storage_path,
            filename="RVcontent_metrics",
            fmt="json",
        )

    def trigger_data_storage(self):
        data = {}
        for key, values in self.RVcontent_info.items():
            data[key] = {}
            for key_1, order_values in values.items():
                data[key][key_1] = {}
                for key_2, final_values in order_values.items():
                    data[key][key_1][key_2] = json_ready_converter(final_values)

        with open(self.get_storage_filename(), mode="w") as handle:
            json.dump(data, handle, indent=4)

        tab_tame = build_filename(
            og_path=self._internalPaths.root_storage_path,
            filename="optimal_bins",
            fmt="txt",
        )
        if not self._optim_table:
            logger.warning(f"No optimization table stored in {self.name}")

        for tab in self._optim_table:
            tab.write_to_file(
                path=tab_tame,
            )

    @classmethod
    def load_from_disk(cls, rv_cube_fpath: Path):
        """Parameters
        ----------
        rv_cube_fpath: path to the RV cube folder. Internally append the folder name from the corresponding data unit

        Returns
        -------

        """
        super().load_from_disk(rv_cube_fpath)
        new_unit = RV_Precision_Unit()
        new_unit.generate_root_path(rv_cube_fpath)

        try:
            with open(new_unit.get_storage_filename()) as handle:
                chi_squared_profile = json.load(handle)
                profile = {}
                for str_key, info in chi_squared_profile.items():
                    profile[int(str_key)] = {int(j): k for j, k in info.items()}
                new_unit.RVcontent_info = profile
        except FileNotFoundError:
            logger.critical(f"Couldn't find the .json file on {new_unit.get_storage_filename()}")

        return new_unit

    def generate_root_path(self, storage_path: Path) -> NoReturn:
        if isinstance(storage_path, str):
            storage_path = Path(storage_path)
        storage_path /= self._content_name
        super().generate_root_path(storage_path)
