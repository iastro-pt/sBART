import json
from pathlib import Path

from matplotlib import pyplot as plt

from SBART.Base_Models.UnitModel import UnitModel
from SBART.utils import custom_exceptions
from SBART.utils.json_ready_converter import json_ready_converter
from SBART.utils.paths_tools import build_filename


class Classical_Unit(UnitModel): 
    _content_name = "Classical"
    _name = UnitModel._name + _content_name

    def __init__(self):
        """
        Parameters
        ----------
        """
        super().__init__(0, 0)

        self.chi_squared_profile = {}

    def store_ChiSquared(self, frameID, order, rvs, chi_squared, fit_coeffs):
        if frameID not in self.chi_squared_profile:
            self.chi_squared_profile[frameID] = {}

        self.chi_squared_profile[frameID][order] = {"RVs": rvs,
                                                    "profile": chi_squared,
                                                    "fit_params": fit_coeffs,
                                                    }

    def get_ChiSquared_order_information(self, frameID: int) -> dict:
        try:
            return self.chi_squared_profile[frameID]
        except KeyError as exc:
            raise custom_exceptions.NoDataError(f"There is no information from {frameID=}")

    def get_ChiSquared_order_order_information(self, frameID, order):
        try:
            return self.get_ChiSquared_order_information(frameID)[order]
        except KeyError as exc:
            raise custom_exceptions.NoDataError(f"There is no information order {order=}")

    def plot_ChiSquared(self, frameID, order):

        if frameID == "all":
            frames = list(self.chi_squared_profile.keys())
            if len(frames) == 0:
                raise custom_exceptions.NoDataError("There is no chi squared value stored in this dataUnit")
        else:
            frames = [frameID]

        fig, axis = plt.subplots()

        for f_ID in frames:
            frame_info = self.get_ChiSquared_order_information(f_ID)

            if order == 'all':
                orders = list(frame_info.keys())
            else:
                orders = order

            for order_to_use in orders:
                ord_info = self.get_ChiSquared_order_order_information(f_ID, order_to_use)
                axis.scatter(ord_info["RVs"], ord_info["profile"])

        plt.show()

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
        data = json_ready_converter(self.chi_squared_profile)

        with open(self.get_storage_filename(), mode="w") as handle:
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
        new_unit = Classical_Unit()
        with open(new_unit.get_storage_filename()) as handle:
            new_unit.chi_squared_profile = json.load(handle)

        return new_unit