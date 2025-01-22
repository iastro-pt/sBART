from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from astropy.io import fits
from loguru import logger

from SBART.utils import custom_exceptions
from SBART.utils.BASE import BASE
from SBART.utils.expected_precision_interval import convert_to_tab, optimize_intervals_over_array
from SBART.utils.paths_tools import build_filename
from SBART.utils.status_codes import ORDER_SKIP, Flag, OrderStatus

if TYPE_CHECKING:
    from SBART.data_objects.DataClass import DataClass
    from SBART.utils.SBARTtypes import UI_PATH


@dataclass
class OrderWiseRVs(BASE):
    frameIDs: list[int]
    N_orders: int

    def __post_init__(self):
        N_epochs = len(self.frameIDs)
        self._Rv_orderwise = np.zeros((N_epochs, self.N_orders)) + np.nan
        self._RvErrors_orderwise = np.zeros((N_epochs, self.N_orders)) + np.nan

        self._OrderStatus = OrderStatus(N_orders=self.N_orders, frameIDs=self.frameIDs)

    def get_index_of_frameID(self, frameID: int) -> int:
        """Get the internal ID of a frameID."""
        return self.frameIDs.index(frameID)

    def get_orderwise_values(self, frameID):
        inds = self.get_index_of_frameID(frameID)
        return (
            self._Rv_orderwise[inds, :],
            self._RvErrors_orderwise[inds, :],
            self._OrderStatus.get_status_from_order(frameID=frameID, all_orders=True),
        )

    def reset_epoch(self, frameID: int) -> None:
        """Fully reset the orderwise information of a given frameID."""
        index = self.get_index_of_frameID(frameID)
        self._OrderStatus.reset_state_of_frameIDs([frameID])
        self._Rv_orderwise[index, :] = np.nan
        self._RvErrors_orderwise[index, :] = np.nan

    def set_merged_mode(self, orders_to_skip: list[int]) -> None:
        self._OrderStatus.add_flag_to_order(
            order=orders_to_skip,
            all_frames=True,
            order_flag=ORDER_SKIP("Skiped due to merged mode"),
        )

    def update_skip_reason(self, orders: list[int] | set[int] | int, skip_reason: Flag) -> None:
        if len(orders) == 0:
            return

        if isinstance(orders, set):
            orders = list(orders)

        self._OrderStatus.add_flag_to_order(order=orders, order_flag=skip_reason, all_frames=True)

    def frame_mimic_status(self, DataClassProxy: DataClass):  # noqa: N803
        for frameID in self.frameIDs:  # noqa: N806
            status = DataClassProxy.get_frame_by_ID(frameID).OrderWiseStatus
            self._OrderStatus.mimic_status(frameID, status)

    def add_new_epochs(self, N_epochs: int, frameIDs: list[int]) -> None:  # noqa: N803
        """Add a new epoch in the status.

        Args:
            N_epochs (int): Number of epochs to be added, must be > 0
            frameIDs (list[int]): FrameID of the provided observations

        Raises:
            custom_exceptions.InvalidConfiguration: If N_epochs < 0
            custom_exceptions.InvalidConfiguration: If number of epochs is not the same as the one of frmaeIDs
            custom_exceptions.InvalidConfiguration: If one of the frameIDs already exists

        """
        if N_epochs <= 0:
            msg = "Can't add negative rows"
            logger.critical(msg)
            raise custom_exceptions.InvalidConfiguration(msg)
        if len(frameIDs) != N_epochs:
            msg = "Number of frameIDs doesn't match the provided number of epochs"
            logger.critical(msg)
            raise custom_exceptions.InvalidConfiguration(msg)

        for entry in frameIDs:
            if entry in self.frameIDs:
                msg = "Adding a repeat of the same frameID"
                raise custom_exceptions.InvalidConfiguration(msg)

        n_orders = self._OrderStatus.shape[1]
        for _ in range(N_epochs):
            new_line = np.zeros(n_orders, dtype=self._Rv_orderwise.dtype)
            self._OrderStatus = np.vstack([self._OrderStatus, new_line])
        self.frameIDs.extend(frameIDs)

    def store_order_data(self, frameID: int, order: int, RV: float, error: float, status: Flag) -> None:
        epoch = self.get_index_of_frameID(frameID)
        self._Rv_orderwise[epoch][order] = RV
        self._RvErrors_orderwise[epoch][order] = error
        self._OrderStatus.add_flag_to_order(order=order, frameID=frameID, order_flag=status)

    @property
    def N_obs(self) -> int:
        """Return the total number of observations."""
        return self._Rv_orderwise.shape[0]

    @property
    def data(self):
        return (
            self._Rv_orderwise.copy(),
            self._RvErrors_orderwise.copy(),
            copy.deepcopy(self._OrderStatus),
        )

    def run_cromatic_interval_optimization(self, N_intervals=3, min_number_orders=10):
        """Optimize the cromatic intervals using the order-wise RV precision.

        Args:
            N_intervals (int, optional): Number of intervals to optimize. Defaults to 3.
            min_number_orders (int, optional): Minimum number of order in each interval. Defaults to 10.

        Returns:
            Table/None: tabletexifier.Table with the results if everything went well. Otherwise, defaults to a None

        """
        precision_array = self._RvErrors_orderwise
        problem_orders = self.problematic_orders

        valid_orders = [i for i in range(self.N_orders) if i not in problem_orders]
        final_arr = precision_array[:, valid_orders]
        tab = None

        try:
            result, intervals = optimize_intervals_over_array(
                list_of_orders=valid_orders,
                array_of_precisions=final_arr,
                N_intervals=N_intervals,
                min_interval_size=min_number_orders,
            )

            tab = convert_to_tab(
                orders_to_run=valid_orders,
                result=result,
                intervals=intervals,
                precision_array=final_arr,
            )
        except custom_exceptions.InvalidConfiguration:
            logger.critical("Not enough orders to generate cromatic intervals")
        except Exception as e:
            logger.critical(f"Found unknown error: {e}")
        return tab

    @property
    def problematic_orders(self) -> set:
        """Get the orders that should be discarded when computing RVs.

        Returns
        -------
        [type]
            [description]

        """
        return self._OrderStatus.common_bad_orders

    def store_to_disk(
        self,
        path_to_store: UI_PATH,
        associated_subInst: str,
        store_flags: bool = True,
        extra_identifier: None | str = None,
    ) -> None:
        header = fits.Header()

        hdu = fits.PrimaryHDU(data=[], header=header)
        hdu_RVs = fits.ImageHDU(data=self._Rv_orderwise, header=header, name="ORDERWISE_RV")
        hdu_ERR = fits.ImageHDU(data=self._RvErrors_orderwise, header=header, name="ORDERWISE_ERR")
        hdu_mask = fits.ImageHDU(
            data=self._OrderStatus.as_boolean().astype(int),
            header=header,
            name="GOOD_ORDER_MASK",
        )
        information = {"FrameID": self.frameIDs}
        coldefs = []
        for key, array in information.items():
            coldefs.append(fits.Column(name=key, format="D", array=array))
        hdu_timeseries = fits.BinTableHDU.from_columns(coldefs, name="TIMESERIES_DATA")

        hdul = fits.HDUList([hdu, hdu_timeseries, hdu_RVs, hdu_ERR, hdu_mask])
        extra = f"_{extra_identifier}" if extra_identifier is not None else ""
        storage_path = build_filename(
            path_to_store,
            f"OrderWiseInfo{extra}_{associated_subInst}",
            fmt="fits",
        )
        hdul.writeto(storage_path, overwrite=True)

        storage_path = build_filename(
            path_to_store,
            f"DetailedFlags{extra}_{associated_subInst}",
            fmt="json",
        )

        with open(storage_path, mode="w") as file:
            json.dump(self._OrderStatus.to_json(), file, indent=4)

    @classmethod
    def load_from_disk(cls, subInst_path, SBART_version: str | None = None, full_path=None) -> OrderWiseRVs:
        if full_path is None:
            subInst = subInst_path.stem
            storage_path = build_filename(
                subInst_path / "RVcube",
                filename=f"OrderWiseInfo_{subInst}",
                fmt="fits",
                SBART_version=SBART_version,
            )
        else:
            storage_path = Path(full_path)
        with fits.open(storage_path) as hdu:
            orderwise_RV = hdu["ORDERWISE_RV"].data
            orderwise_RV_ERR = hdu["ORDERWISE_ERR"].data
            good_order_mask = hdu["GOOD_ORDER_MASK"].data
            timeseries_table = hdu["TIMESERIES_DATA"].data

        new_comp = OrderWiseRVs(
            frameIDs=timeseries_table["FrameID"].astype(int).tolist(), N_orders=orderwise_RV.shape[1]
        )

        new_comp._Rv_orderwise = orderwise_RV  # noqa: SLF001
        new_comp._RvErrors_orderwise = orderwise_RV_ERR  # noqa: SLF001

        for epoch_index, frameID in enumerate(timeseries_table["FrameID"]):
            for order, order_bool_status in enumerate(good_order_mask[epoch_index]):
                if order_bool_status != 1:
                    new_comp._OrderStatus.add_flag_to_order(order=order, order_flag=ORDER_SKIP, frameID=frameID)  # noqa: SLF001
        return new_comp
