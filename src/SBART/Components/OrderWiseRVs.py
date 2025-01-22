from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from astropy.io import fits
from loguru import logger

from SBART.utils import custom_exceptions
from SBART.utils.BASE import BASE
from SBART.utils.paths_tools import build_filename
from SBART.utils.status_codes import ORDER_SKIP, Flag, OrderStatus

if TYPE_CHECKING:
    from SBART.utils.SBARTtypes import UI_PATH


@dataclass
class OrderWiseRVs(BASE):
    frameIDs: list[int]
    N_orders: int

    def __post_init__(self):
        self._Rv_orderwise = np.zeros((self.N_epochs, self.N_orders)) + np.nan
        self._RvErrors_orderwise = np.zeros((self.N_epochs, self.N_orders)) + np.nan

        self._OrderStatus = OrderStatus(N_orders=self.N_orders, frameIDs=self.frameIDs)

    def get_index_of_frameID(self, frameID: int) -> int:
        """Get the internal ID of a frameID."""
        return self._stored_frameIDs.index(frameID)

    def reset_epoch(self, frameID: int) -> None:
        """Fully reset the orderwise information of a given frameID."""
        index = self.get_index_of_frameID(frameID)
        self._OrderStatus.reset_state_of_frameIDs([frameID])
        self._Rv_orderwise[index, :] = np.nan
        self._RvErrors_orderwise[index, :] = np.nan

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
            if entry in self._stored_frameIDs:
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

    def store_to_disk(self, path_to_store: UI_PATH) -> None:
        header = fits.Header()

        hdu = fits.PrimaryHDU(data=[], header=header)
        hdu_RVs = fits.ImageHDU(data=self.orderwiseRvs, header=header, name="ORDERWISE_RV")
        hdu_ERR = fits.ImageHDU(data=self.orderwiseErrors, header=header, name="ORDERWISE_ERR")
        hdu_mask = fits.ImageHDU(
            data=self._OrderStatus.as_boolean().astype(int),
            header=header,
            name="GOOD_ORDER_MASK",
        )
        coldefs = {"FrameID": self.frameIDs}

        hdu_timeseries = fits.BinTableHDU.from_columns(coldefs, name="TIMESERIES_DATA")

        hdul = fits.HDUList([hdu, hdu_timeseries, hdu_RVs, hdu_ERR, hdu_mask])
        storage_path = build_filename(
            path_to_store,
            f"OrderWiseInfo_{self._associated_subInst}",
            fmt="fits",
        )
        hdul.writeto(storage_path, overwrite=True)

    @classmethod
    def load_from_disk(
        cls,
        subInst_path,
        SBART_version: str | None = None,
    ) -> OrderWiseRVs:
        subInst = subInst_path.stem

        storage_path = build_filename(
            subInst_path / "RVcube",
            filename=f"OrderWiseInfo_{subInst}",
            fmt="fits",
            SBART_version=SBART_version,
        )
        with fits.open(storage_path) as hdu:
            orderwise_RV = hdu["ORDERWISE_RV"].data
            orderwise_RV_ERR = hdu["ORDERWISE_ERR"].data
            good_order_mask = hdu["GOOD_ORDER_MASK"].data
            timeseries_table = hdu["TIMESERIES_DATA"].data

        new_comp = OrderWiseRVs(frameIDs=timeseries_table["FrameID"], N_orders=orderwise_RV.shape[1])

        new_comp._Rv_orderwise = orderwise_RV  # noqa: SLF001
        new_comp._RvErrors_orderwise = orderwise_RV_ERR  # noqa: SLF001

        for epoch_index, frameID in enumerate(timeseries_table["FrameID"]):
            for order, order_bool_status in enumerate(good_order_mask[epoch_index]):
                if order_bool_status != 1:
                    new_comp._OrderStatus.add_flag_to_order(order=order, order_flag=ORDER_SKIP, frameID=frameID)  # noqa: SLF001
        return new_comp
