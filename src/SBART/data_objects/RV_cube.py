from __future__ import annotations

import copy
import os
import time
import warnings
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import ujson as json
from astropy.io import fits
from loguru import logger
from tabletexifier import Table

from SBART import __version__
from SBART.Components.OrderWiseRVs import OrderWiseRVs
from SBART.data_objects.Target import Target
from SBART.DataUnits import available_data_units
from SBART.utils import custom_exceptions
from SBART.utils.BASE import BASE
from SBART.utils.choices import DISK_SAVE_MODE
from SBART.utils.custom_exceptions import InvalidConfiguration, NoDataError
from SBART.utils.math_tools.weighted_std import wstd
from SBART.utils.paths_tools import build_filename
from SBART.utils.status_codes import Flag, OrderStatus, Status
from SBART.utils.units import (
    centimeter_second,
    convert_data,
    kilometer_second,
    meter_second,
)
from SBART.utils.work_packages import Package

if TYPE_CHECKING:
    from SBART.Base_Models.UnitModel import UnitModel
    from SBART.data_objects.DataClass import DataClass
    from SBART.utils.SBARTtypes import RV_measurement


class RV_cube(BASE):
    """The RV cube stores the result of a SBART run of a given sub-Instrument.

    It is also responsible for:

    - Provide a way of accessing SBART and /or CCF results
    - Apply RV corrections (drift, SA)
    - Store outputs to disk
    - Create plots and (crude) statistical analysis of the RV timeseries.

    """

    def __init__(
        self,
        subInst: str,
        valid_frameIDs: list[int],
        instrument_properties: dict,
        has_orderwise_rvs,
        is_SA_corrected: bool,
        storage_mode: str,
        disable_SA_computation: bool = False,
        invalid_frameIDs: None | list[int] = None,
    ):
        """It contains:
            - the SA correction value (and applies it)
            - THe (single) drift correction (And applies it)
            - The orderwise RVs, errors and Status
            - The final RV and errors (both raw and corrected)

        Todo:
            - [ ] add check to see if we actually have data in the order-wise arrays!
            - [ ] Implement eniric interface

        """
        self.is_SA_corrected = is_SA_corrected
        self._associated_subInst = subInst
        self._storage_mode = storage_mode
        self._disable_SA_computation = disable_SA_computation
        invalid_frameIDs = invalid_frameIDs if invalid_frameIDs is not None else []
        self._all_frames = [*valid_frameIDs, *invalid_frameIDs]
        self.QC_flag: list[int] = [1 for _ in valid_frameIDs] + [0 for _ in invalid_frameIDs]

        super().__init__(
            user_configs={},
            needed_folders={"plots": "plots", "metrics": "metrics", "RVcube": "RVcube"},
        )
        self.instrument_properties = instrument_properties

        N_epochs = len(valid_frameIDs)
        N_orders = instrument_properties["array_size"][0]

        self.worker_outputs = []

        # RVs and uncertainties -> unitless values in meter/second
        self.orderwise_rvs = OrderWiseRVs(frameIDs=valid_frameIDs, N_orders=N_orders)

        self._extra_storage_units = []
        self.has_orderwise_rvs = has_orderwise_rvs

        self._drift_corrected = instrument_properties[
            "is_drift_corrected"
        ]  # if True then a drift correction will be applied to the data

        self.TM_RVs = [np.nan * meter_second for _ in range(N_epochs)]
        self.TM_RVs_ERR = [np.nan * meter_second for _ in range(N_epochs)]

        self.sBART_version = __version__

        # NOTE: must also add any **new** key to the load_from_disk function!
        needed_keys = [
            "BJD",
            "MJD",
            "drift",
            "drift_ERR",
            "BERV",
            "ISO-DATE",
            "DRS_RV",
            "DRS_RV_ERR",
            "previous_SBART_RV",
            "previous_SBART_RV_ERR",
            "date_folders",
            "bare_filename",
            "CONTRAST",
            "CONTRAST_ERR",
            "BIS SPAN",
            "BIS SPAN_ERR",
            "FWHM",
            "FWHM_ERR",
            "INS MODE",
            "INS NAME",
            "PROG ID",
            "DATE_NIGHT",
            "DRS-VERSION",
        ]
        self._time_key = None
        self.cached_info = {key: [] for key in needed_keys}

        self._loaded_inst_info = False

        self._mode = "individual"

        # TODO: merge GOOD and BAD frames, to ease access to the cached_info

        # Still be to improved:
        self.eniric_RV_precision = []
        self.expected_RV_precision = []
        self.template_RV_precision = []

        self._saved_to_disk = False

    @property
    def frameIDs(self) -> list[int]:
        inds = np.where(np.asarray(self.QC_flag) == 1)
        return [self._all_frames[i] for i in inds[0]]

    @property
    def _invalid_frameIDs(self) -> list[int]:
        inds = np.where(np.asarray(self.QC_flag) == 0)
        return [self._all_frames[i] for i in inds[0]]

    #################################
    # Update internal data
    #################################
    def load_data_from(self, other) -> None:
        """Load the information that is stored in a different RV_cube

        Parameters
        ----------
        other : RV_cube
            [description]

        """
        logger.info("RV cube making copy of {}", other.name)
        self.cached_info = other.cached_info

        self.orderwise_rvs._Rv_orderwise, self.orderwise_rvs._RvErrors_orderwise, self.orderwise_rvs._OrderStatus = (
            other.data
        )
        self._loaded_inst_info = True
        self._saved_to_disk = False

    def add_extra_storage_unit(self, new_unit: UnitModel, generate_root=True):
        logger.info(f"Adding a new storage unit {new_unit.name}")
        if generate_root:
            new_unit.generate_root_path(self._internalPaths.get_path_to("RVcube"))
        self._extra_storage_units.append(new_unit)

    def set_merged_mode(self, orders_to_skip: list[int]) -> None:
        self._mode = "merged_subInst"
        self.orderwise_rvs.set_merged_mode(orders_to_skip)

    def update_skip_reason(self, orders: list[int] | set[int] | int, skip_reason: Flag) -> None:
        self.orderwise_rvs.update_skip_reason(orders, skip_reason)

    def ingest_dataClass_from_rolling(self, DataClassProxy: DataClass):
        good_frames = DataClassProxy.get_valid_frameIDS()
        bad_frames = DataClassProxy.get_invalid_frameIDs()

        for QCstate, framelist in [(1, good_frames), (0, bad_frames)]:
            for frameID in framelist:
                if frameID not in self._all_frames:
                    # Everything is good, we append to the end as it is a new OB
                    self._all_frames.append(frameID)
                    self.QC_flag.append(QCstate)
                    for key in self.cached_info:
                        if key in ["date_folders", "bare_filename"]:
                            continue

                        self.cached_info[key].append(DataClassProxy.get_KW_from_frameID(key, frameID))

                    frame = DataClassProxy.get_frame_by_ID(frameID)
                    self.cached_info["date_folders"].append(frame.file_path)
                    self.cached_info["bare_filename"].append(frame.bare_fname)
                else:
                    # Replacing an OB with a good one. Updating its values
                    index = self._all_frames.index(frameID)
                    self.QC_flag[index] = QCstate
                    for key in self.cached_info:
                        if key in [
                            "date_folders",
                            "bare_filename",
                            "target",
                            "SA_correction",
                            "previous_SBART_RV",
                            "previous_SBART_RV_ERR",
                        ]:
                            continue

                        self.cached_info[key][index] = DataClassProxy.get_KW_from_frameID(key, frameID)

                    frame = DataClassProxy.get_frame_by_ID(frameID)
                    self.cached_info["date_folders"][index] = frame.file_path
                    self.cached_info["bare_filename"][index] = frame.bare_fname

    def load_data_from_DataClass(self, DataClassProxy: DataClass) -> None:
        logger.debug("{} loading frame information from dataclass", self.name)
        for frameID in self._all_frames:
            for key in self.cached_info:
                if key in ["date_folders", "bare_filename"]:
                    continue

                self.cached_info[key].append(DataClassProxy.get_KW_from_frameID(key, frameID))

            frame = DataClassProxy.get_frame_by_ID(frameID)
            self.cached_info["date_folders"].append(frame.file_path)
            self.cached_info["bare_filename"].append(frame.bare_fname)

        self.cached_info["target"] = DataClassProxy.get_Target()
        self._loaded_inst_info = True
        self.orderwise_rvs.frame_mimic_status(DataClassProxy=DataClassProxy)

    def update_worker_information(self, worker_info: list):
        self.worker_outputs = worker_info

    def update_computed_precision(self, expected_precision, eniric_precision, eniric_template):
        """Temporary measure to store the expected precision values. To be replaced by the Frame interface!

        Parameters
        ----------
        expected_precision : [type]
            [description]
        eniric_precision : [type]
            [description]

        """
        for epoch in range(self._RvErrors_orderwise.shape[0]):
            self.eniric_RV_precision.append(eniric_precision[epoch])
            self.expected_RV_precision.append(expected_precision[epoch])
            self.template_RV_precision.append(eniric_template[epoch])

    def update_computed_RVS(self, epoch_rv, epoch_error) -> None:
        if len(epoch_rv) != len(self.TM_RVs):
            logger.critical(
                "{} attempting to store a different number of measurements: {} / {}",
                self.name,
                len(epoch_rv),
                len(self.TM_RVs),
            )
            raise InvalidConfiguration

        self.TM_RVs = epoch_rv
        self.TM_RVs_ERR = epoch_error

    def store_final_RV(self, frameID, epoch_rv, epoch_error) -> None:
        epoch_index = self.frameIDs.index(frameID)
        self.TM_RVs_ERR[epoch_index] = epoch_error
        self.TM_RVs[epoch_index] = epoch_rv

    def store_order_data(self, frameID: int, order: int, RV: float, error: float, status: Flag) -> None:
        self.orderwise_rvs.store_order_data(frameID=frameID, order=order, status=status, RV=RV, error=error)

    def add_new_frame(
        self,
        frameID: int,
        RVs: Iterable[float],
        errors: Iterable[float],
        status: Iterable[Flag],
    ) -> None:
        self.orderwise_rvs.add_new_epochs(N_epochs=1, frameIDs=[frameID])
        orders = list(range(len(RVs)))
        for o, r, e, s in zip(orders, RVs, errors, [i.all_flags for i in status]):
            self.store_order_data(frameID=frameID, order=o, RV=r, error=e, status=s)

    def overwrite_existing_frame(
        self,
        frameID: int,
        RVs: Iterable[float],
        errors: Iterable[float],
        status: Iterable[Flag],
    ) -> None:
        self.orderwise_rvs.reset_epoch(frameID=frameID)
        orders = list(range(len(RVs)))
        for o, r, e, s in zip(orders, RVs, errors, [i.all_flags for i in status]):
            self.store_order_data(frameID=frameID, order=o, RV=r, error=e, status=s)

    # #################################3
    #
    #   Corrections
    # #################################

    def get_RV_timeseries(
        self,
        which: str,
        apply_SA_corr: bool,
        as_value: bool,
        units=None,
        apply_drift_corr=None,
        include_invalid_frames: bool = False,
    ) -> tuple[list, list, list]:
        """Return the RV timeseries.

        Parameters
        ----------
        which : str
            Name of the RV series: TM/previous
        apply_SA_corr : bool
            Apply the SA correction
        as_value : bool
            Convert from astropy units to "unitless" variable
        units : astropy.units, optional
            If not None, convert to this unit. By default None
        apply_drift_corr : [type], optional
            Force the application of the drift correction to be this var (TRUE/FALSE). This ignores
            the default settings for the instrument! By default None

        Returns
        -------
        list: BJD of the observation
        list: list of all RVs
        list: list of the RV uncertainties

        Raises
        ------
        InvalidConfiguration
            [description]
        InvalidConfiguration
            [description]

        """
        if not self._loaded_inst_info:
            msg = "RV cube did not load the information from the instrument!"
            logger.critical(msg)
            raise InvalidConfiguration(msg)

        if which == "SBART":
            final_RVs, final_RVs_ERR = self.get_raw_TM_RVs()
        elif which == "DRS":
            final_RVs, final_RVs_ERR = self.get_raw_DRS_RVs()
        elif which == "previous_SBART":
            final_RVs = self.cached_info["previous_SBART_RV"]
            final_RVs_ERR = self.cached_info["previous_SBART_RV_ERR"]
        else:
            logger.critical(f"which = {which} is not supported by get_RV_timeseries")
            raise InvalidConfiguration
        final_RVs = copy.copy(final_RVs)
        final_RVs_ERR = copy.copy(final_RVs_ERR)
        correct_drift = not self._drift_corrected if apply_drift_corr is None else apply_drift_corr

        if correct_drift:
            logger.info("Cleaning RVs of {} from the drift", self._associated_subInst)
            corrected_rv = []
            corrected_err = []
            for i, raw_rv in enumerate(final_RVs):
                corrected_rv.append(raw_rv - self.cached_info["drift"][i])
                corrected_err.append(np.sqrt(final_RVs_ERR[i] ** 2 + self.cached_info["drift_ERR"][i] ** 2))

            final_RVs = corrected_rv
            final_RVs_ERR = corrected_err

        if apply_SA_corr:
            if self.is_SA_corrected:
                msg = "Applying the SA correction to fdata that is already corrected from it"
                logger.critical(msg)
                raise custom_exceptions.InvalidConfiguration(msg)
            SA_corr = self.compute_SA_correction()
            final_RVs = [final_RVs[i] - SA_corr[i] for i in range(len(final_RVs))]

        output_times = copy.copy(self.obs_times)

        if include_invalid_frames and which == "SBART":
            # SBART data and we want to include invalid -> we must populate with invalids

            bad_rvs = [-99_999 * meter_second for _ in self._invalid_frameIDs]
            bad_err = [-99_999 * meter_second for _ in self._invalid_frameIDs]
            output_times = self.cached_info[self.time_key]
            final_RVs.extend(bad_rvs)
            final_RVs_ERR.extend(bad_err)
        elif which not in ["SBART", "previous_SBART"] and not include_invalid_frames:
            # DRS / previous sbART and don't want to include invalid -< we must remove
            inds = np.where(np.asarray(self.QC_flag) == 0)[0]
            for index in sorted(inds)[::-1]:
                final_RVs.pop(index)
                final_RVs_ERR.pop(index)
        return output_times, convert_data(final_RVs, units, as_value), convert_data(final_RVs_ERR, units, as_value)

    def get_RV_from_ID(
        self,
        frameID: int,
        which: str,
        apply_SA_corr: bool,
        as_value: bool,
        units,
        apply_drift_corr=None,
    ):
        """Retrieve the BJD, RV and RV_ERR from a given frameID.

        Parameters
        ----------
        frameID : int
            frameID
        which : str
            TM/previous
        apply_SA_corr : bool
            apply the SA correction
        as_value : bool
            convert from astropy.units to value
        units : [type]
            units to convert the original measurements
        apply_drift_corr : [type], optional
            For the drift correction to be enable/disabled. If None, use the default
            for the instrument. By default None

        Returns
        -------
        [type]
            [description]

        """
        if frameID in self._invalid_frameIDs:
            # Should we send out the
            ID_index = len(self.frameIDs) + self._invalid_frameIDs.index(frameID)
            rv = -99_999 * meter_second
            err = -99_999 * meter_second
            return self.cached_info[self.time_key][ID_index], *convert_data(
                [rv, err], new_units=units, as_value=as_value
            )

        times, rvs, uncerts = self.get_RV_timeseries(
            which=which,
            apply_SA_corr=apply_SA_corr,
            as_value=as_value,
            units=units,
            apply_drift_corr=apply_drift_corr,
        )
        ID_index = self.frameIDs.index(frameID)

        return times[ID_index], rvs[ID_index], uncerts[ID_index]

    def compute_SA_correction(self) -> list[RV_measurement[meter_second]]:
        """Compute the SA correction for each point. Returns zeros if the SA correction is disabled

        Returns:
            list[RV_measurement[meter_second]]: list with the SA value for each point
        """

        if "SA_correction" in self.cached_info:
            return self.cached_info["SA_correction"]

        if self._disable_SA_computation:
            self.cached_info["SA_correction"] = [0 * meter_second for _ in self.obs_times]
        else:
            logger.info("Starting SA correction")

            SA = self.cached_info["target"].secular_acceleration

            min_time = 55500  # always use the same reference frame
            logger.info("Setting SA reference frame to BJD = {}", min_time)

            secular_correction = [SA * (OBS_time - min_time) / 365.25 for OBS_time in self.obs_times]

            self.cached_info["SA_correction"] = secular_correction

        return self.cached_info["SA_correction"]

    def get_storage_unit(self, storage_name):
        for unit in self._extra_storage_units:
            if unit.is_storage_type(storage_name):
                return unit

        msg = f"Storage unit {storage_name} does not exist ({len(self._extra_storage_units)} available)"
        logger.critical(msg)
        raise NoDataError(msg)

    def merge_tm_activity_units(self, name: str, unit):
        self.get_storage_unit(name).merge_with(unit)

    ##########################
    #
    # Access data
    #

    def get_raw_TM_RVs(self) -> tuple[list, list]:
        return self.TM_RVs, self.TM_RVs_ERR

    def get_raw_DRS_RVs(self) -> tuple[list, list]:
        return self.cached_info["DRS_RV"], self.cached_info["DRS_RV_ERR"]

    def get_TM_activity_indicator(self, act_name):
        """Get the DLW measurements if they exist.Otherwise return array of nans

        Parameters
        ----------
        act_name

        Returns
        -------
        list
            Values
        list
            Uncertainties

        """
        nan_list = [np.nan for _ in self.frameIDs]
        try:
            unit = self.get_storage_unit("Indicators")
        except NoDataError:
            logger.critical("There is no storage unit that stores activity indicators")
            return nan_list, nan_list

        try:
            values, errors = unit.get_combined_measurements(act_name)
        except KeyError:
            values, errors = nan_list, nan_list
        return values, errors

    def get_frame_orderwise_status(self, frameID) -> list[Status]:
        return self.orderwise_rvs._OrderStatus.get_status_from_order(frameID, all_orders=True)

    @property
    def subInst(self) -> str:
        return self._associated_subInst

    def has_data_from_subInst(self, subInst: str) -> bool:
        return self.subInst == subInst

    @property
    def time_key(self) -> str:
        if self._time_key is None:
            found_key = False

            for key in ["BJD", "MJD"]:
                time_list = self.cached_info[key]
                if len(time_list) == 0:
                    continue
                if time_list[0] is not None:
                    found_key = True
                    selected_key = key
                    break

            if not found_key:
                msg = f"{self.name} couldn't find time-related KW with valid values"
                logger.warning(msg)
                raise InvalidConfiguration(msg)

            return selected_key
        return self._time_key

    @property
    def obs_times(self) -> list[float]:
        """Provide a "time" of observation.

        Can either be BJD of MJD (depends on which exists. If both exist,returns the BJD.

        """
        inds = np.where(np.asarray(self.QC_flag) == 1)[0]
        return [self.cached_info[self.time_key][i] for i in inds]

    @property
    def N_orders(self) -> int:
        return self.orderwise_rvs.N_orders

    @property
    def N_obs(self) -> int:
        """Return the total number of observations"""
        return self.orderwise_rvs.N_obs

    @property
    def name(self) -> str:
        return f"RV cube from {self._associated_subInst}"

    @property
    def data(self):
        return self.orderwise_rvs.data

    @property
    def problematic_orders(self) -> set:
        return self.orderwise_rvs.problematic_orders

    ##################################
    #
    # Export data!
    #

    def build_datablock(self, include_invalid_frames: bool = False) -> dict:
        _, raw_rv, raw_err = self.get_RV_timeseries(
            which="SBART",
            apply_SA_corr=False,
            apply_drift_corr=False,
            as_value=True,
            units=kilometer_second,
            include_invalid_frames=include_invalid_frames,
        )

        _, corr_rv, corr_err = self.get_RV_timeseries(
            which="SBART",
            apply_SA_corr=False,
            as_value=True,
            units=kilometer_second,
            include_invalid_frames=include_invalid_frames,
        )
        dlw, dlw_err = self.get_TM_activity_indicator("DLW")
        dlw = copy.copy(dlw)
        dlw_err = copy.copy(dlw_err)
        frameIDs = copy.copy(self.frameIDs)

        if include_invalid_frames:
            # ensure that they array has the same size (no DLW for invalid frames)
            dlw.extend([np.nan for _ in self._invalid_frameIDs])
            dlw_err.extend([np.nan for _ in self._invalid_frameIDs])
            frameIDs.extend(self._invalid_frameIDs)

        out = {
            "RVc": corr_rv,
            "RVc_ERR": corr_err,
            "RV": raw_rv,
            "RV_ERR": raw_err,
            "DLW": dlw,
            "DLW_ERR": dlw_err,
            "frameIDs": list(map(int, frameIDs)),
        }

        tmp = {
            "OBJ": [self.cached_info["target"].true_name for _ in self.obs_times],
            "SA": convert_data(self.compute_SA_correction(), kilometer_second, True),
            "DRIFT": convert_data(self.cached_info["drift"], meter_second, True),
            "DRIFT_ERR": convert_data(self.cached_info["drift_ERR"], meter_second, True),
            "BERV": convert_data(self.cached_info["BERV"], kilometer_second, True),
            "full_path": self.cached_info["date_folders"],
            "filename": [os.path.basename(i) for i in self.cached_info["date_folders"]],
            "QC": list(map(int, self.QC_flag)),
        }

        ind_keys = [f"{a}{b}" for a, b in product(["FWHM", "CONTRAST"], ("", "_ERR"))]

        for key in ["BJD", "MJD", "INS MODE", "INS NAME", "PROG ID", "DATE_NIGHT", "DRS-VERSION", *ind_keys]:
            tmp[key] = self.cached_info[key]
        inds = np.where(self.QC_flag == 1)[0]
        for key, data in tmp.items():
            if include_invalid_frames:
                out[key] = data
            else:
                # Only the first N entries correspond to the valid frames
                out[key] = [out[i] for i in inds]
        return out

    def compute_statistics(self, savefile=True):
        """Compute the scatter and median uncertainty of the different timeseries

        Parameters
        ----------
        main_path : str
            Path to the "base" folder of the outputs
        savefile : bool, optional
            Store to disk if True. By default True

        """
        rv_table = Table(["Method", "std [m/s]", "wstd [m/s]", "median err [m/s]"])
        rv_table.set_decimal_places(5)

        for name in ["DRS", "SBART", "previous_SBART"]:
            _, rvs, uncerts = self.get_RV_timeseries(
                which=name,
                apply_SA_corr=False,
                as_value=True,
                units=meter_second,
                apply_drift_corr=False,
                include_invalid_frames=False,
            )
            row = [name + "_raw", np.std(rvs), wstd(rvs, uncerts), np.median(uncerts)]
            rv_table.add_row(row)
            # SA correction does not enter the RVc column and, consequently, the statistics
            _, rvs, uncerts = self.get_RV_timeseries(
                which=name, apply_SA_corr=False, as_value=True, units=meter_second, include_invalid_frames=False
            )
            row = [name + "_corr", np.std(rvs), wstd(rvs, uncerts), np.median(uncerts)]
            rv_table.add_row(row)

        if savefile:
            file = build_filename(
                self._internalPaths.get_path_to("metrics", as_posix=False),
                f"statistics_{self._associated_subInst}",
                "txt",
            )

            rv_table.write_to_file(file, mode="w", write_LaTeX=True)
        logger.info(rv_table)

    def export_skip_reasons(self, dataClassProxy) -> None:
        final_path = build_filename(
            self._internalPaths.get_path_to("metrics", as_posix=False),
            f"DataRejectionSummary_{self._associated_subInst}_{self._mode}",
            "txt",
        )

        with open(final_path, mode="w") as file:
            order_skip_reasons = {"Warnings": {}, "Rejections": {}}
            file_skip_reasons = {"Warnings": {}, "Rejections": {}}

            file.write("Summary of data rejection:")

            file.write(
                f"\n\tRejected {len(self.problematic_orders)} out of {self.orderwise_rvs.N_orders} available orders:",
            )
            file.write(f"\n\tCommon orders removed:\n{self.problematic_orders}\n")

            file.write("\nFrame-Wise analysis:")
            stellar_template = dataClassProxy.get_stellar_template(self._associated_subInst)
            for current_frameID in dataClassProxy.get_frameIDs_from_subInst(
                self._associated_subInst,
                include_invalid=False,
            ):  # self.frameIDs:
                fpath = dataClassProxy.get_filename_from_frameID(current_frameID)
                file.write(f"\n\tFrame {fpath} ({dataClassProxy.get_KW_from_frameID('ISO-DATE', current_frameID)}):\n")
                if not stellar_template.was_loaded:
                    file.write(
                        f"\n\t\tIn Stellar Template: {stellar_template.check_if_used_frameID(current_frameID)}\n",
                    )

                current_Frame = dataClassProxy.get_frame_by_ID(current_frameID)
                lines, frame_skip = current_Frame.status.description(indent_level=2)
                file.write("".join(lines) + "\n")

                if current_frameID in self.frameIDs:
                    # Valid frame -> will include bad orders from the RV extraction
                    lines, frame_orderskip_reasons = self.orderwise_rvs._OrderStatus.description(
                        indent_level=2,
                        frameID=current_frameID,
                        include_footer=False,
                        include_header=False,
                    )
                else:
                    # Completelly rejected file -> all info exists on the actual Frame Object
                    (
                        lines,
                        frame_orderskip_reasons,
                    ) = current_Frame.OrderStatus.description(
                        indent_level=2,
                        frameID=current_frameID,
                        include_footer=False,
                        include_header=False,
                    )
                file.write("".join(lines) + "\n")

                for master_dict, local_dict in [
                    (order_skip_reasons, frame_orderskip_reasons),
                    (file_skip_reasons, frame_skip),
                ]:
                    for master_key in ["Warnings", "Rejections"]:
                        for key, original_dict in local_dict[master_key].items():
                            master_dict[master_key][key] = original_dict

            file.write("\nSummary of Flags:")
            for skip_name, master_dict in [
                ("File Skip", file_skip_reasons),
                ("Order Skip", order_skip_reasons),
            ]:
                file.write(f"\n\t{skip_name}:")
                for master_key in ["Rejections", "Warnings"]:
                    file.write(f"\n\t\t{master_key}:")
                    for flag, description in master_dict[master_key].items():
                        file.write(f"\n\t\t\t{flag}:{description}")

    def export_results(
        self,
        keys: list[str],
        header: list[str],
        dataClassProxy: DataClass,
        text=True,
        rdb=True,
        append=False,
        include_invalid_frames: bool = False,
    ):
        if self._saved_to_disk:
            return

        storage_path = self._internalPaths.root_storage_path

        logger.debug("RV cube storing data under the main folder: {}", storage_path)

        append = self._storage_mode == "rolling"

        if text:
            self.export_txt(header, append=append, keys=keys, include_invalid_frames=include_invalid_frames)
        if rdb:
            self.export_rdb(append)
            self.export_complete_rdb()
        self.export_skip_reasons(dataClassProxy)
        self.compute_statistics()
        if self.disk_save_level != DISK_SAVE_MODE.EXTREME:
            self.plot_RVs(dataClassProxy)
        self._saved_to_disk = True

    def plot_RVs(self, dataClassProxy: DataClass) -> None:
        """Plot & store the RV timeseries.

        Parameters
        ----------
        storage_path : str
            Main storage path

        """
        diagnostics_path = self._internalPaths.get_path_to("plots", as_posix=False)

        fig, ax = plt.subplots(3, 1, sharex=True)
        plot_info = {
            "SBART": {"color": "red", "marker": "x"},
            "DRS": {"color": "black", "marker": "+"},
        }

        figure_list = [fig]
        cache_rvs = []
        for which in ["SBART", "DRS"]:
            # Not including the SA correction in the plots
            bjd, rv, uncerts = self.get_RV_timeseries(
                which,
                apply_SA_corr=False,
                as_value=False,
                units=kilometer_second,
            )
            bjd = np.subtract(bjd, 2450000)
            rv = convert_data(rv, as_value=True, new_units=meter_second)
            cache_rvs.append(rv)
            err = convert_data(uncerts, as_value=True, new_units=meter_second)

            ind_err = convert_data(uncerts, as_value=True, new_units=centimeter_second)
            ax[1].scatter(
                bjd,
                ind_err,
                color=plot_info[which]["color"],
                marker=plot_info[which]["marker"],
                label=which,
            )

            ax[0].errorbar(
                bjd,
                rv,
                err,
                color=plot_info[which]["color"],
                marker=plot_info[which]["marker"],
                label=which,
                linestyle="",
            )

        ax[2].scatter(
            bjd,
            np.subtract(cache_rvs[0], cache_rvs[1]),
            color=plot_info[which]["color"],
            marker=plot_info[which]["marker"],
            label=which,
        )
        ax[2].axhline(0, color="red", linestyle="--", alpha=0.8)
        ax[2].set_ylabel(r"$\Delta$ RV [m/s]")
        ax[-1].set_xlabel("BJD - 2450000[days]")
        ax[0].set_ylabel("RV [m/s]")
        ax[1].set_ylabel(r"$\sigma_{RV}$ [cm/s]")
        ax[0].legend(loc="best")
        plt.tight_layout()

        final_path = build_filename(diagnostics_path, "RV_timeseries", "png")

        fig.savefig(final_path, dpi=300)

        if self.has_orderwise_rvs:
            fig_full, ax_full = plt.subplots(2, 1, sharex=True)
            fig_part, ax_part = plt.subplots(2, 1, sharex=True)

            figure_list.extend([fig_full, fig_part])

            orders = np.asarray(range(self.orderwise_rvs._Rv_orderwise.shape[1]))
            for epoch, data in enumerate(self.orderwise_rvs._RvErrors_orderwise):
                full_rvs = self.orderwise_rvs._Rv_orderwise[epoch]

                valid_orders = data.copy()
                valid_RVs = full_rvs.copy()
                valid_orders[self.problematic_orders] = np.nan
                valid_RVs[self.problematic_orders] = np.nan

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ax_full[0].errorbar(
                        orders[self.problematic_orders],
                        full_rvs[self.problematic_orders] - np.nanmedian(valid_RVs),
                        data[self.problematic_orders],
                        marker="o",
                        linestyle="",
                        alpha=0.3,
                    )

                    # TODO: understand why this gives UserWarning: Warning: converting a masked element to nan.
                    ax_full[0].errorbar(
                        orders,
                        full_rvs - np.nanmedian(valid_RVs),
                        data,
                        marker="o",
                        linestyle="",
                    )

                    ax_full[1].plot(
                        data[self.problematic_orders],
                        marker="o",
                        linestyle="",
                        alpha=0.3,
                    )
                    ax_full[1].plot(valid_orders, marker="x", linestyle="")

                    centered_RVs = valid_RVs - np.nanmedian(valid_RVs)

                    ax_part[0].errorbar(orders, centered_RVs, valid_orders, marker="o", linestyle="")

                    ax_part[1].plot(valid_orders, marker="x", linestyle="")

            for ax in [ax_full, ax_part]:
                ax[0].set_ylabel("OrderWise RVs")
                ax[1].set_ylabel(r"$\sigma_{RV}$")
                ax[1].set_xlabel("Order")

                ax[1].set_xlim([orders[0] - 1, orders[-1] + 1])
                ax[1].set_xticks(list(map(int, np.linspace(orders[0], orders[-1], 20))))

            final_path = build_filename(diagnostics_path, "RV_raw_orderwise_errors", "png")
            fig_full.tight_layout()
            fig_full.savefig(final_path)

            final_path = build_filename(diagnostics_path, "RV_orderwise_errors", "png")
            fig_part.tight_layout()
            fig_part.savefig(final_path)

            fig, axis = plt.subplots()
            figure_list.append(fig)
            times = self.obs_times
            sorted_IDs = np.asarray(self.frameIDs)[np.argsort(times)].tolist()
            empty_array = np.zeros((self.N_orders, len(times)))

            for pkg in self.worker_outputs:
                for order_pkg in pkg:
                    frameID = order_pkg["frameID"]
                    order = order_pkg["order"]
                    empty_array[order, sorted_IDs.index(frameID)] = order_pkg["Total_Flux_Order"]
            empty_array /= np.max(empty_array, axis=1)[:, None]  # normalize across the orders

            fig, ax = plt.subplots(figsize=(20, 10), constrained_layout=True)
            figure_list.append(fig)
            data = ax.imshow(empty_array.T)
            ax.set_xlabel("Spectral order")
            ax.set_ylabel("Observation number")
            ax.set_yticklabels([])
            ax.set_title(r"$\sum_i Spectra(\lambda_i)$")
            fig.colorbar(data)
            final_path = build_filename(diagnostics_path, "OrderwiseFlux", "png")
            fig.savefig(final_path)

        try:
            unit = self.get_storage_unit("Indicators")
            fig, ax = plt.subplots(2, 2, sharey=True, figsize=(20, 10), constrained_layout=True)
            figure_list.append(fig)
            dlw, err = unit.get_combined_measurements("DLW")
            contrast = dataClassProxy.collect_KW_observations("CONTRAST", [self.subInst])
            FWHM = dataClassProxy.collect_KW_observations("FWHM", [self.subInst])
            BIS = dataClassProxy.collect_KW_observations("BIS SPAN", [self.subInst])
            # print(len(dlw))
            # breakpoint()
            print(len(self.obs_times), len(dlw))
            ax[0, 0].errorbar(self.obs_times, dlw, err, ls="", marker="x")
            ax[0, 0].set_xlabel("BJD")
            ax[1, 0].scatter(FWHM, dlw)
            ax[1, 0].set_xlabel("FWHM")

            ax[0, 1].errorbar(contrast, dlw, err)
            ax[0, 1].set_xlabel("CONTRAST")
            ax[1, 1].errorbar(BIS, dlw, err)
            ax[1, 1].set_xlabel("BIS SPAN")
            final_path = build_filename(diagnostics_path, "DLW_correlations", "png")
            fig.savefig(final_path, dpi=300)

            fig, ax = plt.subplots(1, 1, sharey=True, figsize=(20, 10), constrained_layout=True)

            chroma_val, chroma_err = unit.get_all_orderwise_indicator("DLW")
            for order_info, errors in zip(chroma_val, chroma_err):
                ax.scatter(list(range(self.N_orders)), order_info)
            final_path = build_filename(diagnostics_path, "DLW_orderwise", "png")
            fig.savefig(final_path, dpi=300)

        except NoDataError:
            pass

        logger.debug("Closing figures from {}", self.name)
        for figure in figure_list:
            plt.close(figure)

    def export_txt(self, header, keys, append=False, include_invalid_frames: bool = False):
        """Export the data to a text file

        Parameters
        ----------
        path : str
            Path in which the file will be created
        append : bool, optional
            instead of creating a new file, append to a previously existing one, by default False
        add_to_filename : str, optional
            If it is not None, then add to the beginning of the filename, by default None
        remove_header : bool, optional
            Avoid creating a header [with the description of each collumn] in the stored file, by default False
        detailed_table : bool, optional
            write a detailed table with all of the information of the different corrections, by default True

        """
        data_blocks = self.build_datablock(include_invalid_frames=include_invalid_frames)
        final_path = build_filename(
            self._internalPaths.root_storage_path,
            f"RVs_{self._associated_subInst}_{self._mode}",
            "txt",
        )

        mode = "a" if append else "w"
        table = Table(header=header, table_style="NoLines")
        for epoch in np.argsort(self.obs_times):
            line = []
            for key in keys:
                line.append(data_blocks[key][epoch])
            table.add_row(line)
        table.set_decimal_places(18)
        table.write_to_file(final_path, mode=mode, write_LaTeX=False)

    def export_rdb(self, append=False):
        star_name = self.cached_info["target"].true_name

        final_path = build_filename(
            self._internalPaths.root_storage_path,
            f"{star_name}_{self._associated_subInst}_{self._mode}",
            "rdb",
        )
        mode = "a" if append else "w"

        with open(final_path, mode=mode) as file:
            if not append:
                file.write("rjd\tvrad\tsvrad\n---\t----\t-----\n")

            obs, rvs, uncerts = self.get_RV_timeseries(
                which="SBART",
                apply_SA_corr=False,
                units=kilometer_second,
                as_value=True,
            )

            for index in np.argsort(obs):
                file.write(f"{obs[index] - 24e5}\t{rvs[index]}\t{uncerts[index]}\n")

    def export_complete_rdb(self) -> None:
        """Export a complete rdb file."""

        data_blocks = self.build_datablock(include_invalid_frames=True)
        star_name = self.cached_info["target"].true_name
        final_path = build_filename(
            self._internalPaths.root_storage_path,
            f"complete_{self._associated_subInst}_{self._mode}",
            "rdb",
        )

        float_cols = [
            "FWHM",
            "FWHM_ERR",
            "CONTRAST",
            "CONTRAST_ERR",
            "BERV",
        ]
        int_cols = ["QC"]

        str_cols = ["DATE_NIGHT", "DRS-VERSION", "PROG ID", "filename", "INS MODE", "INS NAME"]

        cols = [*float_cols, *int_cols, *str_cols]
        prods = self.get_RV_timeseries(
            which="SBART", apply_SA_corr=False, units=kilometer_second, as_value=True, include_invalid_frames=True
        )
        out_array = np.zeros((len(prods[0]), len(cols) + 3), dtype=object)

        for col_index, data in enumerate(prods):
            out_array[:, col_index] = data
        for index, key in enumerate(cols):
            out_array[:, 3 + index] = data_blocks[key]

        cols = ["BJD", "RV", "RV_ERR", *cols]

        header = " ".join(cols) + "\n" + " ".join(["-" * len(i) for i in cols])

        np.savetxt(
            fname=final_path,
            X=out_array,
            fmt=[
                "%.7f",
            ]
            * (3 + len(float_cols))
            + ["%i"] * len(int_cols)
            + ["%s"] * len(str_cols),
            header=header,
        )

    ##
    # Under implementation
    ##

    def run_cromatic_interval_optimization(self, N_intervals=3, min_number_orders=10):
        """Optimize the cromatic intervals using the order-wise RV precision.

        Args:
            N_intervals (int, optional): Number of intervals to optimize. Defaults to 3.
            min_number_orders (int, optional): Minimum number of order in each interval. Defaults to 10.

        Returns:
            Table/None: tabletexifier.Table with the results if everything went well. Otherwise, defaults to a None

        """
        return self.orderwise_rvs.run_cromatic_interval_optimization(
            N_intervals=N_intervals, min_number_orders=min_number_orders
        )

    def trigger_data_storage(self, *args, **kwargs):
        t0 = time.time()
        super().trigger_data_storage(*args, **kwargs)
        try:
            logger.info("Computing optimal intervals for RV extraction")
            storage_path = build_filename(
                self._internalPaths.get_path_to("RVcube", as_posix=False),
                f"Optimal_Intervals_{self._associated_subInst}",
                fmt="txt",
            )
            if self.N_obs < 200 and self.disk_save_level != DISK_SAVE_MODE.EXTREME:
                for N_interval in [2, 3]:
                    tab = self.run_cromatic_interval_optimization(N_intervals=N_interval, min_number_orders=10)
                    if tab is not None:
                        tab.write_to_file(path=storage_path)
            elif self.N_obs >= 200:
                logger.warning("More than 200 observations loaded, skipping optimization of order interval")
        except Exception as e:
            logger.critical(f"Generation of optimal intervals failed due to {e}")

        self._store_OrderWise_to_Fits()
        self._store_misc_info()
        self._store_work_packages()

        for unit in self._extra_storage_units:
            unit.trigger_data_storage()

        tf = time.time() - t0
        logger.info(f"Finished export of {self.name} to disk. Took {tf:.2f} seconds")

    def _store_misc_info(self):
        logger.info("Storing misc information")

        storage_path = build_filename(
            self._internalPaths.get_path_to("RVcube", as_posix=False),
            f"miscInfo_{self._associated_subInst}",
            fmt="json",
        )

        data_out = {
            "cached_info": {"target": self.cached_info["target"].json_ready},
        }
        data_out["cached_info"]["ISO-DATE"] = self.cached_info["ISO-DATE"]
        data_out["cached_info"]["date_folders"] = list([i.as_posix() for i in self.cached_info["date_folders"]])

        data_out["has_orderwise_rvs"] = self.has_orderwise_rvs
        data_out["invalidFrameID"] = self._invalid_frameIDs

        with open(storage_path, mode="w") as file:
            json.dump(data_out, file, indent=4)

    def _store_work_packages(self):
        logger.info("Storing work packages")
        complete_outputs = {"work_packages": []}

        for pkg_list in self.worker_outputs:
            for package in pkg_list:
                complete_outputs["work_packages"].append(package.json_ready())

        storage_path = build_filename(
            self._internalPaths.get_path_to("RVcube", as_posix=False),
            f"WorkPackages_{self._associated_subInst}",
            fmt="json",
        )

        with open(storage_path, mode="w") as file:
            json.dump(complete_outputs, file, indent=4)

    def _store_OrderWise_to_Fits(self):
        logger.info("Storing order-wise information to fits file")

        OBS_date, TM_RV, TM_ERR = self.get_RV_timeseries(
            "SBART",
            apply_SA_corr=False,
            apply_drift_corr=False,
            as_value=True,
            units=meter_second,
            include_invalid_frames=True,
        )
        OBS_date, prev_RV, prev_ERR = self.get_RV_timeseries(
            "DRS",
            apply_SA_corr=False,
            apply_drift_corr=False,
            as_value=True,
            units=meter_second,
            include_invalid_frames=True,
        )

        OBS_date, prev_sbart_RV, prev_sbart_ERR = self.get_RV_timeseries(
            "previous_SBART",
            apply_SA_corr=False,
            apply_drift_corr=False,
            as_value=True,
            units=meter_second,
            include_invalid_frames=True,
        )
        # Ensure that we have something cached for it, even if zeros
        self.compute_SA_correction()
        logger.warning("constructing fits file")
        information = {
            "FrameID": [*self.frameIDs, *self._invalid_frameIDs],
            "QC": self.QC_flag,
            "DRS_RV": prev_RV,
            "DRS_RV_ERR": prev_ERR,
            "prevSBART_RV": prev_sbart_RV,
            "prevSBART_RV_ERR": prev_sbart_ERR,
            "TM_raw": TM_RV,
            "TM_raw_ERR": TM_ERR,
            "DRIFT": convert_data(self.cached_info["drift"], new_units=meter_second, as_value=True),
            "DRIFT_ERR": convert_data(self.cached_info["drift_ERR"], new_units=meter_second, as_value=True),
            "SA": convert_data(self.cached_info["SA_correction"], new_units=meter_second, as_value=True),
            "BERV": convert_data(self.cached_info["BERV"], new_units=meter_second, as_value=True),
        }

        for ind, extra in product(["FWHM", "CONTRAST", "BIS SPAN"], ["", "_ERR"]):
            information[f"{ind}{extra}"] = self.cached_info[f"{ind}{extra}"]
        coldefs = []
        for key, array in information.items():
            coldefs.append(fits.Column(name=key, format="D", array=array))

        for key in ["BJD", "MJD"]:
            array = self.cached_info[key]
            if array[0] is not None:
                coldefs.append(fits.Column(name=key, format="D", array=array))

        hdu_timeseries = fits.BinTableHDU.from_columns(coldefs, name="TIMESERIES_DATA")

        header = fits.Header()
        header["HIERARCH drift_corr"] = self._drift_corrected
        header["VERSION"] = self.sBART_version
        header["mode"] = self._mode
        header["HIERARCH is_SA_corrected"] = self.is_SA_corrected
        header["HIERARCH array_size_0"] = self.instrument_properties["array_size"][0]
        header["HIERARCH array_size_1"] = self.instrument_properties["array_size"][1]
        hdu = fits.PrimaryHDU(data=[], header=header)

        hdul = fits.HDUList([hdu, hdu_timeseries])

        storage_path = build_filename(
            self._internalPaths.get_path_to("RVcube", as_posix=False),
            f"CachedInfo_{self._associated_subInst}",
            fmt="fits",
        )
        hdul.writeto(storage_path, overwrite=True)
        self.orderwise_rvs.store_to_disk(
            path_to_store=self._internalPaths.get_path_to("RVcube", as_posix=False),
            associated_subInst=self._associated_subInst,
        )
        text_info = {}
        str_info = ["DRS-VERSION", "DATE_NIGHT", "PROG ID", "INS MODE", "INS NAME", "bare_filename"]
        for key in str_info:
            text_info[key] = self.cached_info[key]

        text_info["root_folder"] = self._internalPaths.root_storage_path.as_posix()

        storage_path = build_filename(
            self._internalPaths.get_path_to("RVcube", as_posix=False),
            f"CachedInfo_{self._associated_subInst}",
            fmt="json",
        )
        with open(storage_path, mode="w") as tow:
            json.dump(text_info, tow)

    @classmethod
    def load_cube_from_disk(
        cls,
        subInst_path,
        load_full_flag: bool = False,
        load_work_pkgs: bool = False,
        SBART_version: Optional[str] = None,
    ):
        # TODO: load and store the data units!!

        subInst = subInst_path.stem
        misc_filename = build_filename(
            subInst_path / "RVcube",
            filename=f"miscInfo_{subInst}",
            fmt="json",
            SBART_version=SBART_version,
        )
        cachedinfo_filename = build_filename(
            subInst_path / "RVcube",
            filename=f"CachedInfo_{subInst}",
            fmt="fits",
            SBART_version=SBART_version,
        )
        cachedJSONinfo_filename = build_filename(
            subInst_path / "RVcube",
            filename=f"CachedInfo_{subInst}",
            fmt="json",
            SBART_version=SBART_version,
        )
        detailed_flags_filename = build_filename(
            subInst_path / "RVcube",
            filename=f"DetailedFlags_{subInst}",
            fmt="json",
            SBART_version=SBART_version,
        )
        workpackages_filename = build_filename(
            subInst_path / "RVcube",
            filename=f"WorkPackages_{subInst}",
            fmt="json",
            SBART_version=SBART_version,
        )
        with open(misc_filename) as file:
            miscInfo = json.load(file)

        # For backwards compatibility retrieve an empty list

        with fits.open(cachedinfo_filename) as hdu:
            header_info = hdu[0].header
            timeseries_table = hdu["TIMESERIES_DATA"].data

        with open(cachedJSONinfo_filename) as tor:
            timeseries_text = json.load(tor)

        instrument_info = {
            "array_size": [header_info[f"HIERARCH array_size_{i}"] for i in range(2)],
            "is_drift_corrected": header_info["HIERARCH drift_corr"],
        }
        frameIDs = timeseries_table["FrameID"].astype(int).tolist()
        QC = timeseries_table["QC"].astype(int).tolist()

        has_orderwise_rvs = miscInfo["has_orderwise_rvs"]
        inds = np.where(np.asarray(QC) == 1)[0]
        good_frameIDs = [frameIDs[i] for i in inds]
        BADinds = np.where(np.asarray(QC) == 0)[0]
        invalid = [frameIDs[i] for i in BADinds]

        new_cube = RV_cube(
            subInst=subInst,
            valid_frameIDs=good_frameIDs,
            instrument_properties=instrument_info,
            has_orderwise_rvs=has_orderwise_rvs,
            # for backwards compatibility:
            is_SA_corrected=header_info.get("HIERARCH is_SA_corrected", False),
            invalid_frameIDs=invalid,
            storage_mode="one-shot",
        )
        cube_root_folder = subInst
        new_cube.generate_root_path(cube_root_folder)

        logger.debug("Loading misc Info:")
        for key, values in miscInfo["cached_info"].items():
            if key == "date_folders":
                values = list([Path(i) for i in values])
            if key == "target":
                values = Target(["foo"], original_name=values)
            new_cube.cached_info[key] = values

        logger.debug("Loading orderwise info")

        new_cube._mode = header_info["mode"]

        logger.debug("Generating the new order mask")
        orderwise = OrderWiseRVs.load_from_disk(subInst_path=subInst_path, SBART_version=SBART_version)
        new_cube.orderwise_rvs = orderwise
        logger.debug("Loading timeseries data")

        convert_to_quantity = lambda data: [elem * meter_second for elem in data]

        for key in ["BJD", "MJD"]:
            try:
                new_cube.cached_info[key] = timeseries_table[key]
            except KeyError:
                logger.info(f"Key <{key}> does not exist! Skipping it")

        entries = {
            "DRS_RV": "DRS_RV",
            "DRS_RV_ERR": "DRS_RV_ERR",
            "SA_correction": "SA",
            "drift": "DRIFT",
            "drift_ERR": "DRIFT_ERR",
            "BERV": "BERV",
        }

        for internal_kw, storage_kw in entries.items():
            new_cube.cached_info[internal_kw] = convert_to_quantity(timeseries_table[storage_kw])

        indexed_entries = {
            "previous_SBART_RV": "prevSBART_RV",
            "previous_SBART_RV_ERR": "prevSBART_RV_ERR",
        }
        for internal_kw, storage_kw in indexed_entries.items():
            new_cube.cached_info[internal_kw] = convert_to_quantity(timeseries_table[storage_kw][inds])

        try:
            for ind, extra in product(["FWHM", "CONTRAST", "BIS SPAN"], ["", "_ERR"]):
                new_cube.cached_info[f"{ind}{extra}"] = timeseries_table[f"{ind}{extra}"]

            for key in ["DRS-VERSION", "DATE_NIGHT", "PROG ID", "INS MODE", "INS NAME", "bare_filename"]:
                new_cube.cached_info[key] = timeseries_text[key]
            new_cube.generate_root_path(Path(timeseries_text["root_folder"]))
        except:
            logger.warning(
                "Missing CCF indicators from previous run. Probably due to loading cube from previous SBART version",
            )

        new_cube.TM_RVs = convert_to_quantity(timeseries_table["TM_raw"][inds])
        new_cube.TM_RVs_ERR = convert_to_quantity(timeseries_table["TM_raw_ERR"][inds])
        new_cube._loaded_inst_info = True

        if load_full_flag:
            logger.debug("Loading entire information of the Flags")

            new_cube._OrderStatus = OrderStatus.load_from_json(storage_path=detailed_flags_filename.as_posix())

        if load_work_pkgs:
            logger.debug("Loading work packages")

            with open(workpackages_filename) as file:
                work_packages = json.load(file)
            converted_work_packages = [Package.create_from_json(elem) for elem in work_packages["work_packages"]]
            new_cube.update_worker_information(converted_work_packages)

        for unit in available_data_units:
            try:
                loaded_units = unit.load_from_disk(subInst_path / "RVcube")
            except NoDataError:
                logger.debug("Failed to find data from {}", unit._name)
                continue

            new_cube.add_extra_storage_unit(loaded_units, generate_root=False)

        return new_cube
