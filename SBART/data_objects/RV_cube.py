import copy
import json
import os
import time
import warnings
from typing import List, NoReturn, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from loguru import logger
from tabletexifier import Table

from SBART import __version__
from SBART.Base_Models.BASE import BASE
from SBART.utils.custom_exceptions import InvalidConfiguration
from SBART.utils.math_tools.weighted_std import wstd
from SBART.utils.paths_tools import build_filename
from SBART.utils.status_codes import ORDER_SKIP, Flag, OrderStatus, Status
from SBART.utils.units import (
    centimeter_second,
    convert_data,
    kilometer_second,
    meter_second,
)
from SBART.utils.work_packages import Package


class RV_cube(BASE):
    """
    The RV cube stores the result of a SBART run of a given sub-Instrument. It is also responsible for:

    - Provide a way of accessing SBART and /or CCF results
    - Apply RV corrections (drift, SA)
    - Store outputs to disk
    - Create plots and (crude) statistical analysis of the RV timeseries.

    """
    def __init__(self, subInst: str, frameIDs: List[int], instrument_properties: dict):
        """
        It contains:
            - the SA correction value (and applies it)
            - THe (single) drift correction (And applies it)
            - The orderwise RVs, errors and Status
            - The final RV and errors (both raw and corrected)

        TODO:
            - [ ] add check to see if we actually have data in the order-wise arrays!
            - [ ] Implement eniric interface
        """
        self._associated_subInst = subInst

        super().__init__(
            user_configs={},
            needed_folders={"plots": "plots", "metrics": "metrics", "RVcube": "RVcube"},
        )
        self.frameIDs = frameIDs

        N_epochs = len(frameIDs)
        N_orders = instrument_properties["array_size"][0]

        self.worker_outputs = []

        # RVs and uncertainties -> unitless values in meter/second
        self._Rv_orderwise = np.zeros((N_epochs, N_orders)) + np.nan
        self._RvErrors_orderwise = np.zeros((N_epochs, N_orders)) + np.nan

        self._OrderStatus = OrderStatus(N_orders=N_orders, frameIDs=frameIDs)

        self._drift_corrected = instrument_properties[
            "is_drift_corrected"
        ]  # if True then a drift correction will be applied to the data

        self.TM_RVs = [np.nan * meter_second for _ in range(N_epochs)]
        self.TM_RVs_ERR = [np.nan * meter_second for _ in range(N_epochs)]

        self.sBART_version = __version__

        # NOTE: must also add any **new** key to the load_from_disk function!
        needed_keys = [
            "BJD",
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
            "BIS SPAN",
            "FWHM",
        ]
        self.cached_info = {key: [] for key in needed_keys}

        self._loaded_inst_info = False

        self._mode = "individual"

        # Still be to improved:
        self.eniric_RV_precision = []
        self.expected_RV_precision = []
        self.template_RV_precision = []

        self._saved_to_disk = False

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

        self._Rv_orderwise, self._RvErrors_orderwise, self._OrderStatus = other.data
        self._loaded_inst_info = True
        self._saved_to_disk = False

    def set_merged_mode(self, orders_to_skip: List[int]) -> None:
        self._mode = "merged_subInst"
        self._OrderStatus.add_flag_to_order(
            order=orders_to_skip,
            all_frames=True,
            order_flag=ORDER_SKIP("Skiped due to merged mode"),
        )

    def update_skip_reason(self, orders: Union[List[int], Set[int], int], skip_reason) -> None:
        if len(orders) == 0:
            return

        if isinstance(orders, set):
            orders = list(orders)

        self._OrderStatus.add_flag_to_order(order=orders, order_flag=skip_reason, all_frames=True)

    def load_data_from_DataClass(self, DataClassProxy) -> None:
        logger.debug("{} loading frame information from dataclass", self.name)
        for frameID in self.frameIDs:
            for key in self.cached_info:
                if key in ["date_folders", "bare_filename"]:
                    continue

                self.cached_info[key].append(DataClassProxy.get_KW_from_frameID(key, frameID))

            frame = DataClassProxy.get_frame_by_ID(frameID)
            self.cached_info["date_folders"].append(frame.file_path)
            self.cached_info["bare_filename"].append(frame.bare_fname)

        self.cached_info["target"] = DataClassProxy.get_Target()
        self._loaded_inst_info = True

        for frameID in self.frameIDs:
            status = DataClassProxy.get_frame_by_ID(frameID).OrderWiseStatus
            self._OrderStatus.mimic_status(frameID, status)

    def update_worker_information(self, worker_info: List):
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

    def store_order_data(self, frameID: int, order: int, RV, error, status: Flag) -> None:
        epoch = self.frameIDs.index(frameID)
        self._Rv_orderwise[epoch][order] = RV
        self._RvErrors_orderwise[epoch][order] = error
        self._OrderStatus.add_flag_to_order(order=order, frameID=frameID, order_flag=status)

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
    ) -> Tuple[list, list, list]:
        """Return the RV timeseries

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
            logger.critical("which = {} is not supported by get_RV_timeseries", which)
            raise InvalidConfiguration

        if apply_drift_corr is None:
            correct_drift = not self._drift_corrected
        else:
            logger.debug("Forcing the drift correction to be set to {}", apply_drift_corr)
            correct_drift = apply_drift_corr

        if correct_drift:

            logger.info("Cleaning RVs of {} from the drift", self._associated_subInst)
            corrected_rv = []
            corrected_err = []
            for i in range(len(final_RVs)):
                corrected_rv.append(final_RVs[i] - self.cached_info["drift"][i])
                corrected_err.append(
                    np.sqrt(final_RVs_ERR[i] ** 2 + self.cached_info["drift_ERR"][i] ** 2)
                )

            final_RVs = corrected_rv
            final_RVs_ERR = corrected_err

        if apply_SA_corr:
            SA_corr = self.compute_SA_correction()
            final_RVs = [final_RVs[i] - SA_corr[i] for i in range(len(final_RVs))]

        final_RVs = convert_data(final_RVs, units, as_value)
        final_RVs_ERR = convert_data(final_RVs_ERR, units, as_value)

        return self.obs_times, final_RVs, final_RVs_ERR

    def get_RV_from_ID(
        self,
        frameID: int,
        which: str,
        apply_SA_corr: bool,
        as_value: bool,
        units,
        apply_drift_corr=None,
    ):
        """Retrieve the BJD, RV and RV_ERR from a given frameID

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
        times, rvs, uncerts = self.get_RV_timeseries(
            which=which,
            apply_SA_corr=apply_SA_corr,
            as_value=as_value,
            units=units,
            apply_drift_corr=apply_drift_corr,
        )
        ID_index = self.frameIDs.index(frameID)

        return times[ID_index], rvs[ID_index], uncerts[ID_index]

    def compute_SA_correction(self):
        if "SA_correction" in self.cached_info:
            return self.cached_info["SA_correction"]

        logger.info("Starting SA correction")

        SA = self.cached_info["target"].secular_acceleration

        min_time = 55500  # always use the same reference frame
        logger.info("Setting SA reference frame to BJD = {}", min_time)

        secular_correction = [
            SA * (self.obs_times[i] - min_time) / 365.25 for i in range(len(self.obs_times))
        ]
        self.cached_info["SA_correction"] = secular_correction

        return secular_correction

    ##########################
    #
    # Access data
    #

    def get_raw_TM_RVs(self) -> Tuple[list, list]:
        return self.TM_RVs, self.TM_RVs_ERR

    def get_raw_DRS_RVs(self) -> Tuple[list, list]:
        return self.cached_info["DRS_RV"], self.cached_info["DRS_RV_ERR"]

    def get_frame_orderwise_status(self, frameID) -> List[Status]:
        return self._OrderStatus.get_status_from_order(frameID, all_orders=True)

    @property
    def subInst(self) -> str:
        return self._associated_subInst

    def has_data_from_subInst(self, subInst: str) -> bool:
        return self.subInst == subInst

    @property
    def obs_times(self) -> List[float]:
        return self.cached_info["BJD"]

    @property
    def N_orders(self) -> int:
        return self._Rv_orderwise.shape[1]

    @property
    def name(self) -> str:
        return "RV cube from {}".format(self._associated_subInst)

    @property
    def data(self):
        return (
            self._Rv_orderwise.copy(),
            self._RvErrors_orderwise.copy(),
            copy.deepcopy(self._OrderStatus),
        )

    @property
    def problematic_orders(self) -> set:
        """Get the orders that should be discarded when computing RVs
        Returns
        -------
        [type]
            [description]
        """
        return self._OrderStatus.common_bad_orders

    ##################################
    #
    # Export data!
    #

    def build_datablock(self) -> dict:
        _, corr_rv, corr_err = self.get_RV_timeseries(
            which="SBART", apply_SA_corr=True, as_value=True, units=kilometer_second
        )
        data_blocks = {
            "BJD": self.obs_times,
            "RVc": corr_rv,
            "RVc_ERR": corr_err,
            "OBJ": [self.cached_info["target"].true_name for _ in self.obs_times],
            "SA": convert_data(self.cached_info["SA_correction"], meter_second, True),
            "DRIFT": convert_data(self.cached_info["drift"], meter_second, True),
            "DRIFT_ERR": convert_data(self.cached_info["drift_ERR"], meter_second, True),
            "full_path": self.cached_info["date_folders"],
            "filename": [os.path.basename(i) for i in self.cached_info["date_folders"]],
            "frameIDs": self.frameIDs,
        }
        return data_blocks

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
            )
            row = [name + "_raw", np.std(rvs), wstd(rvs, uncerts), np.median(uncerts)]
            rv_table.add_row(row)

            _, rvs, uncerts = self.get_RV_timeseries(
                which=name,
                apply_SA_corr=True,
                as_value=True,
                units=meter_second,
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
            "DataRejectionSummary_{}_{}".format(self._associated_subInst, self._mode),
            "txt",
        )

        with open(final_path, mode="w") as file:
            order_skip_reasons = {"Warnings": {}, "Rejections": {}}
            file_skip_reasons = {"Warnings": {}, "Rejections": {}}

            file.write("Summary of data rejection:")

            file.write(
                "\n\tRejected {} out of {} available orders:".format(
                    len(self.problematic_orders), self._RvErrors_orderwise.shape[1]
                )
            )
            file.write("\n\tCommon orders removed:\n{}\n".format(self.problematic_orders))

            file.write("\nFrame-Wise analysis:")
            stellar_template = dataClassProxy.get_stellar_template(self._associated_subInst)
            for current_frameID in dataClassProxy.get_frameIDs_from_subInst(
                self._associated_subInst, include_invalid=True
            ):  # self.frameIDs:
                fpath = dataClassProxy.get_filename_from_frameID(current_frameID)
                file.write(
                    f"\n\tFrame {fpath} ({dataClassProxy.get_KW_from_frameID('ISO-DATE', current_frameID)}):\n"
                )
                if not stellar_template.was_loaded:
                    file.write(
                        f"\n\t\tIn Stellar Template: {stellar_template.check_if_used_frameID(current_frameID)}\n"
                    )

                current_Frame = dataClassProxy.get_frame_by_ID(current_frameID)
                lines, frame_skip = current_Frame.status.description(indent_level=2)
                file.write("".join(lines) + "\n")

                if current_frameID in self.frameIDs:
                    # Valid frame -> will include bad orders from the RV extraction
                    lines, frame_orderskip_reasons = self._OrderStatus.description(
                        indent_level=2,
                        frameID=current_frameID,
                        include_footer=False,
                        include_header=False,
                    )
                else:
                    # Completelly rejected file -> all info exists on the actual Frame Object
                    lines, frame_orderskip_reasons = current_Frame.OrderStatus.description(
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
                    file.write("\n\t\t{}:".format(master_key))
                    for flag, description in master_dict[master_key].items():
                        file.write(f"\n\t\t\t{flag}:{description}")

    def export_results(
        self,
        keys: List[str],
        header: List[str],
        dataClassProxy,
        text=True,
        rdb=True,
        append=False,
    ):
        if self._saved_to_disk:
            return

        storage_path = self._internalPaths.root_storage_path

        logger.debug("RV cube storing data under the main folder: {}", storage_path)

        if text:
            self.export_txt(header, append=append, keys=keys)
        if rdb:
            self.export_rdb(append)

        self.export_skip_reasons(dataClassProxy)
        self.compute_statistics()
        self.plot_RVs()
        self._saved_to_disk = True

    def plot_RVs(self) -> None:
        """Plot & store the RV timeseries

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
            bjd, rv, uncerts = self.get_RV_timeseries(
                which, apply_SA_corr=True, as_value=False, units=kilometer_second
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

        fig_full, ax_full = plt.subplots(2, 1, sharex=True)
        fig_part, ax_part = plt.subplots(2, 1, sharex=True)

        figure_list.extend([fig_full, fig_part])

        orders = np.asarray(range(self._Rv_orderwise.shape[1]))
        for epoch, data in enumerate(self._RvErrors_orderwise):
            full_rvs = self._Rv_orderwise[epoch]

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

                ax_full[1].plot(data[self.problematic_orders], marker="o", linestyle="", alpha=0.3)
                ax_full[1].plot(valid_orders, marker="x", linestyle="")

                centered_RVs = valid_RVs - np.nanmedian(valid_RVs)

                ax_part[0].errorbar(orders, centered_RVs, valid_orders, marker="o", linestyle="")

                ax_part[1].plot(valid_orders, marker="x", linestyle="")

        for ax in [ax_full, ax_part]:
            ax[0].set_ylabel("OrderWise RVs")
            ax[1].set_ylabel(r"$\sigma_{RV}$")
            ax[1].set_xlabel("Order")

        final_path = build_filename(diagnostics_path, "RV_raw_orderwise_errors", "png")
        fig_full.tight_layout()
        fig_full.savefig(final_path)

        final_path = build_filename(diagnostics_path, "RV_orderwise_errors", "png")
        fig_part.tight_layout()
        fig_part.savefig(final_path)

        logger.debug("Closing figures from {}", self.name)
        for figure in figure_list:
            plt.close(figure)

    def export_txt(self, header, keys, append=False):
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

        data_blocks = self.build_datablock()
        final_path = build_filename(
            self._internalPaths.root_storage_path,
            "RVs_{}_{}".format(self._associated_subInst, self._mode),
            "txt",
        )

        mode = "a" if append else "w"

        table = Table(header=header, table_style="NoLines")

        for epoch in np.argsort(data_blocks["BJD"]):
            line = []
            for key in keys:
                line.append(data_blocks[key][epoch])
            table.add_row(line)

        table.write_to_file(final_path, mode=mode, write_LaTeX=False)

    def export_rdb(self, append=False):
        star_name = self.cached_info["target"].original_name

        final_path = build_filename(
            self._internalPaths.root_storage_path,
            "{}_{}_{}".format(star_name, self._associated_subInst, self._mode),
            "rdb",
        )
        mode = "a" if append else "w"

        with open(final_path, mode=mode) as file:
            if not append:
                file.write("jdb\tvrad\tsvrad\n---\t----\t-----\n")

            obs, rvs, uncerts = self.get_RV_timeseries(
                which="SBART", apply_SA_corr=True, units=kilometer_second, as_value=True
            )

            for index in np.argsort(obs):
                file.write(f"{obs[index] - 24e5}\t{rvs[index]}\t{uncerts[index]}\n")

    ##
    # Under implementation
    ##

    def trigger_data_storage(self, *args, **kwargs) -> NoReturn:
        t0 = time.time()
        super().trigger_data_storage(*args, **kwargs)

        self._store_OrderWise_to_Fits()
        self._store_misc_info()
        self._store_work_packages()

        tf = time.time() - t0
        logger.info("Finished export of {} to disk. Took {:.2f} seconds".format(self.name, tf))

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

        for key in ["ISO-DATE", "date_folders"]:
            data_out["cached_info"][key] = self.cached_info[key]

        with open(storage_path, mode="w") as file:
            json.dump(data_out, file, indent=4)

        storage_path = build_filename(
            self._internalPaths.get_path_to("RVcube", as_posix=False),
            f"DetailedFlags_{self._associated_subInst}",
            fmt="json",
        )

        with open(storage_path, mode="w") as file:
            json.dump(self._OrderStatus.to_json(), file, indent=4)

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
        orderwiseRvs, orderwiseErrors, _ = self.data

        OBS_date, TM_RV, TM_ERR = self.get_RV_timeseries(
            "SBART", apply_SA_corr=False, apply_drift_corr=False, as_value=True, units=meter_second
        )
        OBS_date, prev_RV, prev_ERR = self.get_RV_timeseries(
            "DRS",
            apply_SA_corr=False,
            apply_drift_corr=False,
            as_value=True,
            units=meter_second,
        )

        OBS_date, prev_sbart_RV, prev_sbart_ERR = self.get_RV_timeseries(
            "previous_SBART",
            apply_SA_corr=False,
            apply_drift_corr=False,
            as_value=True,
            units=meter_second,
        )
        information = {
            "FrameID": self.frameIDs,
            "BJD": OBS_date,
            "DRS_RV": prev_RV,
            "DRS_RV_ERR": prev_ERR,
            "prevSBART_RV": prev_sbart_RV,
            "prevSBART_RV_ERR": prev_sbart_ERR,
            "TM_raw": TM_RV,
            "TM_raw_ERR": TM_ERR,
            "DRIFT": convert_data(self.cached_info["drift"], new_units=meter_second, as_value=True),
            "DRIFT_ERR": convert_data(
                self.cached_info["drift_ERR"], new_units=meter_second, as_value=True
            ),
            "SA": convert_data(
                self.cached_info["SA_correction"], new_units=meter_second, as_value=True
            ),
            "CONTRAST": self.cached_info["CONTRAST"],
            "FWHM": self.cached_info["FWHM"],
            "BIS SPAN": self.cached_info["BIS SPAN"],
        }
        coldefs = []
        for key, array in information.items():
            coldefs.append(fits.Column(name=key, format="D", array=array))
        hdu_timeseries = fits.BinTableHDU.from_columns(coldefs, name="TIMESERIES_DATA")

        header = fits.Header()
        header["HIERARCH drift_corr"] = self._drift_corrected
        header["VERSION"] = self.sBART_version
        header["mode"] = self._mode

        hdu = fits.PrimaryHDU(data=[], header=header)

        hdu_RVs = fits.ImageHDU(data=orderwiseRvs, header=header, name="ORDERWISE_RV")
        hdu_ERR = fits.ImageHDU(data=orderwiseErrors, header=header, name="ORDERWISE_ERR")
        hdu_mask = fits.ImageHDU(
            data=self._OrderStatus.as_boolean().astype(int), header=header, name="GOOD_ORDER_MASK"
        )

        hdul = fits.HDUList([hdu, hdu_timeseries, hdu_RVs, hdu_ERR, hdu_mask])

        storage_path = build_filename(
            self._internalPaths.get_path_to("RVcube", as_posix=False),
            f"OrderWiseInfo_{self._associated_subInst}",
            fmt="fits",
        )

        hdul.writeto(storage_path, overwrite=True)

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
        orderwise_filename = build_filename(
            subInst_path / "RVcube",
            filename=f"OrderWiseInfo_{subInst}",
            fmt="fits",
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

        with fits.open(orderwise_filename) as hdu:
            header_info = hdu[0].header
            timeseries_table = hdu["TIMESERIES_DATA"].data
            orderwise_RV = hdu["ORDERWISE_RV"].data
            orderwise_RV_ERR = hdu["ORDERWISE_ERR"].data
            good_order_mask = hdu["GOOD_ORDER_MASK"].data

        instrument_info = {
            "array_size": [orderwise_RV.shape[1], 0],
            "is_drift_corrected": header_info["HIERARCH drift_corr"],
        }
        frameIDs = timeseries_table["FrameID"].astype(int).tolist()

        new_cube = RV_cube(
            subInst=subInst, frameIDs=frameIDs, instrument_properties=instrument_info
        )

        logger.debug("Loading misc Info:")
        for key, values in miscInfo.items():
            new_cube.cached_info[key] = values

        logger.debug("Loading orderwise info")
        new_cube._Rv_orderwise = orderwise_RV
        new_cube._RvErrors_orderwise = orderwise_RV_ERR
        new_cube._mode = header_info["mode"]

        logger.debug("Generating the new order mask")

        for epoch_index, frameID in enumerate(frameIDs):
            for order, order_bool_status in enumerate(good_order_mask[epoch_index]):
                if order_bool_status != 1:
                    new_cube._OrderStatus.add_flag_to_order(
                        order=order, order_flag=ORDER_SKIP, frameID=frameID
                    )

        logger.debug("Loading timeseries data")

        convert_to_quantity = lambda data: [elem * meter_second for elem in data]

        new_cube.cached_info["BJD"] = timeseries_table["BJD"]

        entries = {
            "DRS_RV": "DRS_RV",
            "DRS_RV_ERR": "DRS_RV",
            "previous_SBART_RV": "prevSBART_RV",
            "previous_SBART_RV_ERR": "prevSBART_RV_ERR",
            "SA_correction": "SA",
            "drift": "DRIFT",
            "drift_ERR": "DRIFT_ERR",
        }

        for internal_kw, storage_kw in entries.items():
            new_cube.cached_info[internal_kw] = convert_to_quantity(timeseries_table[storage_kw])

        try:
            new_cube.cached_info["CONTRAST"] = timeseries_table["CONTRAST"]
            new_cube.cached_info["FWHM"] = timeseries_table["FWHM"]
            new_cube.cached_info["BIS SPAN"] = timeseries_table["BIS SPAN"]
        except:
            logger.warning(
                "Missing CCF indicators from previous run. Probably due to loading cube from previous SBART version"
            )

        new_cube.TM_RVs = convert_to_quantity(timeseries_table["TM_raw"])
        new_cube.TM_RVs_ERR = convert_to_quantity(timeseries_table["TM_raw_ERR"])
        new_cube._loaded_inst_info = True

        if load_full_flag:
            logger.debug("Loading entire information of the Flags")

            new_cube._OrderStatus = OrderStatus.load_from_json(
                storage_path=detailed_flags_filename.as_posix()
            )

        if load_work_pkgs:
            logger.debug("Loading work packages")

            with open(workpackages_filename) as file:
                work_packages = json.load(file)
            converted_work_packages = [
                Package.create_from_json(elem) for elem in work_packages["work_packages"]
            ]
            new_cube.update_worker_information(converted_work_packages)



        return new_cube
