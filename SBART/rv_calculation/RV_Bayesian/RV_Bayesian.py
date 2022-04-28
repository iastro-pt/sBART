from typing import List, NoReturn

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from SBART.Base_Models.RV_routine import RV_routine
from SBART.ModelParameters import ModelComponent
from SBART.data_objects.RV_cube import RV_cube
from SBART.utils import custom_exceptions
from SBART.utils.RV_utilities.orderwiseRVcombination import orderwise_combination
from SBART.utils.UserConfigs import (
    BooleanValue,
    DefaultValues,
    IntegerValue,
    UserParam,
    ValueFromList,
)
from SBART.utils.concurrent_tools.create_shared_arr import create_shared_array
from SBART.utils.custom_exceptions import InvalidConfiguration
from SBART.utils.status_codes import SUCCESS
from SBART.utils.units import meter_second
from .target_function import SBART_target

plt.rcParams.update({"font.size": 16})


class RV_Bayesian(RV_routine):
    """
    Class that implements the s-BART algorithm. The default user-parameters represent the algorithm that is described in the paper

    .. warning::
        The current implementation of this class makes use of the outputs from :py:class:`~SBART.rv_calculation.rv_stepping.RV_step`
        , as it uses exactly the same orders as the classical template matching

    **User parameters:**

    ========================== ================ ================ ============================== ==================================================================================
    Parameter name               Mandatory        Default Value            Valid Values             Comment
    ========================== ================ ================ ============================== ==================================================================================
    application                False              epoch-wise         order-wise/epoch-wise         How to compute the RVs
    RV_variance_estimator      False              simple            simple/with_correction         Similar to the same keyword from :py:class:`~SBART.rv_calculation.rv_stepping.RV_step.RV_step`
    CONTINUUM_FIT_POLY_DEGREE  False               1                   1                           Degree of polynomial that we assume the flux continuum to follow
    ========================== ================ ================ ============================== ==================================================================================

    .. note::
        Also check the **User parameters** of the parent classes for further customization options of SBART

    """

    _name = "S-BART"

    _default_params = RV_routine._default_params + DefaultValues(
        include_jitter=UserParam(False, constraint=BooleanValue),
        chromatic_trend=UserParam("none", ValueFromList(("none", "OrderWise"))),
        trend_degree=UserParam(2, constraint=IntegerValue),
        application=UserParam("epoch-wise", constraint=ValueFromList(("epoch-wise", "order-wise"))),
        # only used if we compute order-wise RVs
        RV_variance_estimator=UserParam(
            "simple", constraint=ValueFromList(("simple", "with_correction"))
        ),
        PLOT_MODEL_MISSPECIFICATION=UserParam(True, constraint=BooleanValue),
    )
    # Bayesian only accepts linear fit to continuum
    _default_params.update(
        "CONTINUUM_FIT_POLY_DEGREE",
        UserParam(1, constraint=ValueFromList([1])),
    )

    def __init__(self, processes: int, sub_processes: int, RV_configs: dict, sampler):
        """
        Parameters
        ----------------
        processes: int
            Total number of cores
        sub_processes: int
            Threads launched to calculate error propagation through cubic splines. They are only launched if error_prop_type == 'propagation'
        error_prop_type: str
            Method of error propagation. Can be either "propagation" for the analytical propagation, "interpolation" to interpolate input errors or "none" to
            avoid all error propagation (i.e. return zeros)
        compare_metadata: boolean
            If there is a previous result of ROAST compare to see if the input data and ROAST version is the same. If it is, then the computation is halted
        Notes
        ---------
        The configuration of processes/subprocesses is different from the one used to create the stellar template. This can allow for a
        greater control of the CPU burden
        """
        super().__init__(
            N_jobs=processes,
            workers_per_job=sub_processes,
            RV_configs=RV_configs,
            sampler=sampler,
            target=SBART_target,
            valid_samplers=["Laplace", "MCMC"],
        )

        if self._internal_configs["include_jitter"]:
            logger.info("{} including Flux jitter in its model", self.name)
            self.sampler.enable_param("jitter")

        if self._internal_configs["chromatic_trend"] != "none":
            logger.info(
                "{} including a RV polynomial trend of degree {} to be applied as {}",
                self.name,
                self._internal_configs["trend_degree"],
                self._internal_configs["chromatic_trend"],
            )

            if self._internal_configs["trend_degree"] == 1:
                self.sampler.enable_param("trend::slope")

            for k in range(2, self._internal_configs["trend_degree"]):
                # degree N polynomial -> N parameters since we don't want the constant term
                # Assume that it will be small (and ideally don't exist)
                param_name = f"trend::{k}"
                self.sampler.add_extra_param(
                    ModelComponent(param_name, initial_guess=0, bounds=(None, None))
                )
                self.sampler.enable_param(param_name)

        self.sampler.set_mode(self._internal_configs["application"])

    # TODO: check the arguments
    def run_routine(
        self,
        dataClass,
        storage_path,
        orders_to_skip=(),
        store_data=True,
        check_metadata=False,
        store_cube_to_disk=True,
    ) -> None:
        if not isinstance(orders_to_skip, str):
            logger.critical(
                "{} can only use the orders that were skipped by the \chi^2 methodology.",
                self.name,
            )
            raise InvalidConfiguration
        try:
            super().run_routine(
                dataClass=dataClass,
                storage_path=storage_path,
                orders_to_skip=orders_to_skip,
                store_data=store_data,
                check_metadata=check_metadata,
                store_cube_to_disk=store_cube_to_disk,
            )
        except custom_exceptions.NoDataError:
            logger.info("No data to process after checking metadata")
        except Exception as e:
            logger.opt(exception=True).critical("Found unknown error")

    def process_workers_output(self, empty_cube: RV_cube, worker_outputs: List[list]) -> RV_cube:
        """Load information from the worker outputs and store it in the desired format
        inside a RV cube

        Parameters
        ----------
        empty_cube : RV_cube
            RV cube in which the data will be stored
        worker_outputs : List[list]
            List of lists, where each entry is one output from the sampler. The entries are 'Packages'

        Returns
        -------
        RV_cube
            RV cube filled with all of the information
        """
        if self._internal_configs["application"] == "order-wise":
            cube = self._orderwise_processment(empty_cube, worker_outputs)
        elif self._internal_configs["application"] == "epoch-wise":
            cube = self._epochwise_processment(empty_cube, worker_outputs)

        return cube

    def create_extra_plots(self, cube) -> NoReturn:
        # TODO: not the best way of passing the "path" to the "metrics" folder, but....
        self.analyse_FluxModel_misspecification(
            cube.worker_outputs,
            storage_path=cube._internalPaths.get_path_to("plots", as_posix=False),
        )

    def _orderwise_processment(self, empty_cube: RV_cube, worker_outputs: List[list]) -> RV_cube:
        """Process the list of outputs from the worker and store the relevant information into the
        RV cubes. This function caters to data from the order-wise application of sBART

        Parameters
        ----------
        empty_cube : RV_cube
            [description]
        worker_outputs : List[list]
            [description]

        Returns
        -------
        RV_cube
            [description]
        """
        for pkg in worker_outputs:
            for order_pkg in pkg:
                frameID = order_pkg["frameID"]
                order = order_pkg["order"]
                order_status = order_pkg["status"]
                RV = order_pkg["RV"]
                uncert = order_pkg["RV_uncertainty"]

                empty_cube.store_order_data(
                    frameID=frameID,
                    order=order,
                    RV=RV.to(meter_second).value,
                    error=uncert.to(meter_second).value,
                    status=order_status,
                )

        empty_cube.update_worker_information(worker_outputs)
        final_rv, final_error = orderwise_combination(
            empty_cube, self._internal_configs["RV_variance_estimator"]
        )

        final_rv = [i * meter_second for i in final_rv]
        final_error = [i * meter_second for i in final_error]

        empty_cube.update_computed_RVS(final_rv, final_error)
        return empty_cube

    def _epochwise_processment(self, empty_cube: RV_cube, worker_outputs: List[list]) -> RV_cube:
        for pkg in worker_outputs:
            if len(pkg) > 1:
                logger.critical("How do we have more than one package?")

            for epoch_pkg in pkg:
                frameID = epoch_pkg["frameID"]
                order_status = epoch_pkg["status"]
                RV = epoch_pkg["RV"]
                uncert = epoch_pkg["RV_uncertainty"]

                empty_cube.store_final_RV(frameID, RV, uncert)
        empty_cube.update_worker_information(worker_outputs)
        return empty_cube

    def build_target_configs(self) -> dict:

        worker_conf = {
            "include_jitter": self._internal_configs["include_jitter"],
            "chromatic_trend": self._internal_configs["chromatic_trend"],
            "compute_metrics": False,
            "weighted": False,
        }
        return worker_conf

    def analyse_FluxModel_misspecification(self, sampler_outputs, storage_path):
        # TODO: store this to a file!

        logger.info("Analysing pixel Flux missspecification by the model")
        epoch_metric = {}
        total_outliers = 0
        total_valid = 0
        all_metric__values = []
        for epoch_outputs in sampler_outputs:
            for package in epoch_outputs:
                if package["frameID"] not in epoch_metric:
                    epoch_metric[package["frameID"]] = {}  # epoch : dict

                if package["status"] != SUCCESS:
                    logger.warning("FrameID {} did not converge.".format(package["frameID"]))
                    continue
                # it is a list of numpy arrays! in the epoch-wise mode
                if self._internal_configs["application"] == "order-wise":
                    full_model_misspec = package["FluxModel_misspecification"]
                else:
                    full_model_misspec = []
                    for entry in package["FluxModel_misspecification"]:
                        full_model_misspec.extend(entry)

                OrderOutliers = np.where(
                    np.abs(full_model_misspec) > 2
                )  # differences larger than 2 sigma
                all_metric__values.extend(full_model_misspec)

                epoch_metric[package["frameID"]]["number_outliers"] = len(OrderOutliers[0])
                epoch_metric[package["frameID"]]["valid_points"] = len(full_model_misspec)
                epoch_metric[package["frameID"]]["percentage_outliers"] = (
                    epoch_metric[package["frameID"]]["number_outliers"]
                    / epoch_metric[package["frameID"]]["valid_points"]
                )

                total_outliers += epoch_metric[package["frameID"]]["number_outliers"]
                total_valid += epoch_metric[package["frameID"]]["valid_points"]

        logger.info(
            "Outlier [> 2 sigma] percentage : {:.4f} [{}/{} pixels]".format(
                100 * total_outliers / total_valid, total_outliers, total_valid
            )
        )

        if self._internal_configs["PLOT_MODEL_MISSPECIFICATION"]:
            fig = plt.figure()
            bins = np.arange(-5, 5, 0.5)
            plt.hist(
                all_metric__values,
                bins=bins,
                histtype="step",
                edgecolor="black",
                density=True,
            )
            plt.xlabel("Sigma distance to model (template)")
            plt.ylabel("Density of pixels")
            [
                plt.axvline(i, color="red", linestyle="--", label=r"2$\sigma$" if i == -2 else None)
                for i in [-2, 2]
            ]
            [
                plt.axvline(
                    i,
                    color="blue",
                    linestyle="dotted",
                    label=r"1$\sigma$" if i == -1 else None,
                )
                for i in [-1, 1]
            ]

            plt.xlim([-4, 4])
            plt.legend(loc=4, bbox_to_anchor=(0.98, 1), ncol=2)
            plt.tight_layout()

            if self._internal_configs["application"] == "order-wise":
                # TODO: ensure that this KW actually exists
                fname = "model_Flux_missspecification_order{}.png".format(package["order"])
            elif self._internal_configs["application"] == "epoch-wise":
                fname = "model_Flux_missspecification.png"

            plt.savefig(
                (storage_path / fname).as_posix(),
                dpi=300,
            )
            plt.close(fig)
        else:
            logger.warning("Skipping the plot of the model missspecification")

    def calculate_rvs(self, output):
        """
        Store the final RV as computed by the Bayesian method
        """
        # TODO: re-work this function!
        for epoch, data in enumerate(output):
            self.RV_cube.store_single_RV(epoch, data["RV"], data["RV_uncertainty"])

            # order-wise likelihood at the optimal RV
            self._execution_metrics[epoch]["likelihood"] = data["detailed_RV_likelihood"]

            for key in ["jitter", "jitter_uncertainty", "opt_status", "opt_message"]:
                self._execution_metrics[epoch][key] = data[key]

    def _open_shared_memory(self, inst_info: dict) -> None:
        """If we are in the <epoch-wise> mode, open a shared memory array to be used as a cache for the updated mask!"""
        if self._internal_configs["application"] == "epoch-wise":

            buffer_info, _ = create_shared_array(np.zeros(inst_info["array_size"], dtype=np.bool))
            self._shared_mem_buffers["mask_cache"] = buffer_info

            buffer_info, _ = create_shared_array(
                np.zeros(inst_info["array_size"][0], dtype=np.bool)
            )
            self._shared_mem_buffers["cached_orders"] = buffer_info

            self.sampler.store_shared_buffer(self._shared_mem_buffers)

        else:
            logger.debug(
                "{} does not need to place data in shared memory in the {} mode",
                self.name,
                self._internal_configs["application"],
            )
            return

    @property
    def storage_name(self):
        name = self.__class__._name
        name = name + "/" + self.sampler.storage_name
        if self._internal_configs["application"] == "order-wise":
            name = name + "_chromatic"

        return name
