from typing import List

import numpy as np
from loguru import logger

from SBART.Base_Models.RV_routine import RV_routine
from SBART.DataUnits import Classical_Unit
from SBART.data_objects.RV_cube import RV_cube
from SBART.utils import custom_exceptions, meter_second
from SBART.utils.RV_utilities.orderwiseRVcombination import orderwise_combination
from SBART.utils.UserConfigs import DefaultValues, UserParam, ValueFromList
from SBART.utils.custom_exceptions import BadTemplateError
from .target_function import target
from ...DataUnits.Act_Indicator_Unit import ActIndicators_Unit
from ...utils.math_tools.weighted_mean import weighted_mean


class RV_step(RV_routine):
    """
    Classical template matching approach.

    When computing RVs, it automatically uses both "modes" of the *order_removal_mode* user-parameter. This introduces no extra computational cost,
    as it is simply a change in which orders we use for the computation of the weighted mean.


    Only allows to use the following samplers:

    - :py:class:`~SBART.Samplers.chi_squared`
    - :py:class:`~SBART.Samplers.WindowSampler`

    **User parameters:**

    ====================== ================ ================ ======================== ================
    Parameter name             Mandatory      Default Value    Valid Values                Comment
    ====================== ================ ================ ======================== ================
    RV_variance_estimator       False           simple       simple/with_correction     What type of variance estimator we use when combining orderwise-RVs
    CONTINUUM_FIT_TYPE      False           paper               paper / stretch         [1]
    ====================== ================ ================ ======================== ================

    - [1] How to model the continuum level:
        -   If "paper", then a polynomial is used to model the differences in the continuum.
        -   If "stretch", then we use the following continuum model: A*Template + B*wave + C

    *Note:* Also check the **User parameters** of the parent classes for further customization options of SBART
    """

    _name = "RV_step"

    _default_params = RV_routine._default_params + DefaultValues(
        RV_variance_estimator=UserParam(
            "simple", constraint=ValueFromList(("simple", "with_correction"))
        )
    )
    _default_params.update(
        "CONTINUUM_FIT_TYPE",
        UserParam("paper", constraint=ValueFromList(("paper", "stretch"))),
    )

    def __init__(self, processes: int, RV_configs: dict, sampler):
        """
        Parameters
        ----------------
        processes: int
            Total number of cores
        sub_processes: int
            Threads launched to calculate error propagation through cubic splines. They are only launched if error_prop_type == 'propagation'
        RV_configs:
            Dictionary with the user-parameters of this class
        sampler:
            One of the :py:mod:`~SBART.Samplers` that is accepted by this routine.

        Notes
        ---------
        The configuration of processes/subprocesses is different from the one used to create the stellar template. This can allow for a
        greater control of the CPU burden
        """

        # TODO: careful, make this pretty!
        RV_configs_copy = RV_configs.copy()
        if RV_configs is not None:
            RV_configs_copy["order_removal_mode"] = "per_subInstrument"

        super().__init__(
            N_jobs=processes,
            RV_configs=RV_configs_copy,
            sampler=sampler,
            target=target,
            valid_samplers=["chi_squared", "Window"],
        )

    def run_routine(
        self,
        dataClass,
        storage_path,
        orders_to_skip=(),
        store_data=True,
        check_metadata=False,
        store_cube_to_disk=True,
    ) -> None:
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
            logger.opt(exception=True).info("No data to process after checking metadata")
            return

        except Exception as e:
            logger.opt(exception=True).critical("Found unknown error")
            return

        problem_orders = set()
        valid_subInst = self._subInsts_to_use

        if len(valid_subInst) == 0:
            logger.warning("No subInstruments to merge!")
            return

        if len(valid_subInst) == 1:
            logger.debug(
                "No real need to compute the merged set of valid orders when there is only 1 valid subINst"
            )

        stellar_model = dataClass.get_stellar_model()

        for subInst in valid_subInst:
            try:
                template_bad_orders = stellar_model.get_orders_to_skip(subInst=subInst)
            except BadTemplateError:
                return

            problematic_orders = set(self._output_RVcubes.get_orders_to_skip(subInst))
            problem_orders = problem_orders.union(problematic_orders)

        logger.info(
            "{} computed the super-set of orders to skip: {}",
            self.name,
            problem_orders,
        )
        logger.info(
            "Computing RVs with the 'merged' orders to skip for the following subInstruments: {}",
            valid_subInst,
        )
        for inst in valid_subInst:
            original_cube = self._output_RVcubes.get_RV_cube(inst, merged=False)

            cube = self._output_RVcubes.generate_new_cube(
                dataClass,
                inst,
                is_merged=True,  # frameIDs=original_cube.frameIDs
                has_orderwise_rvs=True,
            )

            cube.load_data_from(original_cube)
            cube.set_merged_mode(list(problem_orders))

            final_rv, final_error = orderwise_combination(
                cube, self._internal_configs["RV_variance_estimator"]
            )
            final_rv = [i * meter_second for i in final_rv]
            final_error = [i * meter_second for i in final_error]
            cube.update_computed_RVS(final_rv, final_error)

            data_unit_act = ActIndicators_Unit(
                available_inds=["DLW"],
                tot_number_orders=cube.N_orders,
                number_OBS=len(cube.frameIDs),
                list_of_fIDS=cube.frameIDs,
            )

            ind, errs = data_unit_act.get_all_orderwise_indicator("DLW")
            problematic_orders = cube.problematic_orders

            ind[:, problematic_orders] = np.nan
            errs[:, problematic_orders] = np.nan

            squared_errors = errs**2
            final_ind, ind_error = weighted_mean(ind, squared_errors, "simple")

            for index, frameID in enumerate(cube.frameIDs):
                data_unit_act.store_combined_indicators(
                    frameID, "DLW", ind_value=final_ind[index], ind_err=ind_error[index]
                )

            cube.add_extra_storage_unit(data_unit_act)
            self._output_RVcubes.add_RV_cube(inst, RV_cube=cube, is_merged=True)

            self._output_RVcubes.store_computed_RVs_to_disk(
                dataClassProxy=dataClass,
                which_subInst=inst,
            )

        self.trigger_data_storage(dataClass)

    def process_workers_output(self, empty_cube: RV_cube, worker_outputs: List[list]) -> RV_cube:
        data_unit = Classical_Unit()
        data_unit_act = ActIndicators_Unit(
            available_inds=["DLW"],
            tot_number_orders=empty_cube.N_orders,
            number_OBS=len(empty_cube.frameIDs),
            list_of_fIDS=empty_cube.frameIDs,
        )

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
                    RV=RV,
                    error=uncert,
                    status=order_status,
                )
                if order_status.is_good_flag:
                    data_unit.store_ChiSquared(
                        frameID=frameID,
                        order=order,
                        rvs=order_pkg["RV_array"],
                        chi_squared=order_pkg["metric_evaluations"],
                        fit_coeffs=order_pkg["chi_squared_fit_params"],
                    )
                    data_unit_act.store_orderwise_indicators(
                        frameID=frameID,
                        order=order,
                        ind_name="DLW",
                        ind_value=order_pkg["DLW"],
                        ind_err=order_pkg["DLW_ERR"],
                    )
        empty_cube.update_worker_information(worker_outputs)

        final_rv, final_error = orderwise_combination(
            empty_cube, self._internal_configs["RV_variance_estimator"]
        )

        final_rv = [i * meter_second for i in final_rv]
        final_error = [i * meter_second for i in final_error]
        empty_cube.update_computed_RVS(final_rv, final_error)

        ind, errs = data_unit_act.get_all_orderwise_indicator("DLW")
        problematic_orders = empty_cube.problematic_orders

        ind[:, problematic_orders] = np.nan
        errs[:, problematic_orders] = np.nan

        squared_errors = errs**2
        final_ind, ind_error = weighted_mean(ind, squared_errors, "simple")

        for index, frameID in enumerate(empty_cube.frameIDs):
            data_unit_act.store_combined_indicators(
                frameID, "DLW", ind_value=final_ind[index], ind_err=ind_error[index]
            )
        empty_cube.add_extra_storage_unit(data_unit_act)

        return empty_cube

    def build_chi2_fit(self, fit_metrics):
        start, end, step = fit_metrics["tentative_rvs"]
        rvs = np.arange(start, end, step)
        a, b = fit_metrics["chi_squared_fit"]
        minimum_index = np.argmin(fit_metrics["chi_squared"])
        return (
            fit_metrics["chi_squared"][minimum_index]
            + a * (rvs - rvs[minimum_index] + b / (2 * a)) ** 2
            - (b / (2 * a)) ** 2
        )
