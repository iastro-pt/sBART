from typing import List, Union, Iterable
from loguru import logger

from SBART.Base_Models.RV_routine import RV_routine
from SBART.data_objects.RV_cube import RV_cube
from SBART.utils import custom_exceptions, meter_second
from SBART.utils.RV_utilities.orderwiseRVcombination import orderwise_combination
from SBART.utils.UserConfigs import (
    DefaultValues,
    UserParam,
    ValueFromList,
    PathValue,
    BooleanValue,
)
from SBART.DataUnits import RV_Precision_Unit
from SBART.utils.types import UI_PATH
from .target_function import target


class RV_precision(RV_routine):
    """
    Compute the order-wise expected precision, based on the Bouchy et al 2001 paper

    Only allows to use the following samplers:

    - :py:class:`~SBART.Samplers.RVcontent`

    **User parameters:**

    ====================== ================ ================ ======================== ================
    Parameter name             Mandatory      Default Value    Valid Values                Comment
    ====================== ================ ================ ======================== ================
    CONTINUUM_FIT_TYPE      False           paper               paper / stretch         [1]
    ====================== ================ ================ ======================== ================

    - [1] How to model the continuum level:
        -   If "paper", then a polynomial is used to model the differences in the continuum.
        -   If "stretch", then we use the following continuum model: A*Template + B*wave + C

    *Note:* Also check the **User parameters** of the parent classes for further customization options of SBART
    """

    _name = "RV_precision"

    _default_params = RV_routine._default_params + DefaultValues(
        RV_variance_estimator=UserParam(
            "simple", constraint=ValueFromList(("simple", "with_correction"))
        ),
        RV_SOURCE=UserParam("DRS", constraint=ValueFromList(["DRS", "SBART"])),
        PREVIOUS_SBART_PATH=UserParam(default_value=None, constraint=PathValue),
        USE_MERGED_RVS=UserParam(False, constraint=BooleanValue),
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

        RV_configs_copy = RV_configs.copy()
        if RV_configs is not None:
            RV_configs_copy["order_removal_mode"] = "per_subInstrument"

        super().__init__(
            N_jobs=processes,
            RV_configs=RV_configs_copy,
            sampler=sampler,
            target=target,
            valid_samplers=["RVcontent"],
        )

    def run_routine(
        self,
        dataClass,
        storage_path: UI_PATH,
        orders_to_skip: Union[Iterable, str, dict] = (),
        store_data: bool = True,
        check_metadata: bool = False,
        store_cube_to_disk=True,
    ) -> None:
        if self._internal_configs["RV_SOURCE"] == "SBART":
            logger.info("Triggering load of the previous SBART results")
            if self._internal_configs["PREVIOUS_SBART_PATH"] is None:
                raise custom_exceptions.InvalidConfiguration(
                    "Can't use the SBART RV source without providing path"
                )
            dataClass.load_previous_SBART_results(
                self._internal_configs["PREVIOUS_SBART_PATH"],
                use_merged_cube=self._internal_configs["USE_MERGED_RVS"],
            )

        super().run_routine(
            dataClass, storage_path, orders_to_skip, store_data, check_metadata, store_cube_to_disk
        )

    def process_workers_output(self, empty_cube: RV_cube, worker_outputs: List[list]) -> RV_cube:
        data_unit = RV_Precision_Unit()
        for pkg in worker_outputs:
            for order_pkg in pkg:
                frameID = order_pkg["frameID"]
                order = order_pkg["order"]
                order_status = order_pkg["status"]
                RV = order_pkg["RV"].to(meter_second).value
                uncert = order_pkg["RV_uncertainty"].to(meter_second).value

                empty_cube.store_order_data(
                    frameID=frameID,
                    order=order,
                    RV=RV,
                    error=uncert,
                    status=order_status,
                )
                data_unit.store_RVcontent(
                    frameID=frameID,
                    order=order,
                    quality=order_pkg["quality"],
                    pix_sum_in_template=order_pkg["pix_sum_in_template"],
                )

        empty_cube.update_worker_information(worker_outputs)

        final_rv, final_error = orderwise_combination(
            empty_cube, self._internal_configs["RV_variance_estimator"]
        )

        final_rv = [i * meter_second for i in final_rv]
        final_error = [i * meter_second for i in final_error]

        empty_cube.update_computed_RVS(final_rv, final_error)
        empty_cube.add_extra_storage_unit(data_unit)
        return empty_cube
