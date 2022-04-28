"""
Chi-squared minimization, for a classical template matching approach.
"""

from typing import Tuple

import numpy as np
from loguru import logger
from scipy.optimize import minimize_scalar

from SBART.Base_Models.Sampler_Model import SamplerModel
from SBART.utils.status_codes import CONVERGENCE_FAIL, SUCCESS, Flag
from SBART.utils.units import meter_second
from SBART.utils.work_packages import Package


class chi_squared_sampler(SamplerModel):
    """
    The Chi-squared sampler implements a bounded minimization of a chi-squared curve.

    This metric is defined in the RV_step worker. After finding the optimal value, fit a parabola to estimate the
    true minimum value and the RV that would be associated with it. It also uses the curvature of the chi squared
    curve to estimate the RV uncertainty.

    """

    _name = "chi_squared"

    def __init__(self, rv_step, rv_prior):
        """

        Parameters
        ----------
        rv_step: RV_measurement
            Step to use when computing the numerical derivatives of the metric function (for the parabolic fit)
        rv_prior
            Specifies the "effective" RV window that the sampler can use. More details in :ref:`here <SamplerInit>`.

        """
        super().__init__(
            mode="order-wise",
            RV_step=rv_step,
            RV_window=rv_prior,
        )

    def _orderwise_manager(
        self, dataClass, subInst: str, run_info: dict, package_queue, output_pool
    ) -> list:
        if self.mem_save_enabled:
            return super()._orderwise_manager(
                dataClass, subInst, run_info, package_queue, output_pool
            )

        logger.info("Memory saving mode is disabled. Using optimal sampling strategy")

        _ = dataClass.load_all_from_subInst(subInst)
        valid_IDS = dataClass.get_frameIDs_from_subInst(subInst)
        worker_prods = []
        logger.debug("Running frameIDs : {}", valid_IDS)
        N_packages = 0
        for frameID in valid_IDS:
            # open before multiple cores attempt to open it!
            for order in run_info["valid_orders"]:
                worker_IN_pkg = self._generate_WorkerIn_Package(frameID, order, run_info, subInst)

                package_queue.put(worker_IN_pkg)
                N_packages += 1

        worker_prods.append(
            self._receive_data_workers(N_packages=N_packages, output_pool=output_pool)
        )
        return worker_prods

    def optimize_orderwise(self, target, target_kwargs: dict) -> Tuple[Package, Flag]:
        """
        Compute the RV for an entire order, followed by a parabolic fit to estimate
        uncertainty and better adjust chosen RV

        Parameters
        ----------
        target : [type]
            [description]
        target_kwargs : [type]
            Input arguments of the target function. Must contain the following:
                - dataClassProxy,
                - frameID
                - order

        Returns
        -------
        [type]
            [description]
        """
        out_pkg = Package(("RV", "RV_uncertainty"))
        init_guess, rv_bounds = self.model_params.generate_optimizer_inputs(
            frameID=target_kwargs["current_frameID"], rv_units=meter_second
        )
        optimization_output = minimize_scalar(
            self.apply_orderwise,
            bounds=rv_bounds[0],
            args=(target, target_kwargs),
            method="bounded",
        )

        bad_order = False
        msg = ""
        if optimization_output.success:
            minimum_value = optimization_output.x
            order_status = SUCCESS

            rv_step = self.RV_step.to(meter_second).value
            local_rvs = np.arange(
                minimum_value - 5 * rv_step, minimum_value + 5.1 * rv_step, rv_step
            )
            if local_rvs[0] < rv_bounds[0][0] or local_rvs[-1] > rv_bounds[0][1]:
                bad_order = True
                msg = "Optimal RV less than 5 RV_step away from the edges of the window"
                logger.warning(msg)

            if not bad_order:
                local_curve = [target(i, **target_kwargs) for i in local_rvs]
                try:
                    rv, rv_err, a, b = self._apply_parabolic_fit(local_rvs, local_curve, rv_step)
                except IndexError:
                    # If the minimum value is not in the middle of the inverval, an error will be raised
                    # This might occur due to:
                    #    1) We are using a step size too small for the instrument
                    #    2) Something weird going on with the data
                    bad_order = True
                except:
                    logger.opt(exception=True).critical(
                        "Parabolic fit has failed, rejecting spectral order {}",
                        target_kwargs["current_order"],
                    )
                    bad_order = True
                    msg = "Parabolic fit has failed"
        else:
            bad_order = True

        if bad_order:
            rv, rv_err = np.nan, np.nan
            local_rvs = []
            local_curve = []
            a, b = np.nan, np.nan
            order_status = CONVERGENCE_FAIL(msg)
        else:
            new_target_kwargs = {**target_kwargs, **{"get_minimum_information": True}}
            min_info = target(rv, **new_target_kwargs)
            for key, val in min_info.items():
                out_pkg[key] = val

        # TODO: add optimization status & message
        out_pkg["RV"] = rv
        out_pkg["RV_uncertainty"] = rv_err
        out_pkg["RV_array"] = local_rvs
        out_pkg["metric_evaluations"] = local_curve
        out_pkg["chi_squared_fit_params"] = [a, b]

        return out_pkg, order_status

    def _apply_parabolic_fit(self, rvs, chi_squared, rv_step):
        """
        Apply the parabolic fit to the chi-square curve to estimate RV and error
        """
        index = np.argmin(chi_squared)
        if len(chi_squared) == 3 and index != 1:
            raise Exception(
                f"Minimum value is not True. adjacent point is smaller: rvs : {rvs} - chi : {chi_squared}"
            )

        if index - 1 < 0:
            raise IndexError()

        # If we have an index error, the caller will handle it!
        rv_minimum = rvs[index]

        rv = rv_minimum - 0.5 * rv_step * (chi_squared[index + 1] - chi_squared[index - 1]) / (
            chi_squared[index - 1] - 2 * chi_squared[index] + chi_squared[index + 1]
        )

        rv_err = (
            2
            * (rv_step ** 2)
            / (chi_squared[index - 1] - 2 * chi_squared[index] + chi_squared[index + 1])
        )

        a = (chi_squared[index - 1] - 2 * chi_squared[index] + chi_squared[index + 1]) / (
            2 * rv_step ** 2
        )
        b = (chi_squared[index + 1] - chi_squared[index - 1]) / (2 * rv_step)
        return rv, np.sqrt(rv_err), a, b
