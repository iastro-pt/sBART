"""
Chi-squared minimization, for a classical template matching approach.
"""

from typing import Tuple, Dict, Any, Optional

import numpy as np
from loguru import logger
from scipy.optimize import minimize_scalar

from SBART.Base_Models.Sampler_Model import SamplerModel
from SBART.utils.UserConfigs import ValueFromList, UserParam, DefaultValues
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
    _default_params = SamplerModel._default_params + DefaultValues(
        RV_ESTIMATION_MODE=UserParam(
            "NORMAL", constraint=ValueFromList(("NORMAL", "DRS-LIKE")), mandatory=False
        )
    )

    def __init__(self, rv_step, rv_prior, user_configs: Optional[Dict[str, Any]] = None):
        """

        Parameters
        ----------
        rv_step: RV_measurement
            Step to use when computing the numerical derivatives of the metric function (for the parabolic fit)
        rv_prior
            Specifies the "effective" RV window that the sampler can use. More details in :ref:`here <SamplerInit>`.

        """
        super().__init__(
            mode="order-wise", RV_step=rv_step, RV_window=rv_prior, user_configs=user_configs
        )

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
        apply_parabolic_fit = False
        rv_step = self.RV_step.to(meter_second).value
        RV_estimation_mode = self._internal_configs["RV_ESTIMATION_MODE"]

        bad_order = False
        msg = ""

        if RV_estimation_mode == "NORMAL":
            optimization_output = minimize_scalar(
                self.apply_orderwise,
                bounds=rv_bounds[0],
                args=(target, target_kwargs),
                method="bounded",
            )

            if optimization_output.success:
                minimum_value = optimization_output.x
                order_status = SUCCESS

                local_rvs = np.arange(
                    minimum_value - 5 * rv_step, minimum_value + 5.1 * rv_step, rv_step
                )
                if local_rvs[0] < rv_bounds[0][0] or local_rvs[-1] > rv_bounds[0][1]:
                    bad_order = True
                    msg = "Optimal RV less than 5 RV_step away from the edges of the window"
                    logger.warning(msg)

                if not bad_order:
                    local_curve = [target(i, **target_kwargs) for i in local_rvs]
                    apply_parabolic_fit = True

        elif RV_estimation_mode == "DRS-LIKE":
            local_rvs = np.arange(rv_bounds[0][0], rv_bounds[0][1], rv_step)

            local_curve = list(
                map(lambda x: self.apply_orderwise(x, target, target_kwargs), local_rvs)
            )
            apply_parabolic_fit = True
            order_status = SUCCESS

        else:
            raise NotImplementedError(
                f"{self.name} does not implement a RV_ESTIMATION_MODE of {RV_estimation_mode}"
            )

        if apply_parabolic_fit:
            try:
                rv, rv_err, a, b = self._apply_parabolic_fit(local_rvs, local_curve, rv_step)
            except IndexError:
                # If the minimum value is not in the middle of the inverval, an error will be raised
                # This might occur due to:
                #    1) We are using a step size too small for the instrument
                #    2) Something weird going on with the data
                bad_order = True
                logger.critical("Problem with the minimum search")
            except:
                logger.opt(exception=True).critical(
                    "Parabolic fit has failed, rejecting spectral order {}",
                    target_kwargs["current_order"],
                )
                bad_order = True
                msg = "Parabolic fit has failed"
        else:
            logger.warning(f"Convergence failed due to {optimization_output.message}")
            bad_order = True

        if bad_order:
            rv, rv_err = np.nan, np.nan
            local_rvs = [0]
            local_curve = [0]
            a, b = np.nan, np.nan
            order_status = CONVERGENCE_FAIL(msg)
        else:
            new_target_kwargs = {
                **target_kwargs,
                **{"get_minimum_information": True, "SAVE_DISK_SPACE": self.disk_save_enabled},
            }
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
            * (rv_step**2)
            / (chi_squared[index - 1] - 2 * chi_squared[index] + chi_squared[index + 1])
        )

        a = (chi_squared[index - 1] - 2 * chi_squared[index] + chi_squared[index + 1]) / (
            2 * rv_step**2
        )
        b = (chi_squared[index + 1] - chi_squared[index - 1]) / (2 * rv_step)
        return rv, np.sqrt(rv_err), a, b
