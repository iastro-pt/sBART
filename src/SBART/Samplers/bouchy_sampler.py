"""Chi-squared minimization, for a classical template matching approach."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import minimize_scalar

from SBART import sbart_logger as logger
from SBART.Base_Models.Sampler_Model import SamplerModel
from SBART.utils.choices import RV_EXTRACTION_MODE
from SBART.utils.status_codes import CONVERGENCE_FAIL, SUCCESS, Flag
from SBART.utils.units import meter_second
from SBART.utils.UserConfigs import BooleanValue, DefaultValues, Positive_Value_Constraint, UserParam, ValueFromIterable
from SBART.utils.work_packages import Package


class bouchy_sampler(SamplerModel):
    """The Chi-squared sampler implements a bounded minimization of a chi-squared curve.

    This metric is defined in the RV_step worker. After finding the optimal value, fit a parabola to estimate the
    true minimum value and the RV that would be associated with it. It also uses the curvature of the chi squared
    curve to estimate the RV uncertainty.

    """

    _name = "Bouchy"
    _default_params = SamplerModel._default_params + DefaultValues(
        RV_ESTIMATION_MODE=UserParam(
            "NORMAL",
            constraint=ValueFromIterable(("NORMAL", "DRS-LIKE")),
            mandatory=False,
        ),
        ITER_NUMBER=UserParam(
            default_value=4,
            constraint=Positive_Value_Constraint,
            description="Number of iterations for the RV estimation",
        ),
        USE_TAYLOR_TERM_2=UserParam(
            default_value=False,
            constraint=BooleanValue,
            description="Use second term of the Taylor expansion",
        ),
    )

    def __init__(self, rv_step, rv_prior, user_configs: Optional[Dict[str, Any]] = None):
        super().__init__(
            mode=RV_EXTRACTION_MODE.ORDER_WISE,
            RV_step=rv_step,
            RV_window=rv_prior,
            user_configs=user_configs,
        )

    def optimize_orderwise(self, target, target_kwargs: dict) -> Tuple[Package, Flag]:
        out_pkg = Package(("RV", "RV_uncertainty"))
        init_guess, rv_bounds = self.model_params.generate_optimizer_inputs(
            frameID=target_kwargs["current_frameID"],
            rv_units=meter_second,
        )

        bad_order = False
        msg = ""
        target_kwargs["USE_TAYLOR_TERM_2"] = self._internal_configs["USE_TAYLOR_TERM_2"]
        rv, rv_err = target(self._internal_configs["ITER_NUMBER"], init_guess, **target_kwargs)
        order_status = SUCCESS
        if bad_order:
            rv, rv_err = np.nan, np.nan
            local_rvs = [0]
            local_curve = [0]
            a, b = np.nan, np.nan
            order_status = CONVERGENCE_FAIL(msg)
        # else:
        #     new_target_kwargs = {
        #         **target_kwargs,
        #         "get_minimum_information": True,
        #         "SAVE_DISK_SPACE": self.disk_save_enabled,
        #     }
        #     min_info = target(rv, init_guess, **new_target_kwargs)
        #     for key, val in min_info.items():
        #         out_pkg[key] = val

        # TODO: add optimization status & message
        out_pkg["RV"] = rv
        out_pkg["RV_uncertainty"] = rv_err
        # out_pkg["RV_array"] = local_rvs
        # out_pkg["metric_evaluations"] = local_curve
        # out_pkg["chi_squared_fit_params"] = [a, b]

        return out_pkg, order_status
