from typing import Any, Dict, Optional, Tuple

import numpy as np
from loguru import logger
from scipy.misc import derivative
from scipy.optimize import minimize, minimize_scalar

from SBART.utils.status_codes import CONVERGENCE_FAIL, SUCCESS, Flag
from SBART.utils.units import meter_second
from SBART.utils.work_packages import Package
from SBART.utils.types import RV_measurement

from .SbartBaseSampler import SbartBaseSampler


class Laplace_approx(SbartBaseSampler):
    """
    Laplace's approximation to the model's posterior distribution.
    Can be aplied either on epoch-wise or order-wise mode, with this configuration being made from the user parameters of the
    :py:mod:`~SBART.rv_calculation.RV_Bayesian` routine.
    """

    _name = "Laplace"

    def __init__(self, RV_step: RV_measurement, rv_prior: Tuple[RV_measurement, RV_measurement], user_configs: Optional[Dict[str, Any]] = None):
        """

        Parameters
        ----------
        RV_step
            Step that will be used to estimate the numerical derivative around the posterior's MAP
        rv_prior:
            Specifies the "effective" RV window that the sampler can use. More details in :ref:`here <SamplerInit>`.
        user_configs
        """
        super().__init__(
            mode="order-wise", RV_step=RV_step, RV_window=rv_prior, user_configs=user_configs
        )

        self._optimizers_map = {
            "scipy": minimize_scalar,
        }

    def optimize(self, target, target_kwargs: dict) -> Tuple[Package, Flag]:
        """
        Compute the RVs in the selected mode.

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

        params_to_use = self.model_params.get_enabled_params()

        initial_guesses, bounds = self.model_params.generate_optimizer_inputs(
            frameID=target_kwargs["run_information"]["frameID"], rv_units=meter_second
        )

        if self.mode == "order-wise":
            internal_func = self.apply_orderwise
        elif self.mode == "epoch-wise":
            logger.debug("Initial guesses: {}", initial_guesses)
            logger.debug("Param bounds: {}", bounds)
            internal_func = self.apply_epochwise
            out_pkg["frameID"] = target_kwargs["run_information"]["frameID"]

        if len(params_to_use) == 1:  # Brentt method for 1D optimization
            optimization_output = minimize_scalar(
                fun=internal_func,
                bounds=bounds[0],
                method="bounded",
                args=(target, target_kwargs),
            )
        else:
            optimization_output = minimize(
                fun=internal_func,
                x0=initial_guesses,  # uses the l-bfgs-b method for the minimization
                bounds=bounds,
                args=(target, target_kwargs),
            )

        out_pkg, order_status = self.process_posterior(
            optimization_output=optimization_output,
            target=target,
            target_kwargs=target_kwargs,
            output_pkg=out_pkg,
        )

        out_pkg["status"] = order_status
        return out_pkg, order_status

    def process_posterior(
        self, optimization_output, target, target_kwargs, output_pkg
    ) -> Tuple[Package, Flag]:
        """
        Process the results of the application of the Laplace Approximation

        Parameters
        ----------
        optimization_output : [type]
            outputs from the scipy's optimizer
        target : [type]
            target function from the RV routine
        target_kwargs : [type]
            input data for the target
        output_pkg : [type]
            pakacge in which the data will be stored

        Returns
        -------
        [type]
            [description]
        """
        status = SUCCESS
        if optimization_output.success:
            if self.N_model_params == 1:
                posterior_RV_mean_value = optimization_output.x
            else:
                posterior_RV_mean_value = optimization_output.x[0]
        else:
            posterior_RV_mean_value = np.nan
            status = CONVERGENCE_FAIL

        output_pkg["RV"] = posterior_RV_mean_value * meter_second
        output_pkg["opt_message"] = optimization_output["message"]

        if optimization_output.success:
            if self.model_params.check_if_enabled("jitter"):
                output_pkg["jitter"] = optimization_output.x[1]
                output_pkg["jitter_uncertainty"] = 0

            if self.model_params.check_if_enabled("chromatic_trend"):
                # The jitter will be the second entry of the optimization vector (see in self.params_of_model)
                # The params of the chromatic trend will start afterwards!
                trend_offset = 1 if self.model_params.check_if_enabled("jitter") else 0
                output_pkg["trend_params"] = optimization_output.x[1 + trend_offset :]
        else:
            output_pkg["trend_params"] = [0]

            for key in ["jitter", "jitter_uncertainty"]:
                output_pkg[key] = np.nan

            for key in ["RV", "RV_uncertainty"]:
                output_pkg[key] = np.nan * meter_second

        if optimization_output.success:
            target_interface = (
                self.apply_orderwise if self.mode == "order-wise" else self.apply_epochwise
            )

            if self.N_model_params == 1:
                # If we only use the "base" S-BART we can simply pass the base functions
                free_RV_target = lambda RV: target_interface(RV, target, target_kwargs)
            else:
                # Fix all parameters to MAP estimate and compute the 2nd derivative on RV
                free_RV_target = lambda RV: target_interface(
                    [RV, *optimization_output.x[1:]], target, target_kwargs
                )

            RV_variance = 1 / derivative(
                free_RV_target,
                posterior_RV_mean_value,
                dx=self.RV_step.to(meter_second).value,
                n=2,
                order=7,
            )

            output_pkg["RV_uncertainty"] = np.sqrt(RV_variance) * meter_second

            if self.mode == "epoch-wise":
                target_kwargs["run_information"]["target_specific_configs"][
                    "compute_metrics"
                ] = True
                target_kwargs["run_information"]["target_specific_configs"]["weighted"] = True
                model_misspec, log_likelihood, orders = target_interface(
                    optimization_output.x, target, target_kwargs
                )

            else:
                target_kwargs["compute_metrics"] = True
                target_kwargs["weighted"] = True
                _, model_misspec = target_interface(optimization_output.x, target, target_kwargs)

            output_pkg["FluxModel_misspecification"] = model_misspec

        return output_pkg, status
