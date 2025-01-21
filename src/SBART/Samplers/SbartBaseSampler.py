"""Common interface of the SBART samplers.

**Note:** Not supposed to be used by the user!

"""

from collections import defaultdict
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from SBART.Base_Models.Sampler_Model import SamplerModel
from SBART.ModelParameters import ModelComponent
from SBART.utils.custom_exceptions import FrameError
from SBART.utils.SBARTtypes import RV_measurement
from SBART.utils.status_codes import SUCCESS, Flag
from SBART.utils.work_packages import Package


class SbartBaseSampler(SamplerModel):
    """Base semi-Bayesian sampler, which implements the SBART model as described in the paper.

    The posterior characterization algorithms inherit from this:

    - Laplace's approximation: :py:class:`~SBART.Samplers.Laplace_approx`
    - MCMC: :py:class:`~SBART.Samplers.MCMC_sampler`

    """

    _name = SamplerModel._name + " SBART"

    def __init__(
        self,
        mode: str,
        RV_step: RV_measurement,
        RV_window: tuple[RV_measurement, RV_measurement],
        user_configs,
        sampler_folders: Optional[dict[str, str]] = None,
    ):
        """Approximate the posterior distribution with a LaPlace approximation;"""
        extra_model_components = [
            ModelComponent("jitter", initial_guess=10, bounds=[0, None]),
            ModelComponent("trend::slope", initial_guess=0, bounds=[None, None]),
        ]

        super().__init__(
            mode="order-wise",
            RV_step=RV_step,
            RV_window=RV_window,
            params_of_model=extra_model_components,
            user_configs=user_configs,
            needed_folders=sampler_folders,
        )

    def optimize_orderwise(self, target, target_kwargs):
        return self.optimize(target, target_kwargs)

    def optimize_epochwise(self, target, target_kwargs):
        return self.optimize(target, target_kwargs)

    def optimize(self, target, target_kwargs: dict) -> tuple[Package, Flag]:
        """Compute the RV for an entire order, followed by a parabolic fit to estimate
        uncertainty and better adjust chosen RV.


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

    def process_epochwise_metrics(self, outputs) -> dict[str, list]:
        processed_package = defaultdict(list)
        for pkg in outputs:
            if pkg["status"] == SUCCESS:
                for key, item in pkg.items():
                    processed_package[key].append(item)
        if len(set(processed_package["frameID"])) != 1:
            msg = f"Mixing multiple frameIDs {set(processed_package['frameID'])}"
            raise FrameError(msg)
        processed_package["frameID"] = processed_package["frameID"][0]

        return processed_package

    def compute_epochwise_combination(self, outputs):
        return np.sum([pkg["log_likelihood_from_order"] for pkg in outputs if pkg["status"] == SUCCESS])

    def show_posterior(self, mean_value, variance, RVs):
        """Plot the approximated (Gaussian) posterior"""
        std = np.sqrt(variance)
        gaussian = lambda x, mean, std: np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))

        plt.scatter(RVs, gaussian(RVs, mean_value, std))

        plt.show()
