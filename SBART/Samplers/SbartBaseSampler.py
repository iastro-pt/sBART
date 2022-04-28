"""
Common interface of the SBART samplers.

**Note:** Not supposed to be used by the user!

"""
import time
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from SBART.Base_Models.Sampler_Model import SamplerModel
from SBART.utils.types import RV_measurement
from SBART.utils.custom_exceptions import FrameError
from SBART.utils.status_codes import SUCCESS, Flag
from SBART.utils.work_packages import Package


class SbartBaseSampler(SamplerModel):
    """

    Base semi-Bayesian sampler, which implements the SBART model as described in the paper.

    The posterior characterization algorithms inherit from this:

    - Laplace's approximation: :py:class:`~SBART.Samplers.Laplace_approx`
    - MCMC: :py:class:`~SBART.Samplers.MCMC_sampler`

    """

    _name = SamplerModel._name + " SBART"

    def __init__(
        self,
        mode: str,
        RV_step: RV_measurement,
        RV_window: Tuple[RV_measurement, RV_measurement],
        user_configs,
        sampler_folders: Optional[Dict[str, str]] = None,
    ):
        """
        Approximate the posterior distribution with a LaPlace approximation;
        """
        extra_model_components = [
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

    def optimize(self, target, target_kwargs: dict) -> Tuple[Package, Flag]:
        """Compute the RV for an entire order, followed by a parabolic fit to estimate
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
        pass

    def _epochwise_manager(
        self, dataClass, subInst: str, run_info, package_queue, output_pool
    ) -> List[List[Package]]:

        valid_IDS = dataClass.get_frameIDs_from_subInst(subInst)

        worker_prods = []

        for frameID in valid_IDS:
            try:
                _ = dataClass.load_frame_by_ID(frameID)
            except FrameError:
                logger.warning("RunTimeRejection of frameID = {}", frameID)
                continue
            logger.info(
                "Starting RV extraction of {}",
                dataClass.get_filename_from_frameID(frameID),
            )
            starting_time = time.time()

            # Make sure that we have these two options disabled!
            run_info["target_specific_configs"]["compute_metrics"] = False
            run_info["target_specific_configs"]["weighted"] = False

            run_info["frameID"] = frameID
            run_info["subInst"] = subInst

            target_kwargs = {
                "run_information": run_info,
                "pkg_queue": package_queue,
                "output_pool": output_pool,
            }

            # for the epoch-wise application, the target is resolved inside the self.optimize function
            out_pkg, status = self.optimize_epochwise(target=None, target_kwargs=target_kwargs)

            if status != SUCCESS:
                logger.warning(
                    "Frame {} did not converge",
                    dataClass.get_filename_from_frameID(frameID),
                )
            # to mimic the outputs from the order-wise approach -> guarantee that the analysis of Flux misspec works
            worker_prods.append([out_pkg])

            logger.info("RV extraction took {} seconds", time.time() - starting_time)
            if self.mem_save_enabled:
                dataClass.close_frame_by_ID(frameID)

        return worker_prods

    def apply_epochwise(self, model_parameters, _, config_dict):
        """Computes the "global" logLikelihood

        Parameters
        ----------
        model_parameters : [type]
            [description]
        _ : [type]
            Empty argument to be consistent with the apply_orderwise arguments
        config_dict : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        run_info = config_dict["run_information"]
        package_queue = config_dict["pkg_queue"]
        output_pool = config_dict["output_pool"]

        for order in run_info["valid_orders"]:
            worker_IN_pkg = self._generate_WorkerIn_Package(
                frameID=run_info["frameID"],
                order=order,
                run_info=run_info,
                subInst=run_info["subInst"],
            )

            worker_IN_pkg["model_parameters"] = model_parameters

            package_queue.put(worker_IN_pkg)
        outputs = self._receive_data_workers(len(run_info["valid_orders"]), output_pool, quiet=True)

        if run_info["target_specific_configs"]["compute_metrics"]:
            flux_misspec, log_like, orders = [], [], []
            for pkg in outputs:
                if pkg["status"] == SUCCESS:
                    flux_misspec.append(pkg["FluxModel_misspec_from_order"].tolist())
                    log_like.append(pkg["log_likelihood_from_order"].tolist())
                    orders.append(pkg["order"])

            return flux_misspec, log_like, orders

        else:
            global_likeliood = np.sum(
                [pkg["log_likelihood_from_order"] for pkg in outputs if pkg["status"] == SUCCESS]
            )

        return global_likeliood

    def show_posterior(self, mean_value, variance, RVs):
        """
        Plot the approximated (Gaussian) posterior
        """
        std = np.sqrt(variance)
        gaussian = lambda x, mean, std: np.exp(-0.5 * ((x - mean) / std) ** 2) / (
            std * np.sqrt(2 * np.pi)
        )

        plt.scatter(RVs, gaussian(RVs, mean_value, std))

        plt.show()
