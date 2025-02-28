"""This sampler is used mainly for debug purposes.
It will evaluate the "metric" of a given RV_routine in a pre-determined set of points
"""

from collections import defaultdict
from typing import Tuple

import numpy as np
from loguru import logger

from SBART.Base_Models.Sampler_Model import SamplerModel
from SBART.utils.choices import RV_EXTRACTION_MODE
from SBART.utils.custom_exceptions import FrameError
from SBART.utils.status_codes import SUCCESS, Flag
from SBART.utils.units import meter_second
from SBART.utils.work_packages import Package


class WindowSampler(SamplerModel):
    _name = "Window"

    def __init__(self, rv_step, rv_window, fixed_window: bool = True):
        super().__init__(
            mode=RV_EXTRACTION_MODE.ORDER_WISE,
            RV_step=rv_step,
            RV_window=rv_window,
        )

        self._fixed_window = fixed_window

    def generate_prior_of_model(self, dataClassProxy):
        super().generate_prior_of_model(dataClassProxy)

        if self._fixed_window:
            self.model_params.lock_param("RV")

    def _orderwise_manager(self, dataClass, subInst: str, run_info: dict, package_queue, output_pool) -> list:
        if self.mem_save_enabled:
            return super()._orderwise_manager(dataClass, subInst, run_info, package_queue, output_pool)

        logger.info("Memory saving mode is disabled. Using optimal sampling strategy")

        valid_IDS = dataClass.get_frameIDs_from_subInst(subInst)
        worker_prods = []
        _ = dataClass.load_all_from_subInst(subInst)
        logger.debug("Running frameIDs : {}", valid_IDS)
        N_packages = 0
        for frameID in valid_IDS:
            # open before multiple cores attempt to open it!
            for order in run_info["valid_orders"]:
                worker_IN_pkg = self._generate_WorkerIn_Package(frameID, order, run_info, subInst)
                package_queue.put(worker_IN_pkg)
                N_packages += 1

        worker_prods.append(self._receive_data_workers(N_packages=N_packages, output_pool=output_pool))
        return worker_prods

    def optimize_orderwise(self, target, target_kwargs: dict) -> Tuple[Package, Flag]:
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
        out_pkg = Package(("RV_array", "metric_evaluations"))

        metric_profile = []
        RV_array = []
        RV_window = self.model_params.get_RV_bounds(target_kwargs["current_frameID"])
        current_RV = RV_window[0].copy()
        while True:
            if current_RV >= RV_window[1]:
                break
            metric_profile.append(self.apply_orderwise(current_RV, target, target_kwargs))
            RV_array.append(current_RV)

            current_RV = current_RV + self.RV_step
        out_pkg["RV_array"] = RV_array
        out_pkg["metric_evaluations"] = metric_profile

        # Compatibility with RV_step
        out_pkg["RV"] = 0
        out_pkg["RV_uncertainty"] = 1
        out_pkg["chi_squared_fit_params"] = [0, 0]
        out_pkg["DLW"] = 0
        out_pkg["DLW_ERR"] = 0
        out_pkg["order"] = 0
        return out_pkg, SUCCESS

    def optimize_epochwise(self, target, target_kwargs: dict) -> Tuple[Package, Flag]:
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
        out_pkg = Package(("RV_array", "metric_evaluations"))

        metric_profile = []
        RV_array = []
        target_kwargs["current_frameID"] = target_kwargs["run_information"]["frameID"]
        RV_window = self.model_params.get_RV_bounds(target_kwargs["current_frameID"])
        current_RV = RV_window[0].copy()
        while True:
            if current_RV >= RV_window[1]:
                break
            metric_profile.append(
                self.apply_epochwise(
                    optimizer_estimate=[current_RV.to(meter_second).value],
                    config_dict=target_kwargs,
                )
            )
            RV_array.append(current_RV)

            current_RV = current_RV + self.RV_step
        out_pkg["RV_array"] = RV_array
        out_pkg["status"] = SUCCESS
        out_pkg["metric_evaluations"] = metric_profile
        out_pkg["frameID"] = target_kwargs["run_information"]["frameID"]
        # Compatibility with RV_step
        out_pkg["RV"] = 0 * meter_second
        out_pkg["RV_uncertainty"] = 1 * meter_second
        out_pkg["chi_squared_fit_params"] = [0, 0]
        out_pkg["DLW"] = [0]
        out_pkg["DLW_ERR"] = [0]
        out_pkg["order"] = [1]
        out_pkg["FluxModel_misspecification_from_order"] = [[100, 100]]
        return out_pkg, SUCCESS

    def compute_epochwise_combination(self, outputs):
        return np.sum([pkg["log_likelihood_from_order"] for pkg in outputs if pkg["status"] == SUCCESS])

    def process_epochwise_metrics(self, outputs):
        processed_package = defaultdict(list)
        for pkg in outputs:
            if pkg["status"] == SUCCESS:
                for key, item in pkg.items():
                    processed_package[key].append(item)
        if len(set(processed_package["frameID"])) != 1:
            raise FrameError(f"Mixing multiple frameIDs {set(processed_package['frameID'])}")
        processed_package["frameID"] = processed_package["frameID"][0]

        return processed_package
