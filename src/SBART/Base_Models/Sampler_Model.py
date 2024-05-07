import time
from typing import Iterable, NoReturn, Optional, Tuple, Union, List

import numpy as np
from astropy.units import Quantity
from loguru import logger

from SBART.utils.BASE import BASE
from SBART.ModelParameters import ModelComponent, RV_component, RV_Model
from SBART.utils.UserConfigs import DefaultValues, UserParam, ValueFromList
from SBART.utils.custom_exceptions import (
    DeadWorkerError,
    FrameError,
    InvalidConfiguration,
)
from SBART.utils.status_codes import INTERNAL_ERROR, Flag, SUCCESS
from SBART.utils.types import UI_DICT
from SBART.utils.units import meter_second
from SBART.utils.work_packages import Package, WorkerInput


class SamplerModel(BASE):
    """
    Base Class for all SBART samplers.

    The samplers allow for the instantiation of a RV model, using the :py:mod:`~SBART.ModelParameters.RV_Model` which contain
    multiple  :py:mod:`~SBART.ModelParameters.Parameter` representing the different free-parameters of the full RV model. By default,
    all samplers **must** have a :py:class:`~SBART.ModelParameters.Parameter.RV_component`. Internally, SBART works in meters per
    second and all RV-related measurements are converted to it after entering the pipeline.

    The samplers implement a simple loop:

    * Draw a new, tentative, set of parameters of our RV model
    * Propose the new parameters to the model and evaluate its response
    * Evaluate any convergence metric that might exist.
    * Either restart the loop of exist (depending on convergence)


    The Samplers are implemented in such a way that they can be applied at a order-by-order level or at a global, achromatic level.

    *Note*: The majority of the Samplers only implement one of the two options

    *Note:* The memory saving mode is controlled by the :py:mod:`~SBART.rv_calculation` routines
    """

    _object_type = "Sampler"
    _name = "Model"

    _default_params = BASE._default_params + DefaultValues(
        # How to cotrol the creation of the RV window in which we sample the target functions
        #     GLOBAL - common to all valid frames [min(CCF RV) - RV_window[0]; max(CCF RV) + RV_window[1]
        #     SUB-INSTRUMENT - equal to GLOBAL, but inside the sub-Instruments
        #     OBSERVATION - only inside a observation
        WINDOW_GENERATION_MODE=UserParam(
            default_value="OBSERVATION",
            constraint=ValueFromList(("GLOBAL", "SUB-INSTRUMENT", "OBSERVATION")),
        ),
        STARTING_RV_PIPELINE=UserParam("DRS", constraint=ValueFromList(("DRS", "SBART"))),
    )

    def __init__(
        self,
        mode: str,
        RV_step,
        RV_window,
        params_of_model: Optional[Iterable[ModelComponent]] = None,
        user_configs: Optional[UI_DICT] = None,
        needed_folders: Optional[Iterable[str]] = None,
    ):
        super().__init__(
            user_configs=user_configs, needed_folders=needed_folders, root_level_path=None
        )

        self.mode = mode
        self.RV_step = RV_step
        self.mem_save_enabled = False
        self.disk_save_enabled = False
        self.is_merged_subInst = False

        RV_param = RV_component(
            RVwindow=RV_window,
            RV_keyword=self.RV_keyword,
            user_configs={"GENERATION_MODE": self._internal_configs["WINDOW_GENERATION_MODE"]},
        )

        model_components = [RV_param]
        if params_of_model is not None:
            model_components.extend(params_of_model)
        self.model_params = RV_Model(params_of_model=model_components)

        # shared memory buffers that the worker / target will have access to
        self.shared_buffers = {}

    def store_shared_buffer(self, buffer: dict) -> None:
        for key in buffer:
            if key in self.shared_buffers:
                logger.critical("Two shared memory buffers have the same name!")
                raise InvalidConfiguration
        self.shared_buffers = {**self.shared_buffers, **buffer}

    def set_mode(self, mode: str) -> None:
        """
        Set the sampler to one of its two working modes:

        - order-wise
        - epoch-wise

        Parameters
        ----------
        mode: str
            The mode for the sampler

        Raises
        -------
        InvalidConfiguration
            If the mode is not one of the valid options

        """
        valid_modes = ["order-wise", "epoch-wise"]
        if mode not in valid_modes:
            logger.critical(
                "Input argument <mode = {}> is not valid for {}. Select one from {}",
                mode,
                self.name,
                valid_modes,
            )
            raise InvalidConfiguration
        logger.info("Configured {} to be in mode <{}>", self.name, mode)
        self.mode = mode

    #################################
    #    Parameters of the model    #
    #################################

    def generate_prior_of_model(self, dataClassProxy):
        self.model_params.generate_priors(dataClassProxy)

    def add_extra_param(self, parameter):
        self.model_params.add_extra_param(parameter)

    def enable_param(self, param_name: str) -> NoReturn:
        """
        External activation of the parameters of the model

        Parameters
        ----------
        param_name: str
            One of the available parameters
        bounds:
            Bounds for the parameter
        guess
            Initial guess for the parameter

        Returns
        -------

        """
        self.model_params.enable_param(param_name)

    def lock_model_param(self, param_name: str) -> NoReturn:
        self.model_params.lock_param(param_name)

    #################################
    #    RV computation interface   #
    #################################

    def apply_orderwise(self, optimizer_estimate: Union[float, list], target, target_kwargs):
        """
        Minimize the target function for the data of a single order. As the models might have multiple
        free-parameters, we ensure that the target **always** receives a list of elements.

        Parameters
        ----------
        optimizer_estimate : Union[float, list]
            Value at which the target function will be evaluated.
        target : func
            Target function, which will be one the **worker** of the selected RV routine
        target_kwargs : [type]
            Input arguments of the target function. Must contain the following:
                - dataClassProxy,
                - frameID
                - order
        Returns
        -------
        func_output
            Evaluation of

        """

        params = None
        if isinstance(optimizer_estimate, Quantity):
            optimizer_estimate = optimizer_estimate.to(meter_second).value

        if isinstance(optimizer_estimate, (float, int)):
            params = [optimizer_estimate]
        else:
            params = optimizer_estimate

        return target(*params, **target_kwargs)

    def optimize_orderwise(self, target, target_kwargs: dict) -> Tuple[Package, Flag]:
        """
        Optimization over the functions that implements the orde-rwise application. This must be implemented
        by the children classes, as each model will use a different optimization strategy

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
        if self.mode == "epoch-wise":
            raise InvalidConfiguration
        raise NotImplementedError(f"{self.name} does not support orderwise application")

    def apply_epochwise(self, optimizer_estimate, config_dict):
        """
        Application of the model's parameters to all spectral orders at the same time. The children classes
        must implement this on their own, as the application stratagies will end up being different for each

        Parameters
        ----------
        config_dict:
            Dictionary that will be passed to the target function
        model_parameters
            List with the model parameters in the correct order

        Returns
        -------

        """
        if self.mode != "epoch-wise":
            raise InvalidConfiguration(f"Sampler is not in the epoch-wise mode")

        run_info = config_dict["run_information"]
        package_queue = config_dict["pkg_queue"]
        output_pool = config_dict["output_pool"]

        for order in run_info["valid_orders"]:
            worker_IN_pkg = self._generate_WorkerIn_Package(
                frameID=run_info["frameID"],
                order=order,
                run_info=run_info,
                subInst=run_info["subInst"],
                model_parameters=optimizer_estimate,
            )
            package_queue.put(worker_IN_pkg)

        outputs = self._receive_data_workers(len(run_info["valid_orders"]), output_pool, quiet=True)

        if run_info["target_specific_configs"]["compute_metrics"]:
            return self.process_epochwise_metrics(outputs)
        else:
            return self.compute_epochwise_combination(outputs)

    ######################################
    #  Comms interface with the workers  #
    ######################################

    def manage_RV_calculation(
        self,
        dataClass,
        subInst: str,
        valid_orders: tuple,
        target_specific_configs: dict,
        target_function,
        package_queue,
        output_pool,
    ) -> list:
        """ "
        This function has the goal of dispatching/managing the handling of the spectral data. Depending on the "mode", it will either
        launch the order- or epoch-wise managers. Those managers control the data that is currently in memory, the "work packages"
        that are available to the pool of workers and the communication between the main "process" and the workers.


        There is no "base" implementation of the epoch-wise manager, but there is base implementation of the order-wise manager:

        - Sequential iteration over all frames. For each frame:

            - Trigger the opening of the S2D arrays
            - Populate the package_queue with "work packages" for all valid orders
            - Wait for the responses
            - If the memory saving mode is enabled, close the S2D arrays

        Parameters
        ----------
        dataClass
        subInst
            subInst for which we want to compute RVs
        valid_orders:
            List of the valid orders for the RV calculation
        target_specific_configs
        target_function
        package_queue
            Communication queue between the main core (this one) and the workers
        output_pool
            Communication queue on which the workers place their outputs

        Returns
        -------
        worker_products
            List of products that was collected from the workers
            TODO: confirm/update this type hint

        Raises
        ----------
        InvalidConfiguration
            If there are no valid orders for the calculation of RVs
        """

        logger.debug(
            "{} managing the RV calculation of {} in <{}> mode",
            self.name,
            subInst,
            self.mode,
        )
        run_information = {
            "valid_orders": valid_orders,
            "target_function": target_function,
            "target_specific_configs": target_specific_configs,
        }

        if len(valid_orders) == 0:
            raise InvalidConfiguration(
                "{} has no valid order for which it can compute RVs".format(self.name)
            )

        if self.mode == "order-wise":
            return self._orderwise_manager(
                dataClass, subInst, run_information, package_queue, output_pool
            )
        elif self.mode == "epoch-wise":
            return self._epochwise_manager(
                dataClass, subInst, run_information, package_queue, output_pool
            )
        raise InvalidConfiguration("{} does not support mode <{}>".format(self.name, self.mode))

    def _orderwise_manager(
        self, dataClass, subInst: str, run_info: dict, package_queue, output_pool
    ) -> list:
        """
        Handle communication with the workers, when computing order-wise RVs.
        If the memory saving mode is enabled, the S2D arrays of the frames are closed afterwards
        """

        logger.debug("Starting orderWise manager")
        valid_IDS = dataClass.get_frameIDs_from_subInst(subInst)
        logger.debug("Running frameIDs : {}", valid_IDS)
        worker_prods = []
        if self.mem_save_enabled:
            logger.info("Memory saving mode is enabled. Using optimal RAM-saving strategy")
            for frameID in valid_IDS:
                # open before multiple cores attempt to open it!
                try:
                    _ = dataClass.load_frame_by_ID(frameID)
                except FrameError:
                    logger.warning("RunTimeRejection of frameID = {}", frameID)
                    continue
                logger.debug(
                    "Using RV window of: {}".format(self.model_params.get_RV_bounds(frameID))
                )
                N_packages = 0

                for order in run_info["valid_orders"]:
                    worker_IN_pkg = self._generate_WorkerIn_Package(
                        frameID, order, run_info, subInst
                    )

                    package_queue.put(worker_IN_pkg)
                    N_packages += 1

                worker_prods.append(
                    self._receive_data_workers(N_packages=N_packages, output_pool=output_pool)
                )
                if self.mem_save_enabled:
                    dataClass.close_frame_by_ID(frameID)
        else:
            logger.info("Memory saving mode is disabled. Using optimal sampling strategy")
            _ = dataClass.load_all_from_subInst(subInst)
            N_packages = 0
            for frameID in valid_IDS:
                for order in run_info["valid_orders"]:
                    worker_IN_pkg = self._generate_WorkerIn_Package(
                        frameID, order, run_info, subInst
                    )
                    package_queue.put(worker_IN_pkg)
                    N_packages += 1
            worker_prods.append(
                self._receive_data_workers(N_packages=N_packages, output_pool=output_pool)
            )
        return worker_prods

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

    def _receive_data_workers(self, N_packages: int, output_pool, quiet: bool = False) -> list:
        """
        Wait for the workers to populate the output_pool with the results.
        This will wait for exactly N_packages, without having any kind of timeout

        Parameters
        ----------
        N_packages : int
            Number of packages that we expect to receive

        Returns
        -------
        list
            List with the collected packages
        """

        received = 0
        outputs = []
        if not quiet:
            logger.debug("Waiting for {} data cubes from workers", N_packages)
        while True:
            data = output_pool.get()
            if data == INTERNAL_ERROR:
                logger.critical("One of the workers is dead. Shutting down all others")
                raise DeadWorkerError
            outputs.append(data)
            received += 1
            if received == N_packages:
                break
        if not quiet:
            logger.debug("Received all data.")
        return outputs

    @property
    def N_model_params(self) -> int:
        """

        Returns
        -------
        number_params:
            Number of free-parameters that are currently enabled in our model
        """
        return len(self.model_params.get_enabled_params())

    def is_sampler(self, sampler_type: str) -> bool:
        """
        Check if the sampler is of a given type

        Parameters
        ----------
        sampler_type

        Returns
        -------

        """
        return self.__class__._name == sampler_type

    def __repr__(self):
        return self.name

    def enable_memory_savings(self):
        logger.info("{} enabling memory saving mode", self.name)
        self.mem_save_enabled = True

    def disable_memory_savings(self):
        logger.info("{} disabling memory saving mode", self.name)
        self.mem_save_enabled = False

    def enable_disk_savings(self) -> NoReturn:
        """
        Save, as much as possible, disk space when saving the worker outputs. Each target function will
        decide on the details of such "savings"
        Returns
        -------

        """
        self.disk_save_enabled = True

    def disable_disk_savings(self) -> NoReturn:
        self.disk_save_enabled = False

    def _generate_WorkerIn_Package(
        self, frameID, order, run_info, subInst, **kwargs
    ) -> WorkerInput:
        worker_IN_pkg = WorkerInput()
        worker_IN_pkg["frameID"] = frameID
        worker_IN_pkg["order"] = order
        worker_IN_pkg["target_function"] = run_info["target_function"]
        worker_IN_pkg["subInst"] = subInst
        worker_IN_pkg["target_specific_configs"] = run_info["target_specific_configs"]
        worker_IN_pkg["force_RVbounds"] = False
        worker_IN_pkg["RVprior"] = self.model_params.get_RV_bounds(frameID)

        for key, value in kwargs.items():
            worker_IN_pkg[key] = value

        return worker_IN_pkg

    @property
    def RV_keyword(self) -> str:
        if self._internal_configs["STARTING_RV_PIPELINE"] == "SBART":
            RV_KW_start = "previous_SBART_RV"
        else:
            RV_KW_start = "DRS_RV"

        return RV_KW_start

    def process_epochwise_metrics(self, outputs):
        """
        Each children class must implement this, as it will be used to parse the outputs when
        the optimal RV is provided to the target!

        Parameters
        ----------
        outputs

        Returns
        -------

        """
        return [], []

    def compute_epochwise_combination(self, outputs):
        """
        Each children class must implement this to combine the order-wise metrics into a "global" value for the
        optimization process
        Parameters
        ----------
        outputs

        Returns
        -------

        """
        raise NotImplementedError("The children classes must override the epoch-wise combination")


if __name__ == "__main__":

    def target(val):
        return 3 * val**2 + 0 * val - 2

    sampler = SamplerModel()
    print(sampler.optimize(target))
