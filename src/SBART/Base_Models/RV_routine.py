import time
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Any, Dict, Iterable, NoReturn, Optional, Union

import numpy as np
from loguru import logger

from SBART.utils.BASE import BASE
from SBART.data_objects import DataClass
from SBART.data_objects.MetaData import MetaData
from SBART.data_objects.RV_cube import RV_cube
from SBART.data_objects.RV_outputs import RV_holder
from SBART.rv_calculation.worker import worker
from SBART.utils import custom_exceptions
from SBART.utils.concurrent_tools.evaluate_worker_shutdown import evaluate_shutdown
from SBART.utils.custom_exceptions import (
    BadTemplateError,
    DeadWorkerError,
    InvalidConfiguration,
)
from SBART.utils.status_codes import BAD_TEMPLATE, ORDER_SKIP
from SBART.utils.types import UI_PATH
from SBART.utils.UserConfigs import (
    BooleanValue,
    DefaultValues,
    NumericValue,
    Positive_Value_Constraint,
    UserParam,
    ValueFromDtype,
    ValueFromList,
    IterableMustHave,
)
from SBART.utils.work_packages import ShutdownPackage


class RV_routine(BASE):
    """
    Base class for the all the RV extraction routines.


    **User parameters:**

    ========================== ================ ==================== ============================== ==================================================================================
    Parameter name                  Mandatory    Default Value            Valid Values                   Comment
    ========================== ================ ==================== ============================== ==================================================================================
    uncertainty_prop_type           False        interpolation        interpolation / propagation     How to propagate uncertainties when interpolation the stellar template
    order_removal_mode              False        per_subInstrument    per_subInstrument / global    How to combine the bad orders of the different sub-Instruments [1]
    sigma_outliers_tolerance        False            6                  Integer >= 0                 Tolerance to flag pixels as outliers (when compared with the template)
    min_block_size                  False           50                  Integer >= 0                If we have less than this number of consecutive valid pixels, reject that region
    output_fmt                      False           [2]                    [3]                      Control over the outputs that SBART will write to disk [4]
    MEMORY_SAVE_MODE                False           False                  boolean                  Save RAM at the expense of more disk operations
    CONTINUUM_FIT_POLY_DEGREE       False           1                  Integer >= 0                 Degree of the polynomial fit to the continuum.
    CONTINUUM_FIT_TYPE          False              "paper"              "paper"                     How to model the continuum
    ========================== ================ ==================== ============================== ==================================================================================

    - [1] The valid options represent:

        - per_subInstrument: each sub-Instrument is assumes to be independent, no ensurance that we are always using the same spectral orders
        - global: We compute a global set of bad orders which is applied for all sub-Instruments

    - [2] The default output format is: "BJD","RVc","RVc_ERR","SA","DRIFT","DRIFT_ERR","filename","frameIDs",

    - [3] The valid options are:
            - BJD :
            - MJD :
            - RVc : RV corrected from SA and drift
            - RVc_ERR : RV uncertainty
            - OBJ : Object name
            - SA : SA correction value
            - DRIFT : Drift value
            - DRIFT_ERR : Drift uncertainty
            - full_path : Full path to S2D file
            - filename : Only the filename
            - frameIDs : Internal ID of the observation

    - [4] This User parameter is a list where the entries can be options specified in [3]. The list **must** start with
        a "time-related" key (BJD/MJD), followed by RVc and RVc_ERR.

    *Note:* Also check the **User parameters** of the parent classes for further customization options of SBART

    """

    sampler_name = ""  # Laplace/MCMC/chi_squared
    _object_type = "RV method"
    _name = "BaseModel"

    _default_params = BASE._default_params + DefaultValues(
        uncertainty_prop_type=UserParam(
            "interpolation", constraint=ValueFromList(("interpolation", "propagation"))
        ),
        RV_extraction=UserParam("order-wise", constraint=ValueFromList(("order-wise",))),
        order_removal_mode=UserParam(
            "per_subInstrument", constraint=ValueFromList(("per_subInstrument", "global"))
        ),
        sigma_outliers_tolerance=UserParam(
            6, constraint=NumericValue
        ),  # tolerance for outliers in spectra - temp comp
        outlier_metric=UserParam("Paper", ValueFromList(("Paper", "MAD"))),
        remove_OBS_from_template=UserParam(
            False,
            BooleanValue,
        ),
        min_block_size=UserParam(
            50, constraint=Positive_Value_Constraint
        ),  # Min number of consecutive points to not reject a region
        output_fmt=UserParam(
            [
                "BJD",
                "RVc",
                "RVc_ERR",
                "SA",
                "DRIFT",
                "DRIFT_ERR",
                "DLW",
                "DLW_ERR",
                "filename",
                "frameIDs",
            ],
            constraint=ValueFromList(
                [
                    "BJD",
                    "MJD",
                    "RVc",
                    "RVc_ERR",
                    "OBJ",
                    "SA",
                    "DRIFT",
                    "DRIFT_ERR",
                    "full_path",
                    "filename",
                    "frameIDs",
                    "DLW",
                    "DLW_ERR",
                ]
            )
            + IterableMustHave(("RVc", "RVc_ERR"))
            + IterableMustHave(("MJD", "BJD"), mode="either"),
        ),  # RV_cube keys to store the outputs
        MEMORY_SAVE_MODE=UserParam(False, constraint=BooleanValue),
        SAVE_DISK_SPACE=UserParam(False, constraint=BooleanValue),
        CONTINUUM_FIT_TYPE=UserParam("paper", constraint=ValueFromList(("paper",))),
        CONTINUUM_FIT_POLY_DEGREE=UserParam(
            1, constraint=Positive_Value_Constraint + ValueFromDtype((int,))
        ),
        # How we select the wavelength regions to use. TODO: think about this one
        #     GLOBAL - common to all valid frames
        #     SUB-INSTRUMENT - equal to GLOBAL, but inside the sub-Instruments
        #     OBSERVATION - only inside a observation
        # Note: All that are not "OBSERVATION" need the spectral_analysis to run first
        COMMON_WAVELENGTHS_MODE=UserParam(
            default_value="OBSERVATION",
            constraint=ValueFromList(("GLOBAL", "SUB-INSTRUMENT", "OBSERVATION")),
        ),
    )

    def __init__(
        self,
        N_jobs: int,
        RV_configs: dict,
        sampler,
        target,
        valid_samplers: Iterable[str],
        extra_folders_needed: Optional[Dict[str, str]] = None,
    ):
        super().__init__(RV_configs, needed_folders=extra_folders_needed)
        self.package_pool = None
        self.output_pool = None

        self.N_jobs = N_jobs
        self._live_workers = 0

        self.to_skip = {}
        self._output_RVcubes = None
        self.sampler = sampler
        self._target_function = target

        self._validate_sampler(valid_samplers)

        self._data_in_shared_mem = False
        self._shared_mem_buffers = {}

        self._storage_path = None

        self.loaded_from_previous_run = False
        self._subInsts_to_use = []

        self.iteration_number: int = 0

    def _validate_sampler(self, valid_samplers: Iterable[str]) -> None:
        if not any(map(self.sampler.is_sampler, valid_samplers)):
            raise InvalidConfiguration(
                "{} does not accept the following Sampler : {}".format(
                    self.name,
                    self.sampler.name,
                )
            )

    def load_previous_RVoutputs(self):
        # TODO: understand what is going on!:
        # when comparing metadata this is called. Not sure if I want this or not ....
        logger.info("Loading previous RVoutputs from disk")
        try:
            self._output_RVcubes = RV_holder.load_from_disk(self._internalPaths.root_storage_path)
            self._output_RVcubes.update_output_keys(self._internal_configs["output_fmt"])
        except (custom_exceptions.NoDataError, custom_exceptions.InvalidConfiguration) as exc:
            logger.warning("Couldn't load previous RV outputs")
            raise custom_exceptions.StopComputationError from exc

    def find_subInstruments_to_use(self, dataClass, check_metadata: bool) -> None:
        """Check to see which subInstruments should be used!
        By default only compare the previous MetaData (if it exists) with the current one

        TODO: also check for the validity of stellar template in here!

        Parameters
        ----------
        dataClass : [type]
            [description]
        storage_path : str
            [description]
        check_metadata : bool
            [description]

        Raises
        -------
        NoDataError
            If all all sub-Instruments were rejected
        """
        self._subInsts_to_use = dataClass.get_subInstruments_with_valid_frames()

        if len(self._subInsts_to_use) == 0:
            logger.info("No new data available on disk. Stopping execution")
            raise custom_exceptions.NoDataError

        if check_metadata:
            logger.debug(
                f"Comparing metadata with the one stored in {dataClass.get_internalPaths().root_storage_path}"
            )

            try:
                previous_metadata = MetaData.load_from_json(
                    dataClass.get_internalPaths().root_storage_path
                )
            except custom_exceptions.NoDataError as exc:
                logger.warning("Failed to load Metadata. Skipping comparison")
                raise custom_exceptions.StopComputationError from exc

            try:
                self.load_previous_RVoutputs()
            except custom_exceptions.NoDataError as exc:
                logger.critical("Couldn't find previous sBART outputs in the provided path")
                raise custom_exceptions.StopComputationError from exc

            bad_subInst = []
            for subInst in self._subInsts_to_use:
                metaData = dataClass.get_metaData()
                equal_metadata = metaData.subInstrument_comparison(previous_metadata, subInst)

                if equal_metadata:
                    logger.info(
                        "Equal metadata found for subInstrument {}; Not computing RVs for it",
                        subInst,
                    )
                    bad_subInst.append(subInst)
                else:
                    logger.info(
                        "Different metadata found for subInstrument {}; Removing results from previous TM run",
                        subInst,
                    )
                    self._output_RVcubes.remove_subInstrument_data(subInst)

            is_merged = self._internal_configs["order_removal_mode"] == "global"

            if is_merged and len(bad_subInst) != len(self._subInsts_to_use):
                # If using merged, we must always recompute everything!
                return

            for badInst in bad_subInst:
                self._subInsts_to_use.remove(badInst)

            if len(self._subInsts_to_use) == 0:
                raise custom_exceptions.NoDataError("Metadata check removed all subInsts")

    def run_routine(
        self,
        dataClass,
        storage_path: UI_PATH,
        orders_to_skip: Union[Iterable, str, dict] = (),
        store_data: bool = True,
        check_metadata: bool = False,
        store_cube_to_disk=True,
    ) -> None:
        """
        Trigger the RV extraction for all sub-Instruments

        Parameters
        ----------
        check_metadata: bool
            If True, the TM check the Metadata if it already exists on disk (if it is the same: does nothing).
            By default False
        store_data: bool
            If True, saves the data to disk. By default True
        storage_path: Union[pathlib.Path, str]
            Path in which the outputs of the run will be stored
        dataClass : :class:`~SBART.data_objects.DataClass.DataClass`
            [description]
        orders_to_skip : Union[list,tuple,str,dict], optional
            Orders to skip for the RV calculation, in each subInstrument. If list/tuple remove for
            all subInstrument the same orders. If dict, the keys should be the subInstrument and the values a list to skip
            (if the key does not exist, assume that there are None to skip). If str, load a previous RV cube from disk and use the
            orders that the previous run used!. By default ()
        """

        if isinstance(storage_path, str):
            # Emsure pathlib path
            storage_path = Path(storage_path)
        storage_path = storage_path.absolute()

        self.iteration_number = dataClass.get_stellar_model().iteration_number

        # Note: self.storage_name from RV_Bayesian also includes the sampler name!
        self._internalPaths.add_root_path(
            storage_path / f"Iteration_{self.iteration_number}", self.storage_name
        )

        self.sampler.generate_root_path(self._internalPaths.root_storage_path)

        # The dataclass has nothing to store inside the Iteration folder!
        dataClass.generate_root_path(self._internalPaths.root_storage_path)
        self.sampler.generate_prior_of_model(dataClass)

        self.select_wavelength_regions(dataClass)

        self.find_subInstruments_to_use(
            dataClass=dataClass,
            check_metadata=check_metadata,
        )

        if self._internal_configs["SAVE_DISK_SPACE"]:
            logger.info(
                f"{self.name} will save disk space. Setting up the sampler to store less data"
            )
            self.sampler.enable_disk_savings()
        else:
            self.sampler.disable_disk_savings()

        if self._internal_configs["MEMORY_SAVE_MODE"]:
            self.sampler.enable_memory_savings()
        else:
            # Check here if the Stellar template will be interpolated with a GP:
            logger.info(f"{dataClass.get_stellar_model().get_interpol_modes()}")
            if "GP" in dataClass.get_stellar_model().get_interpol_modes():
                raise custom_exceptions.InternalError(
                    "Can't interpolate with GPs without having the memory saving mode enabled"
                )

            self.sampler.disable_memory_savings()

        if self._output_RVcubes is None:
            self._output_RVcubes = RV_holder(
                subInsts=self._subInsts_to_use,
                output_keys=self._internal_configs["output_fmt"],
                storage_path=self._internalPaths.root_storage_path,
            )

        # TO be over-written by the child classes
        logger.info("Computing RVs with {}", self.name)

        self.to_skip = self.process_orders_to_skip_from_user(orders_to_skip)
        self.complement_orders_to_skip(dataClass)

        original_to_skip = self.to_skip.copy()
        self.apply_orderskip_method()
        self.open_queues()

        # making sure that shared memory is always closed before exiting
        try:
            # Open the shared memory first, as we want to have the buffer info stored in the sampler!
            self._open_shared_memory(dataClass.get_instrument_information())
            self.launch_workers(dataClass)
            for subInst in self._subInsts_to_use:
                logger.info("Processing RVs from {}", subInst)
                output_cube = self.apply_routine_to_subInst(
                    dataClass=dataClass,
                    subInst=subInst,
                )

                if self.loaded_from_previous_run:
                    # if we are loading from the orders to skip from a previous run,
                    # the RV cubes of this iteration must follow the same "naming scheme"
                    is_merged = self._internal_configs["order_removal_mode"] == "global"
                else:
                    is_merged = False
                self._output_RVcubes.add_RV_cube(subInst, RV_cube=output_cube, is_merged=is_merged)

                if store_cube_to_disk:
                    self._output_RVcubes.store_computed_RVs_to_disk(
                        dataClassProxy=dataClass,
                        which_subInst=subInst,
                    )

        except DeadWorkerError:
            # we raised the exception after finding the first death
            # all others will still be preserved inside the output_pool
            self._live_workers -= 1
        except Exception:
            logger.opt(exception=True).critical("RV routine raised an Exception")

        self.close_multiprocessing()

        if len(self._subInsts_to_use) == 0:
            logger.warning("No data was processed. {} storing nothing to disk!", self.name)

        if store_data:
            self.trigger_data_storage(dataClass)

    def _validate_template_with_frame(self, stellar_template, first_frame) -> NoReturn:
        """
        Checks if the stellar template and the first frame share the same state of Flux Corrections
        """
        base_message = "Comparing spectra and template with different"

        comparison_map = (
            ("is_blaze_corrected", "BLAZE correction states"),
            (
                "flux_atmos_balance_corrected",
                "corrections of the flux balance due to the atmosphere",
            ),
            (
                "flux_dispersion_balance_corrected",
                "corrections of the flux dispersion with wavelength",
            ),
            ("was_telluric_corrected", "telluric correction states"),
        )
        messages_to_pass = []
        bad_comparison = False
        for kw_name, key_message in comparison_map:
            template_val = getattr(stellar_template, kw_name)
            frame_val = first_frame.check_if_data_correction_enabled(kw_name)
            if frame_val != template_val:
                messages_to_pass.append(
                    f"{base_message} {key_message} ({template_val} vs {frame_val})"
                )

                if kw_name != "was_telluric_corrected":
                    bad_comparison = True

        for message in messages_to_pass:
            logger.warning(message)

        if bad_comparison:
            raise custom_exceptions.InvalidConfiguration(
                "Failed comparison between template and spectra"
            )

    def apply_routine_to_subInst(self, dataClass: DataClass, subInst: str) -> RV_cube:
        # TO be over-written by the child classes
        valid_IDS = dataClass.get_frameIDs_from_subInst(subInst)
        N_epochs = len(valid_IDS)
        logger.info("Applying the RV routine to {} observations of {}", N_epochs, subInst)
        init_time = time.time()
        stellar_model = dataClass.get_stellar_model()

        stellar_template = stellar_model.request_data(subInstrument=subInst)
        first_frame = dataClass.get_frame_by_ID(valid_IDS[0])

        self._validate_template_with_frame(
            stellar_template=stellar_template, first_frame=first_frame
        )

        try:
            template_bad_orders = list(stellar_model.get_orders_to_skip(subInst=subInst))
        except BadTemplateError:
            logger.opt(exception=True).warning(
                "SubInst {} does not have a valid stellar template", subInst
            )
            return RV_cube(subInst, valid_IDS, dataClass.get_instrument_information())

        is_merged = self._internal_configs["order_removal_mode"] == "global"
        # UGLYYYY!
        self.sampler.is_merged_subInst = is_merged

        worker_outputs = self.sampler.manage_RV_calculation(
            dataClass,
            subInst,
            valid_orders=self.generate_valid_orders(subInst, dataClass),
            target_specific_configs=self.build_target_configs(),
            target_function=self._target_function,
            package_queue=self.package_pool,
            output_pool=self.output_pool,
        )

        cube = self._output_RVcubes.generate_new_cube(
            dataClass,
            subInst,
            is_merged=is_merged,
            has_orderwise_rvs=self._internal_configs["RV_extraction"] == "order-wise",
        )
        cube.update_skip_reason(self.to_skip[subInst], ORDER_SKIP)

        cube.update_skip_reason(template_bad_orders, BAD_TEMPLATE)
        cube.load_data_from_DataClass(dataClass)
        updated_cube = self.process_workers_output(cube, worker_outputs)
        logger.info(
            "Finished the computation of RVs from {}. Took: {} seconds",
            subInst,
            time.time() - init_time,
        )
        self.create_extra_plots(updated_cube)
        return updated_cube

    def create_extra_plots(self, cube) -> NoReturn:
        pass

    def process_workers_output(self, empty_cube: RV_cube, worker_outputs: list) -> RV_cube:
        logger.debug("{} processing outputs from the workers", self.name)
        raise NotImplementedError
        return empty_cube

    ##########################
    #   Comms with workers   #
    ##########################

    def build_target_configs(self) -> dict:
        """Create a dict with extra information to be passed inside the target functions, as a kwarg

        Returns
        -------
        dict
            [description]
        """
        return {}

    def generate_worker_configs(self, dataClassProxy) -> Dict[str, Any]:
        """
        Generate the dictionary that will be passed to the launching of the workers!

        Parameters
        ----------
        dataClassProxy

        Returns
        -------

        """
        worker_configs = {
            "OUTLIER_TOLERANCE": self._internal_configs["sigma_outliers_tolerance"],
            "MAX_ITERATIONS": 100,
            "METRIC_TO_USE": self._internal_configs["outlier_metric"],
            "remove_OBS_from_template": self._internal_configs["remove_OBS_from_template"],
            "min_block_size": self._internal_configs["min_block_size"],
            "min_pixel_in_order": dataClassProxy.min_pixel_in_order(),
            "uncertainty_prop_type": self._internal_configs["uncertainty_prop_type"],
            "CONTINUUM_FIT_TYPE": self._internal_configs["CONTINUUM_FIT_TYPE"],
            "CONTINUUM_FIT_POLY_DEGREE": self._internal_configs["CONTINUUM_FIT_POLY_DEGREE"],
            "RV_keyword": dataClassProxy.get_stellar_model().RV_keyword,
            "SAVE_DISK_SPACE": self._internal_configs["SAVE_DISK_SPACE"],
        }
        return worker_configs

    def launch_workers(self, dataClassProxy) -> None:
        logger.info("Lauching {} workers", self.N_jobs)

        worker_configs = self.generate_worker_configs(dataClassProxy)

        for _ in range(self.N_jobs):
            p = Process(
                target=worker,
                args=(
                    dataClassProxy,
                    self.package_pool,
                    self.output_pool,
                    worker_configs,
                    self.sampler,
                ),
            )
            p.start()
            self._live_workers += 1
        logger.info("Workers have been launched")

    ##########################
    #   Order selection          #
    ##########################
    def apply_orderskip_method(self) -> None:
        """
        Computing the orders that will be rejected for each subInstrument
        Returns
        -------

        """
        logger.debug("Applying order-skip mode")
        if self._internal_configs["order_removal_mode"] == "per_subInstrument":
            logger.debug("per_subInstrument mode selected. Doing nothing")
        elif self._internal_configs["order_removal_mode"] == "global":
            logger.debug(
                "Selecting common rejection among all subInstruments. Updating orders to skip"
            )
            bad_orders = set()
            for orders_to_skip in self.to_skip.values():
                bad_orders = bad_orders.union(orders_to_skip)

            self.to_skip = {key: bad_orders for key in self.to_skip}
        else:
            raise InvalidConfiguration()

    def complement_orders_to_skip(self, dataClass) -> None:
        """
        Search for bad orders in the stellar template of all subInstruments.

        Do not search the individual frames, as they might not be opened when we reach here

        Parameters
        ----------
        dataClass : [type]
            [description]
        """
        logger.debug("{} loading bad orders from the stellar templates", self.name)
        stellar_model = dataClass.get_stellar_model()
        # individual load in here. apply_orderskip_method will convert if needed
        for inst in self.to_skip:
            try:
                bad_orders = stellar_model.get_orders_to_skip(subInst=inst)
            except BadTemplateError:
                logger.opt(exception=True).warning(
                    "SubInst {} does not have a valid stellar template", inst
                )
                continue

            self.to_skip[inst] = bad_orders.union(self.to_skip[inst])
            logger.debug("Subinst {}, skip: {}", inst, self.to_skip[inst])

    def process_orders_to_skip_from_user(self, to_skip) -> dict:
        """
        Evaluate the input orders to skip and put them in the proper format

        Parameters
        ----------
        dataClass : [type]
            DataClass
        to_skip : [type]
            Orders to skip

        Returns
        -------
        dict
            Keys will be the subinstruments, values will be a set with the orders to skip
        Raises
        ------
        NotImplementedError
            [description]
        """
        if isinstance(to_skip, (list, tuple)):
            logger.info("Skipping the same orders across all subInstruments {}", to_skip)
            orders_to_skip = {key: set(to_skip) for key in self._subInsts_to_use}

        elif isinstance(to_skip, str):
            logger.info("Loading orders to skip from previous run of SBART: {}", to_skip)
            self.loaded_from_previous_run = True
            previous_RV_outputs = RV_holder.load_from_disk(to_skip)
            orders_to_skip = {}

            for key in self._subInsts_to_use:
                if self._internal_configs["order_removal_mode"] == "per_subInstrument":
                    orders_to_skip[key] = previous_RV_outputs.get_orders_to_skip(key)
                elif self._internal_configs["order_removal_mode"] == "global":
                    orders_to_skip[key] = previous_RV_outputs.get_orders_to_skip("merged")

        elif isinstance(to_skip, dict):
            logger.info("Skipping different orders for each subInstrument:")
            [logger.info("{} - {}", key, value) for key, value in to_skip.items()]
            orders_to_skip = {key: set(value) for key, value in to_skip.items()}

        return orders_to_skip

    def generate_valid_orders(self, subInst, dataClass) -> list:
        bad_order = self.to_skip[subInst]
        N_orders = dataClass.get_instrument_information()["array_size"][0]
        return [i for i in range(N_orders) if i not in bad_order]

    #########################
    #      Data storage     #
    #########################

    def trigger_data_storage(self, dataClassProxy, store_data: bool = True) -> NoReturn:
        if not store_data:
            logger.info("Storage of products from {} is temporarily disabled!", self.name)
            return
        t0 = time.time()
        BASE_path = self._internalPaths.root_storage_path

        logger.debug("{} storing data to {}", self.name, BASE_path)

        self._output_RVcubes.store_complete_timeseries()

        self._output_RVcubes.trigger_data_storage()
        dataClassProxy.trigger_data_storage()

        tf = time.time() - t0
        logger.debug("{} data storage took {} seconds", self.name, tf)

    #########################
    #        Misc           #
    #########################

    @property
    def subInstruments_to_use(self):
        return self._subInsts_to_use

    #########################
    # Shared memory handles #
    #########################

    def close_multiprocessing(self) -> None:
        logger.debug("Shutting down the multiprocessing interfaces")
        self.kill_workers()
        self.close_queues()
        self.close_shared_mem_arrays()

    def close_shared_mem_arrays(self) -> None:
        """Close any array that might exist in shared memory"""
        logger.debug("Closing shared memory arrays")
        if not self._data_in_shared_mem:
            logger.debug("{} has no shared memory arrays!", self.name)
        for mem_block in self._shared_mem_buffers.values():
            mem_block[0].close()
            mem_block[0].unlink()

        self._data_in_shared_mem = False
        self.shm = {}

    def kill_workers(self):
        logger.debug("Sending shutdown signal to workers")

        good, bad = evaluate_shutdown(self.output_pool)
        logger.debug("Good shutdowns: {}; Bad shutdowns: {}".format(good, bad))

        self._live_workers -= good + bad
        logger.debug(
            "There are {} live workers. Sending shutdown signal for all".format(self._live_workers)
        )
        for _ in range(self._live_workers):
            self.package_pool.put(ShutdownPackage())
        logger.debug("Waiting for worker response")

        no_shutdown_counter = 0
        while self._live_workers > 0:
            good, bad = evaluate_shutdown(self.output_pool)
            self._live_workers -= good + bad

            if good + bad != 0:
                logger.debug(
                    "Received {} shutdown signals. Still missing  {}",
                    good + bad,
                    self._live_workers,
                )
            else:
                if no_shutdown_counter == 400:
                    logger.warning("Workers are refusing to shutdown!")
                no_shutdown_counter += 1

        if self._live_workers < 0:
            logger.critical("Number of live workers is negative ...")
        if self._live_workers > 0:
            logger.critical("There are still workers to close")
        else:
            logger.info("All workers are closed")

    def _open_shared_memory(self, inst_info: dict) -> None:
        logger.debug("{} does not need to place data in shared memory", self.name)
        return

    def open_queues(self) -> None:
        logger.debug("{} opening multiprocessing interfaces", self.name)

        self.package_pool = Queue()
        self.output_pool = Queue()

    def close_queues(self) -> None:
        logger.debug("{} closing multiprocessing interfaces", self.name)
        if self.package_pool is None:
            logger.warning(
                "RV routine {} does not have multiprocessing queues open. Nothing to close!",
                self.name,
            )
            return

        self.package_pool.close()
        self.output_pool.close()

    def select_wavelength_regions(self, dataClass):
        logger.info("Selecting wavelength regions to use")
        if self._internal_configs["COMMON_WAVELENGTHS_MODE"] == "OBSERVATION":
            logger.info("Selected OBSERVATION mode, doing nothing")
            return

        spec_analysis_path = self._internalPaths.root_storage_path.parent / "spectral_analysis"
        for subInst in dataClass.get_subInstruments_with_valid_frames():
            path = spec_analysis_path / "wavelength_analysis"

            if self._internal_configs["COMMON_WAVELENGTHS_MODE"] == "GLOBAL":
                wave_analysis_path = path / "common_wavelengths_global.json"
            elif self._internal_configs["COMMON_WAVELENGTHS_MODE"] == "SUB-INSTRUMENT":
                wave_analysis_path = path / f"common_wavelengths_{subInst}.json"

            dataClass.select_common_wavelengths(wave_analysis_path, subInst)

    def launch_wavelength_selection(self, DataClassProxy: DataClass):
        """
        Currently not 100% implemented!

        Parameters
        ----------
        DataClassProxy

        Returns
        -------

        """
        outlier_search_configs = {
            "CONTINUUM_FIT_POLY_DEGREE": 0,
            "OUTLIER_TOLERANCE": 0,
            "METRIC_TO_USE": "MAD",
            "MAX_ITERATIONS": 100,
        }

        available_orders = DataClassProxy.get_instrument_information()["array_size"][0]

        for subInst in self._subInsts_to_use:
            frameIDs = DataClassProxy.get_frameIDs_from_subInst(subInst)
            orders = self.generate_valid_orders(subInst=subInst, dataClass=DataClassProxy)

            subInst_combined_counters = np.zeros(
                DataClassProxy.get_instrument_information()["array_size"]
            )
            # apply outlier search in here

            results = []

            for pkg in results:
                subInst_combined_counters[pkg["order"], pkg["masked_values"]] += 1

        # Re-generate the set of orders to skip after updating them
        self.apply_orderskip_method()
