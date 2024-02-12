from loguru import logger
from typing import NoReturn, Dict, Any

from SBART.utils import custom_exceptions
from SBART.utils.BASE import BASE
from SBART.ModelParameters import Model, ModelComponent
from SBART.utils.UserConfigs import (
    BooleanValue,
    DefaultValues,
    UserParam,
    Positive_Value_Constraint,
    IntegerValue,
)


class ModellingBase(BASE):
    _name = "SpecModelBase"

    # The _default parameters that we define, must be added as configurations in the Modelling class
    # Otherwise, the user values will never reach here!
    _default_params = BASE._default_params + DefaultValues(
        FORCE_MODEL_GENERATION=UserParam(False, constraint=BooleanValue),
        NUMBER_WORKERS=UserParam(2, constraint=Positive_Value_Constraint + IntegerValue),
    )

    def __init__(self, obj_info: Dict[str, Any], user_configs, needed_folders=None):
        super().__init__(
            user_configs=user_configs, needed_folders=needed_folders, quiet_user_params=True
        )

        # Avoid multiple loads of disk information
        self._loaded_disk_model: bool = False

        # Avoid multiple calls to disk loading if the file does not exist
        self._attempted_to_load_disk_model: bool = False

        self._modelling_parameters = Model(params_of_model=[])
        self.object_info = obj_info
        self._init_model()

    def _init_model(self):
        for order in range(self.object_info["N_orders"]):
            self._modelling_parameters.generate_prior_from_frameID(order)

    def generate_model_from_order(self, order: int) -> NoReturn:
        if not self._internal_configs["FORCE_MODEL_GENERATION"]:
            try:
                if not self._attempted_to_load_disk_model:
                    self.load_previous_model_results_from_disk(
                        model_component_in_use=ModelComponent
                    )
            except custom_exceptions.NoDataError:
                logger.warning("No information found on disk from previous modelling.")
        else:
            logger.info("Forcing model generation. Skipping disk-searches of previous outputs")

        if self._modelling_parameters.has_valid_identifier_results(order):
            # logger.info(f"Parameters of order {order} already exist on memory. Not fitting a new model")
            raise custom_exceptions.AlreadyLoaded

    def interpolate_spectrum_to_wavelength(self, og_lambda, og_spectra, og_err, new_wavelengths):
        ...

    def set_interpolation_properties(self, new_properties) -> NoReturn:
        self._internal_configs.update_configs_with_values(new_properties)

    def load_previous_model_results_from_disk(self, model_component_in_use):
        if self._loaded_disk_model or self._attempted_to_load_disk_model:
            raise custom_exceptions.AlreadyLoaded

        self._attempted_to_load_disk_model = True

        logger.debug(
            "Searching for previous model on disk: {}".format(self._get_model_storage_filename())
        )

        try:
            storage_name = self._get_model_storage_filename()
        except custom_exceptions.MissingRootPath:
            logger.debug("Missing Root path information. Giving up on loading data")
            raise custom_exceptions.NoDataError

        try:
            loaded_model = Model.load_from_json(
                storage_name, component_to_use=model_component_in_use
            )
            self._loaded_disk_model = True
            self._modelling_parameters = loaded_model
        except FileNotFoundError:
            self._loaded_disk_model = False
            logger.debug("Failed to find disk model")
            raise custom_exceptions.NoDataError

    def _store_model_to_disk(self) -> NoReturn:
        """
        Store the fit parameters to disk, to avoid multiple computations in the future

        Returns
        -------

        """
        logger.info("Storing parameters of GP models to disk")
        if not self._modelling_parameters.has_results_stored:
            msg = "No results have been stored. Skipping data storage"
            logger.warning(msg)
            return

        full_fname = self._get_model_storage_filename()

        self._modelling_parameters.save_to_json_file(full_fname)

        logger.debug("Finished storage of spectral model")

    def trigger_data_storage(self, *args, **kwargs) -> NoReturn:
        super().trigger_data_storage(args, kwargs)
        self._store_model_to_disk()

    def _get_model_storage_filename(self) -> str:
        obj_type = self.object_info["object_type"]
        if obj_type == "Frame":
            filename_start = self.object_info["filename"]
        elif obj_type == "Template":
            filename_start = f"Template_{self.object_info['subInstrument']}"
        else:
            raise custom_exceptions.InvalidConfiguration(
                "Spectral modelling can't save results for {}", self._object_type
            )

        return filename_start
