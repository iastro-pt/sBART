from loguru import logger
from typing import NoReturn, Dict, Any

from SBART.utils import custom_exceptions
from SBART.Base_Models.BASE import BASE
from SBART.ModelParameters import Model, ModelComponent
from SBART.utils.UserConfigs import (
    BooleanValue,
    DefaultValues,
    UserParam,
    Positive_Value_Constraint,
    IntegerValue
)


class NormalizationBase(BASE):
    _name = "SpecNormBase"
    _default_params = BASE._default_params + DefaultValues(
        NUMBER_WORKERS=UserParam(2, constraint=Positive_Value_Constraint + IntegerValue)
    )

    def __init__(self, obj_info: Dict[str, Any], user_configs, needed_folders=None):
        super().__init__(user_configs=user_configs,
                         needed_folders=needed_folders,
                         quiet_user_params=True
                         )
        print(obj_info)
        self._spec_info = obj_info
        # Avoid multiple loads of disk information
        self._loaded_disk_model: bool = False

        # Avoid multiple calls to disk loading if the file does not exist
        self._attempted_to_load_disk_model: bool = False

    def launch_normalization(self, wavelengths, flux, uncertainties):
        self._normalization_sanity_checks()


    def trigger_data_storage(self, *args, **kwargs) -> NoReturn:
        super().trigger_data_storage(args, kwargs)
        self._store_model_to_disk()

    def _normalization_sanity_checks(self):
        if self._spec_info["is_S1D"]:
            raise custom_exceptions.InvalidConfiguration("Can't normalize S1D spectra")
