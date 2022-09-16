from loguru import logger

from SBART.spectral_normalization.normalization_base import NormalizationBase
from SBART.utils.UserConfigs import (
    DefaultValues
)


class RASSINE_normalization(NormalizationBase):
    """

    **User parameters:**


    *Note:* Also check the **User parameters** of the parent classes for further customization options of SBART

    """

    # TODO: confirm the kernels that we want to allow
    _default_params = NormalizationBase._default_params + DefaultValues()

    def __init__(self, obj_info, user_configs):
        super().__init__(obj_info=obj_info,
                         user_configs=user_configs,
                         )

    def launch_normalization(self, wavelengths, flux, uncertainties):
        logger.info("here")
        super().launch_normalization(wavelengths, flux, uncertainties)
        logger.info("Launching Rassine normalization")
        # TODO: implement the interface in here!
        return flux/10, uncertainties

    def _normalization_sanity_checks(self):
        super()._normalization_sanity_checks()
        # TODO: see what kind of data we want to use!


