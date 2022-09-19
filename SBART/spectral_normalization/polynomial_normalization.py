from loguru import logger

from SBART.spectral_normalization.normalization_base import NormalizationBase
from SBART.utils.UserConfigs import (
    DefaultValues
)


class Polynomial_normalization(NormalizationBase):
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

    def fit_normalization(self, wavelengths, flux, uncertainties):
        return *self.apply_normalization(wavelengths, flux, uncertainties), {"ddd":21,
                                                                             "kasdkhjjkasdha":1
                                                                             }

    def apply_normalization(self, wavelengths, flux, uncertainties, **kwargs):
        super().apply_normalization(wavelengths, flux, uncertainties, **kwargs)
        return flux/10, uncertainties/10
    def _normalization_sanity_checks(self):
        super()._normalization_sanity_checks()
        # TODO: see what kind of data we want to use!


