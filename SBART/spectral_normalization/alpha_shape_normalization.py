from loguru import logger

from SBART.spectral_normalization.normalization_base import NormalizationBase
from SBART.utils.UserConfigs import (
    DefaultValues
)
from SBART.utils import custom_exceptions

class AlphaShape_normalization(NormalizationBase):
    """

    **User parameters:**


    *Note:* Also check the **User parameters** of the parent classes for further customization options of SBART

    """
    _name = "AlphaShape"

    # TODO: confirm the kernels that we want to allow
    _default_params = NormalizationBase._default_params + DefaultValues()
    orderwise_application = True

    def __init__(self, obj_info, user_configs):
        super().__init__(obj_info=obj_info,
                         user_configs=user_configs,
                         )

    def _fit_orderwise_normalization(self, wavelengths, flux, uncertainties):
        super().launch_orderwise_normalization(wavelengths, flux, uncertainties)
        # TODO: implement the interface in here!
        ...

    def _apply_orderwise_normalization(self, wavelengths, flux, uncertainties, **kwargs):
        return flux/10, uncertainties

    def _normalization_sanity_checks(self):
        super()._normalization_sanity_checks()
        # TODO: see what kind of data we want to use!

        if not self._spec_info["blaze_corrected"]:
            raise custom_exceptions.InvalidConfiguration(f"{self.name} can't normalize spectra that was not BLAZE corrected")