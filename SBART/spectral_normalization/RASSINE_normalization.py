import matplotlib.pyplot as plt
import numpy as np

from SBART.utils import custom_exceptions
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

    _default_params = NormalizationBase._default_params + DefaultValues()
    _name = "RASSINE"

    def __init__(self, obj_info, user_configs):
        super().__init__(obj_info=obj_info,
                         user_configs=user_configs,
                         )

        self.run_RASSINE()

    def run_RASSINE(self):
        ...
        # TODO: check the commands to launch RASSINE

    def fit_normalization(self, wavelengths, flux, uncertainties):
        # This will do nothing, as the RASSINE FIT was computed at init-time
        # TODO: check the info to return in here!
        ...

    def apply_normalization(self, wavelengths, flux, uncertainties, **kwargs):
        super().apply_normalization(wavelengths, flux, uncertainties, **kwargs)
        poly = np.poly1d(kwargs["param_vector"])
        model = poly(wavelengths)
        return flux / model, uncertainties / model

    def _normalization_sanity_checks(self):
        # TODO: check this, maybe we will be limited to BLAZE-corrected spectra!
        logger.debug("{} does not apply any sanity check on the data!")
        ...
