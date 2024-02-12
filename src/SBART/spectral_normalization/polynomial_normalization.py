import matplotlib.pyplot as plt
import numpy as np

from SBART.utils import custom_exceptions
from loguru import logger

from SBART.spectral_normalization.normalization_base import NormalizationBase
from SBART.utils.UserConfigs import DefaultValues


class Polynomial_normalization(NormalizationBase):
    """
    Order by order fit (inverse variance weights) of the blaze-corrected S2D spectra using a first degree polynomial. This does
    not iterate multiple times to remove outliers, leading to a **very** sub-optimal continuum model!

    Name of the normalizer: Poly-Norm

    **Example:**

    .. code-block:: python

        f = ESPRESSO(main_path / file_start,
                     user_configs={"NORMALIZE_SPECTRA": True,
                                   "NORMALIZATION_MODE": "Poly-Norm"
                                   }
             )

        f.normalize_spectra()
        f.trigger_data_storage()

    """

    _default_params = NormalizationBase._default_params + DefaultValues()
    _name = "Poly"
    orderwise_application = True

    def __init__(self, obj_info, user_configs):
        super().__init__(
            obj_info=obj_info,
            user_configs=user_configs,
        )

    def _fit_orderwise_normalization(self, wavelengths, flux, uncertainties):
        out = np.polyfit(x=wavelengths, y=flux, deg=1, w=1 / uncertainties**2)

        optim_result = {"param_vector": out}
        return *self._apply_orderwise_normalization(
            wavelengths, flux, uncertainties, param_vector=out
        ), optim_result

    def _apply_orderwise_normalization(self, wavelengths, flux, uncertainties, **kwargs):
        super()._apply_orderwise_normalization(wavelengths, flux, uncertainties, **kwargs)
        poly = np.poly1d(kwargs["param_vector"])
        model = poly(wavelengths)
        return flux / model, uncertainties / model

    def _normalization_sanity_checks(self):
        super()._normalization_sanity_checks()
        # TODO: see what kind of data we want to use!

        keys = {
            "flux_dispersion_balance_corrected": "flux-balanced",
            "blaze_corrected": "BLAZE corrected",
        }
        for key, value in keys.items():
            if not self._spec_info[key]:
                raise custom_exceptions.InvalidConfiguration(
                    f"{self.name} can't normalize spectra that was not {value}"
                )
