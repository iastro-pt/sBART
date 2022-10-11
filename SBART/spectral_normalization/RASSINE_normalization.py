import matplotlib.pyplot as plt
import numpy as np

from SBART.utils import custom_exceptions
from loguru import logger

from SBART.spectral_normalization.normalization_base import NormalizationBase
from SBART.utils.UserConfigs import (
    DefaultValues,
    PathValue,
    UserParam,
    BooleanValue
)


class RASSINE_normalization(NormalizationBase):
    """

    **User parameters:**


    *Note:* Also check the **User parameters** of the parent classes for further customization options of SBART

    """

    _default_params = NormalizationBase._default_params + \
                      DefaultValues(S1D_folder=UserParam(mandatory=True, constraint=PathValue),
                                    )
    _name = "RASSINE"

    orderwise_application = False

    def __init__(self, obj_info, user_configs):
        super().__init__(obj_info=obj_info,
                         user_configs=user_configs,
                         )

    def run_RASSINE(self):
        # TODO: check the file format that we must build!
        # TODO: check the commands to launch RASSINE
        ...

    def fit_normalization(self, wavelengths, flux, uncertainties):
        # TODO: think about SNR problems that might arise within SBART if this goes through without adding an offset

        s1d_sol, s1d_sci, s1d_err = np.zeros(wavelengths.shape), np.zeros(wavelengths.shape), np.zeros(wavelengths.shape)

        ## Build txt file inputs

        self.run_RASSINE()

        if self._spec_info["is_S2D"]:
            # If the input frame is a S2D file, then we re-arrange the S1D file to fit the expected "shape"
            # of the S2D files!

            reconstructed_S2D = np.zeros(wavelengths.shape)
            reconstructed_wavelengths = np.zeros(wavelengths.shape)
            reconstructed_uncertainties = np.zeros(wavelengths.shape)

            order_number = 0
            order_size = wavelengths[0].size
            to_break = False

            while not to_break:
                start_order = order_size * order_number
                end_order = start_order + order_size
                if end_order >= s1d_sol.size:
                    to_break = True
                    end_order = s1d_sol.size

                slice_size = end_order - start_order
                reconstructed_wavelengths[order_number] = np.pad(s1d_sol[start_order:end_order], (0, order_size - slice_size))
                reconstructed_S2D[order_number] = np.pad(s1d_sci[start_order:end_order], (0, order_size - slice_size))
                reconstructed_uncertainties[order_number] = np.pad(s1d_err[start_order:end_order], (0, order_size - slice_size))
                order_number += 1
            # The "new" orders that don't have any information will have a flux of zero. Thus, they will be deemed to
            # be invalid during the mask creation process (that is re-launched after this routine is done)

        # TODO: missing the parameters that will be cached!
        return reconstructed_wavelengths, reconstructed_S2D, reconstructed_uncertainties, {}

    def apply_normalization(self, wavelengths, flux, uncertainties, **kwargs):
        super().apply_normalization(wavelengths, flux, uncertainties, **kwargs)
        poly = np.poly1d(kwargs["param_vector"])
        model = poly(wavelengths)
        return flux / model, uncertainties / model

    def _normalization_sanity_checks(self):
        # TODO: check this, maybe we will be limited to BLAZE-corrected spectra!
        logger.debug("{} does not apply any sanity check on the data!")
