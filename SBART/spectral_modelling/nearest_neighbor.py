from typing import NoReturn

import numpy as np

from SBART.spectral_modelling.modelling_base import ModellingBase


class NearestNeighbor(ModellingBase):
    """

    **User parameters:**

    *Note:* Also check the **User parameters** of the parent classes for further customization options of SBART

    """

    # TODO: confirm the kernels that we want to allow
    _default_params = ModellingBase._default_params

    def __init__(self, obj_info, user_configs):
        super().__init__(obj_info=obj_info,
                         user_configs=user_configs,
                         )

    def generate_model_from_order(self, order: int) -> NoReturn:
        """
        Overrides the parent implementation to make sure that nothing is done (as there is no need to
        generate models)
        """
        return

    def _store_model_to_disk(self) -> NoReturn:
        """
        There is nothing to be stored. Overriding parent implementation to avoid issues
        """
        return

    def interpolate_spectrum_to_wavelength(self, og_lambda, og_spectra, og_err, new_wavelengths, order):
        """
        Interpolate the order of this spectrum to a given wavelength, using a spline.
        Parameters
        ----------
        order
            Spectral order to interpolate
        new_wavelengths
            New wavelength solution, for which we want to interpolate the spectrum to

        Returns
        -------

        Raises
        --------
        NoConvergenceError
            If the fit for this order failed
        """

        new_data = np.zeros(new_wavelengths.shape)
        new_errors = np.zeros(new_wavelengths.shape)

        for pixel_ind, pixel in enumerate(new_wavelengths):
            location = np.abs(og_lambda - pixel).argmin()
            new_data[pixel_ind] = og_spectra[location]
            new_errors[pixel_ind] = og_err[location]

        return new_data, new_errors
