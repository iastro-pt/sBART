from typing import NoReturn

import numpy as np

from SBART.utils.UserConfigs import (
    DefaultValues,
    UserParam,
    ValueFromList,
)
from scipy.interpolate import CubicSpline

from SBART.utils.math_tools.Cubic_spline import CustomCubicSpline
from SBART.utils import custom_exceptions
from modelling_base import ModellingBase


class ScipyInterpolSpecModel(ModellingBase):
    """

    **User parameters:**

    ============================ ================ ================ ======================== ================
    Parameter name                 Mandatory      Default Value    Valid Values                 Comment
    ============================ ================ ================ ======================== ================
    SPLINE_TYPE                     False           cubic               cubic/quadratic       Which spline
    ============================ ================ ================ ======================== ================

    *Note:* Also check the **User parameters** of the parent classes for further customization options of SBART

    """

    # TODO: confirm the kernels that we want to allow
    _default_params = ModellingBase._default_params + DefaultValues(
        SPLINE_TYPE=UserParam("cubic",
                              constraint=ValueFromList(["cubic", "quadratic"])
                              ),
        ERROR_PROP_MODE=UserParam("interpolation",
                                  constraint=ValueFromList(["none", "interpolation", "propagation"])
                                  )
    )

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

    def interpolate_spectrum_to_wavelength(self, og_lambda, og_spectra, og_err, new_wavelengths):
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

        propagate_interpol_errors = self._internal_configs["ERROR_PROP_MODE"]

        if propagate_interpol_errors == "propagation":
            # Custom Cubic spline routine!
            CSplineInterpolator = CustomCubicSpline(
                og_lambda,
                og_spectra,
                og_err,
                n_threads=self._internal_configs["NUMBER_WORKERS"],
            )
            new_data, new_errors = CSplineInterpolator.interpolate(new_wavelengths)

        elif propagate_interpol_errors in ["interpolation", "none"]:
            CSplineInterpolator = CubicSpline(og_lambda, og_spectra)
            new_data = CSplineInterpolator(new_wavelengths)

            if propagate_interpol_errors == "none":
                new_errors = np.zeros(new_data.shape)
            else:
                CSplineInterpolator = CubicSpline(og_lambda, og_err)
                new_errors = CSplineInterpolator(new_wavelengths)
        else:
            raise custom_exceptions.InvalidConfiguration(f"How did we get {propagate_interpol_errors=}?")

        return new_data, new_errors
