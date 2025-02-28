from typing import NoReturn

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator, RBFInterpolator, interp1d

from SBART.spectral_modelling.modelling_base import ModellingBase
from SBART.utils import custom_exceptions
from SBART.utils.choices import INTERPOLATION_ERR_PROP, SPLINE_INTERPOL_MODE
from SBART.utils.math_tools.Cubic_spline import CustomCubicSpline
from SBART.utils.UserConfigs import (
    DefaultValues,
    IntegerValue,
    Positive_Value_Constraint,
    UserParam,
    ValueFromIterable,
)


class ScipyInterpolSpecModel(ModellingBase):
    """**User parameters:**

    ============================ ================ ================ ======================== ================
    Parameter name                 Mandatory      Default Value    Valid Values                 Comment
    ============================ ================ ================ ======================== ================
    SPLINE_TYPE                     False           cubic            cubic/quadratic/pchip       Which spline
    INTERPOLATION_ERR_PROP          False           interpolation     [1]                       [2]
    NUMBER_WORKERS                  False           1                   Interger >= 0           [3]
    ============================ ================ ================ ======================== ================

    - [1] : One of interpolation / none / propagation
    - [2] - How the uncertainties are propagated through the spline interpolation
    - [3] - Number of workers to launch (this will happen for each core if [1] is propagation)
    *Note:* Also check the **User parameters** of the parent classes for further customization options of SBART

    """

    # TODO: confirm the kernels that we want to allow
    _default_params = ModellingBase._default_params + DefaultValues(
        SPLINE_TYPE=UserParam(
            SPLINE_INTERPOL_MODE.CUBIC_SPLINE,
            constraint=ValueFromIterable(SPLINE_INTERPOL_MODE),
        ),
        INTERPOLATION_ERR_PROP=UserParam(
            INTERPOLATION_ERR_PROP.interpolation,
            constraint=ValueFromIterable(INTERPOLATION_ERR_PROP),
        ),
        NUMBER_WORKERS=UserParam(1, IntegerValue + Positive_Value_Constraint),
    )

    def __init__(self, obj_info, user_configs):
        super().__init__(
            obj_info=obj_info,
            user_configs=user_configs,
        )

    def generate_model_from_order(self, order: int) -> NoReturn:
        """Overrides the parent implementation to make sure that nothing is done (as there is no need to
        generate models)
        """
        return

    def _store_model_to_disk(self) -> NoReturn:
        """There is nothing to be stored. Overriding parent implementation to avoid issues"""
        return

    def interpolate_spectrum_to_wavelength(self, og_lambda, og_spectra, og_err, new_wavelengths, order):
        """Interpolate the order of this spectrum to a given wavelength, using a spline.

        Parameters
        ----------
        order
            Spectral order to interpolate
        new_wavelengths
            New wavelength solution, for which we want to interpolate the spectrum to

        Returns
        -------

        Raises
        ------
        NoConvergenceError
            If the fit for this order failed

        """
        propagate_interpol_errors = self._internal_configs["INTERPOLATION_ERR_PROP"]

        interpolator_map = {
            SPLINE_INTERPOL_MODE.CUBIC_SPLINE: CubicSpline,
            SPLINE_INTERPOL_MODE.PCHIP: PchipInterpolator,
            SPLINE_INTERPOL_MODE.QUADRATIC_SPLINE: lambda x, y: interp1d(x, y, kind="quadratic"),
            SPLINE_INTERPOL_MODE.NEAREST: lambda x, y: interp1d(x, y, kind="nearest"),
            SPLINE_INTERPOL_MODE.RBF: lambda x, y: RBFInterpolator(x, y, kernel="cubic"),
        }

        if propagate_interpol_errors == INTERPOLATION_ERR_PROP.propagation:
            # Custom Cubic spline routine!
            if self._internal_configs["SPLINE_TYPE"] != SPLINE_INTERPOL_MODE.CUBIC_SPLINE:
                raise custom_exceptions.InvalidConfiguration("Can't use non cubic-splines with propagation")
            CSplineInterpolator = CustomCubicSpline(
                og_lambda,
                og_spectra,
                og_err,
                n_threads=self._internal_configs["NUMBER_WORKERS"],
            )
            new_data, new_errors = CSplineInterpolator.interpolate(new_wavelengths)

        elif propagate_interpol_errors in [
            INTERPOLATION_ERR_PROP.interpolation,
            INTERPOLATION_ERR_PROP.none,
        ]:
            if self._internal_configs["SPLINE_TYPE"] == SPLINE_INTERPOL_MODE.CUBIC_SPLINE:
                extra = {"bc_type": "natural"}
            else:
                extra = {}

            if self._internal_configs["SPLINE_TYPE"] == SPLINE_INTERPOL_MODE.RBF:
                # RBF interpolation needs 2d arrays
                og_lambda = og_lambda[:, np.newaxis]
                og_spectra = og_spectra[:, np.newaxis]
                og_err = og_err[:, np.newaxis]
                new_wavelengths = new_wavelengths[:, np.newaxis]

            CSplineInterpolator = interpolator_map[self._internal_configs["SPLINE_TYPE"]](
                og_lambda,
                og_spectra,
                **extra,
            )
            new_data = CSplineInterpolator(new_wavelengths)

            if propagate_interpol_errors == INTERPOLATION_ERR_PROP.none:
                new_errors = np.zeros(new_data.shape)
            else:
                CSplineInterpolator = interpolator_map[self._internal_configs["SPLINE_TYPE"]](
                    og_lambda,
                    og_err,
                    **extra,
                )
                new_errors = CSplineInterpolator(new_wavelengths)
        else:
            raise custom_exceptions.InvalidConfiguration(f"How did we get {propagate_interpol_errors=}?")
        if self._internal_configs["SPLINE_TYPE"] == SPLINE_INTERPOL_MODE.RBF:
            new_data = new_data[:, 0]
            new_errors = new_errors[:, 0]
        return new_data, new_errors
