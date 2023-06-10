from typing import NoReturn

import numpy as np

from SBART.utils.UserConfigs import (
    DefaultValues,
    UserParam,
    ValueFromList,
    IntegerValue,
    Positive_Value_Constraint
)
from scipy.interpolate import CubicSpline, PchipInterpolator, interp1d, RBFInterpolator

from SBART.utils.math_tools.Cubic_spline import CustomCubicSpline
from SBART.utils import custom_exceptions
from SBART.spectral_modelling.modelling_base import ModellingBase


def sinc_interp(x, s, u):
    """
    Interpolates x, sampled at "s" instants
    Output y is sampled at "u" instants ("u" for "upsampled")

    from Matlab:
    http://phaseportrait.blogspot.com/2008/06/sinc-interpolation-in-matlab.html
    """

    if len(x) != len(s):
        raise Exception('x and s must be the same length')

    # Find the period
    T = s[1] - s[0]

    sincM = np.tile(u, (len(s), 1)) - np.tile(s[:, np.newaxis], (1, len(u)))
    y = np.dot(x, np.sinc(sincM / T))
    return y

class ScipyInterpolSpecModel(ModellingBase):
    """

    **User parameters:**

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
        SPLINE_TYPE=UserParam("cubic",
                              constraint=ValueFromList(["cubic",
                                                        "quadratic",
                                                        "pchip",
                                                        "nearest"
                                                        ])
                              ),
        INTERPOLATION_ERR_PROP=UserParam("interpolation",
                                         constraint=ValueFromList(["none", "interpolation", "propagation"])
                                         ),
        NUMBER_WORKERS=UserParam(1, IntegerValue + Positive_Value_Constraint),
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

        propagate_interpol_errors = self._internal_configs["INTERPOLATION_ERR_PROP"]

        interpolator_map = {"cubic": CubicSpline,
                            "pchip": PchipInterpolator,
                            "quadratic": lambda x, y: interp1d(x, y, kind="quadratic"),
                            "nearest": lambda x, y: interp1d(x, y, kind="nearest"),
                            "RBF": lambda x, y: RBFInterpolator(x, y, kind="nearest")
                            }

        if propagate_interpol_errors == "propagation":
            # Custom Cubic spline routine!
            if self._internal_configs["SPLINE_TYPE"] != "cubic":
                raise custom_exceptions.InvalidConfiguration("Can't use non cubic-splines with propagation")
            CSplineInterpolator = CustomCubicSpline(
                og_lambda,
                og_spectra,
                og_err,
                n_threads=self._internal_configs["NUMBER_WORKERS"],
            )
            new_data, new_errors = CSplineInterpolator.interpolate(new_wavelengths)

        elif propagate_interpol_errors in ["interpolation", "none"]:
            if self._internal_configs["SPLINE_TYPE"] == "cubic":
                extra = {"bc_type": "natural"}
            else:
                extra = {}
            CSplineInterpolator = interpolator_map[self._internal_configs["SPLINE_TYPE"]](og_lambda, og_spectra, **extra)
            new_data = CSplineInterpolator(new_wavelengths)

            if propagate_interpol_errors == "none":
                new_errors = np.zeros(new_data.shape)
            else:
                CSplineInterpolator = interpolator_map[self._internal_configs["SPLINE_TYPE"]](og_lambda, og_err, **extra)
                new_errors = CSplineInterpolator(new_wavelengths)
        else:
            raise custom_exceptions.InvalidConfiguration(f"How did we get {propagate_interpol_errors=}?")

        return new_data, new_errors
