from typing import NoReturn

from SBART.utils.UserConfigs import (
    DefaultValues,
    UserParam,
    ValueFromList,
)

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

    def interpolate_spectrum_to_wavelength(self, order, new_wavelengths):
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
        mu
            Mean prediction
        sigma
            Model uncertainty

        Raises
        --------
        NoConvergenceError
            If the fit for this order failed
        """

        ...
