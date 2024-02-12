from loguru import logger
from typing import NoReturn, Dict, Any

from SBART.utils import custom_exceptions
from SBART.utils.BASE import BASE
from SBART.ModelParameters import Model, ModelComponent
from SBART.utils.UserConfigs import (
    BooleanValue,
    DefaultValues,
    UserParam,
    Positive_Value_Constraint,
    IntegerValue,
)


class NormalizationBase(BASE):
    """
    **Description:**

    Extend a Spectrum object to allow to normalize the fluxes of S1D and S2D spectra.
    This functionality is extended  by allowing to fit & normalize the continuum levels and can be:

    1) applied to each order, independently;
    2) Applied to the combined S1D spectrum;

    In the second case, we can apply this process to the S2D spectra, even though the fit is made on the S1D products.
    To do this, the general approach is to first normalize the S1D spectra and, afterwards, divide it into
    order-sized chunks. This has the intended effect of re-generating the spectral mask and removing all
    previous rejections of spectral orders.

    """

    _object_type = "Spectral normalizer"

    _name = "SpecNormBase"
    _default_params = BASE._default_params + DefaultValues(
        NUMBER_WORKERS=UserParam(2, constraint=Positive_Value_Constraint + IntegerValue)
    )

    # If True, we will optimize the model for each spectral order! Otherwise, the NORMALIZER will receive
    # the entire S1D spectra as input, and it will be expected to return S2D matrices
    orderwise_application = True

    def __init__(self, obj_info: Dict[str, Any], user_configs, needed_folders=None):
        super().__init__(
            user_configs=user_configs, needed_folders=needed_folders, quiet_user_params=True
        )
        self._spec_info = obj_info
        self._ran_normalization_fit: bool = False

    def launch_epochwise_normalization(self, wavelengths, flux, uncertainties, loaded_info):
        self._ensure_epochwise_normalizer()
        if len(loaded_info) != 0:
            return *self._apply_epoch_normalization(
                wavelengths, flux, uncertainties, **loaded_info
            ), loaded_info
        return self._fit_epochwise_normalization(wavelengths, flux, uncertainties)

    def _apply_epoch_normalization(self, wavelengths, flux, uncertainties, **kwargs):
        ...

    def _fit_epochwise_normalization(self, wavelengths, flux, uncertainties):
        self._ran_normalization_fit = True

    def launch_orderwise_normalization(self, wavelengths, flux, uncertainties, loaded_info):
        """
        Launch a normalizer that will be applied to each spectral order (does not need to know order).
        This will:
        i) Directly apply the normalization from the loaded config values
        ii) Fit the model if it wasn't previously computed!

        Parameters
        ----------
        wavelengths
        flux
        uncertainties
        loaded_info

        Returns
        -------

        """
        self._ensure_orderwise_normalizer()
        self._normalization_sanity_checks()

        if len(loaded_info) != 0:
            return *self._apply_orderwise_normalization(
                wavelengths, flux, uncertainties, **loaded_info
            ), loaded_info
        return self._fit_orderwise_normalization(wavelengths, flux, uncertainties)

    def _fit_orderwise_normalization(self, wavelengths, flux, uncertainties):
        self._ensure_orderwise_normalizer()

    def _apply_orderwise_normalization(self, wavelengths, flux, uncertainties, **kwargs):
        self._ensure_orderwise_normalizer()

    def trigger_data_storage(self, *args, **kwargs) -> NoReturn:
        super().trigger_data_storage(args, kwargs)
        self._store_model_to_disk()

    def _normalization_sanity_checks(self):
        if self._spec_info["is_S1D"]:
            raise custom_exceptions.InvalidConfiguration("Can't normalize S1D spectra")

    def _ensure_orderwise_normalizer(self):
        """
        For internal usage. To call whenever we call a method to fit/apply normalization
        Returns
        -------

        """
        if not self.orderwise_application:
            raise custom_exceptions.InvalidConfiguration(
                f"Can't ask for order-wise normalization on {self.name}"
            )

    def _ensure_epochwise_normalizer(self):
        """
        For internal usage. To call whenever we call a method to fit/apply normalization
        Returns
        -------

        """
        if self.orderwise_application:
            raise custom_exceptions.InvalidConfiguration(
                f"Can't ask for epoch-wise normalization on {self.name}"
            )
