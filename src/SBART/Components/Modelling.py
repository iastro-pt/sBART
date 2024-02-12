from pathlib import Path
from loguru import logger
from typing import NoReturn, Dict

import numpy as np
from SBART.utils.BASE import BASE
from SBART.utils.UserConfigs import (
    DefaultValues,
    UserParam,
    ValueFromList,
)

from SBART.spectral_modelling import ScipyInterpolSpecModel, GPSpecModel
from SBART.utils.shift_spectra import apply_RVshift, remove_RVshift
from SBART.utils import custom_exceptions


class Spectral_Modelling(BASE):
    """

    Introduces, in a given object, the functionality to model and interpolate the stellar orders.
    In order to inherit from this class, it must also be a children of :class:`SBART.Components.SpectrumComponent.Spectrum`

    **User parameters:**

    ============================ ================ ================ ======================== ================
    Parameter name                 Mandatory      Default Value    Valid Values                 Comment
    ============================ ================ ================ ======================== ================
    INTERPOL_MODE                   False           splines         splines / GP / NN           [1]
    ============================ ================ ================ ======================== ================

    .. note::
        This flag will select which algorithm we will use to interpolate the spectra. Depending on the selection,
        we might want to pass extra-parameters, which can be set by passing a dictionary with the parameters
        defined in:
            - splines: :class:`SBART.Components.scipy_interpol.ScipyInterpolSpecModel`
            - GP: :class:`SBART.Components.GPSectralmodel.GPSpecModel`

        Those configuration are passed in different ways, depending on if we are dealing with Frames or
        a StellarModel object. The easy way to change them both is to call the following functions:
            -   DataClass.update_interpol_properties_of_all_frames
            -   DataClass.update_interpol_properties_of_stellar_model

    *Note:* Also check the **User parameters** of the parent classes for further customization options of SBART

    """

    # TODO: confirm the kernels that we want to allow
    _default_params = BASE._default_params + DefaultValues(
        INTERPOL_MODE=UserParam("splines", constraint=ValueFromList(("splines",))),
        # We have to add this here, so that the parameters are not rejected by the config validation
        SPLINE_TYPE=UserParam("cubic"),
        INTERPOLATION_ERR_PROP=UserParam("interpolation"),
        NUMBER_WORKERS=UserParam(1),
    )

    def __init__(self, **kwargs):
        self._default_params = self._default_params + Spectral_Modelling._default_params
        self.has_modelling_component = True
        super().__init__(**kwargs)

        if not self.has_spectrum_component:
            # TODO: ensure that it is safe to do this in here
            # TODO 1: won't this raise an Exception depending on the instantiation order???
            logger.critical("Can't add modelling component to class without a spectrum")
            raise Exception("Can't add modelling component to class without a spectrum")

        self.initialized_interface = False

        self._modelling_interfaces: Dict[str, "ModellingBase"] = {}

    def initialize_modelling_interface(self):
        if self.initialized_interface:
            return
        interface_init = {
            "obj_info": self.spectrum_information,
            "user_configs": self._internal_configs.get_user_configs(),
        }
        self._modelling_interfaces: Dict[str, "ModellingBase"] = {
            "GP": GPSpecModel(**interface_init),
            "splines": ScipyInterpolSpecModel(**interface_init),
        }

        if self._internalPaths.root_storage_path is None:
            logger.critical(
                "{self.name} launching modelling interface without a root path. Fallback to current directory"
            )
            self.generate_root_path(Path("."))

        for comp in self._modelling_interfaces.values():
            comp.generate_root_path(self._internalPaths.root_storage_path)

        self.initialized_interface = True

    @property
    def interpol_mode(self) -> str:
        return self._internal_configs["INTERPOL_MODE"]

    @property
    def interpolation_interface(self):
        self.initialize_modelling_interface()
        return self._modelling_interfaces[self.interpol_mode]

    def set_interpolation_properties(self, new_properties):
        self.initialize_modelling_interface()
        try:
            key = "INTERPOL_MODE"
            self._internal_configs.update_configs_with_values({key: new_properties[key]})
            logger.info(
                "Changing the interpolation mode of {} to {}", self.name, new_properties[key]
            )
        except KeyError as e:
            pass

        self.interpolation_interface.set_interpolation_properties(new_properties)

    def interpolate_spectrum_to_wavelength(
        self, order, new_wavelengths, shift_RV_by, RV_shift_mode, include_invalid=False
    ):
        self.initialize_modelling_interface()

        wavelength, flux, uncertainties, mask = self.get_data_from_spectral_order(
            order, include_invalid
        )
        desired_inds = ~mask

        og_lambda, og_spectra, og_errs = (
            wavelength[desired_inds],
            flux[desired_inds],
            uncertainties[desired_inds],
        )

        if RV_shift_mode == "apply":
            shift_function = apply_RVshift
        elif RV_shift_mode == "remove":
            shift_function = remove_RVshift
        else:
            raise custom_exceptions.InvalidConfiguration("Unknown mode")

        og_lambda = shift_function(wave=og_lambda, stellar_RV=shift_RV_by)

        try:
            new_flux, new_errors = self.interpolation_interface.interpolate_spectrum_to_wavelength(
                og_lambda=og_lambda,
                og_spectra=og_spectra,
                og_err=og_errs,
                new_wavelengths=new_wavelengths,
                order=order,
            )
        except custom_exceptions.StopComputationError as exc:
            logger.critical("Interpolation of {} has failed", self.name)
            raise exc

        return np.asarray(new_flux), np.asarray(new_errors)

    def trigger_data_storage(self, *args, **kwargs) -> NoReturn:
        super().trigger_data_storage(*args, **kwargs)
        for model_name, comp in self._modelling_interfaces.items():
            comp.trigger_data_storage()
