from pathlib import Path
from loguru import logger
from typing import NoReturn, Dict

import numpy as np
from SBART.Base_Models.BASE import BASE
from SBART.utils.UserConfigs import (
    DefaultValues,
    UserParam,
    ValueFromList, BooleanValue
)

from SBART.spectral_normalization.normalization_base import NormalizationBase
from SBART.spectral_modelling import ScipyInterpolSpecModel
from SBART.utils.shift_spectra import apply_RVshift, remove_RVshift
from SBART.utils import custom_exceptions


class Spectral_Normalization(BASE):
    """

    Introduces, in a given object, the functionality to normalize the continuum level.
    In order to inherit from this class, it must also be a children of :class:`SBART.Components.SpectrumComponent.Spectrum`

    **User parameters:**

    ============================ ================ ================ ======================== ================
    Parameter name                 Mandatory      Default Value    Valid Values                 Comment
    ============================ ================ ================ ======================== ================
    NORMALIZATION_MODE               False           RASSINE                                   [1]
    ============================ ================ ================ ======================== ================

    *Note:* Also check the **User parameters** of the parent classes for further customization options of SBART

    """

    # TODO: confirm the kernels that we want to allow
    _default_params = BASE._default_params + DefaultValues(
        NORMALIZE_SPECTRA=UserParam(False, constraint=BooleanValue),
        NORMALIZATION_MODE=UserParam("RASSINE", constraint=ValueFromList(("RASSINE",)))
    )

    def __init__(self, **kwargs):
        self._default_params = self._default_params + Spectral_Normalization._default_params
        self.has_normalization_component = True
        super().__init__(**kwargs)

        if not self.has_spectrum_component:
            msg = "Can't add modelling component to class without a spectrum"
            logger.critical(msg)
            raise Exception(msg)

        self.initialized_normalization_interface = False

        self._normalization_interfaces: Dict[str, NormalizationBase] = {}

    def initialize_modelling_interface(self):
        if self.initialized_normalization_interface:
            return
        interface_init = {"obj_info": self.spectrum_information,
                          "user_configs": self._internal_configs.get_user_configs()
                          }
        self._normalization_interfaces: Dict[str,] = {
            "splines": ScipyInterpolSpecModel(**interface_init),
        }

        if self._internalPaths.root_storage_path is None:
            logger.critical("{self.name} launching modelling interface without a root path. Fallback to current directory")
            self.generate_root_path(Path("."))

        for comp in self._modelling_interfaces.values():
            comp.generate_root_path(self._internalPaths.root_storage_path)

        self.initialized_interface = True

    def normalize_spectra(self):
        """
        TODO: See if we need to paralelize this!

        Launch the normalization of the spectra, using the selected algorithm
        Returns
        -------

        """
        if not self._internal_configs["NORMALIZE_SPECTRA"]:
            return

        norm_interface = self._normalization_interfaces[self._internal_configs["NORMALIZATION_MODE"]]

        for order in range(self.N_orders):
            wavelengths, flux, uncerts, mask = self.get_data_from_spectral_order(order,
                                                                                 include_invalid=True
                                                                                 )

            new_flux, new_uncerts = norm_interface.launch_normalization(wavelengths,
                                                                        flux,
                                                                        uncerts
                                                                        )
            self.flux[order] = new_flux
            self.uncertainties[order] = new_uncerts

    def trigger_data_storage(self, *args, **kwargs) -> NoReturn:
        super().trigger_data_storage(*args, **kwargs)
        for model_name, comp in self._normalization_interfaces.items():
            comp.trigger_data_storage()
