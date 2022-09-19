from pathlib import Path
from loguru import logger
from typing import NoReturn, Dict, Optional

import numpy as np
from SBART.Base_Models.BASE import BASE
from SBART.utils.UserConfigs import (
    DefaultValues,
    UserParam,
    ValueFromList, BooleanValue
)

from SBART.spectral_normalization.normalization_base import NormalizationBase
from SBART.spectral_normalization import available_normalization_interfaces
from SBART.utils.shift_spectra import apply_RVshift, remove_RVshift
from SBART.utils import custom_exceptions
from SBART.DataUnits import SpecNorm_Unit


class Spectral_Normalization(BASE):
    """

    Introduces, in a given object, the functionality to normalize the continuum level.
    In order to inherit from this class, it must also be a children of :class:`SBART.Components.SpectrumComponent.Spectrum`

    **User parameters:**

    ============================ ================ ================ ======================== ================
    Parameter name                 Mandatory      Default Value    Valid Values                 Comment
    ============================ ================ ================ ======================== ================
    NORMALIZATION_MODE               False           Alpha-Shape                                   [1]
    ============================ ================ ================ ======================== ================

    *Note:* Also check the **User parameters** of the parent classes for further customization options of SBART

    """

    # TODO: confirm the kernels that we want to allow
    _default_params = BASE._default_params + DefaultValues(
        NORMALIZE_SPECTRA=UserParam(False, constraint=BooleanValue),
        NORMALIZATION_MODE=UserParam("Alpha-Shape", constraint=ValueFromList(list(available_normalization_interfaces.keys())))
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
        self._already_normalized_data = False
        self._normalization_interfaces: Dict[str, NormalizationBase] = {}
        self._normalization_information: Optional[SpecNorm_Unit] = None

    def initialize_normalization_interface(self):
        if self.initialized_normalization_interface:
            return
        interface_init = {"obj_info": self.spectrum_information,
                          "user_configs": self._internal_configs.get_user_configs()
                          }

        self._normalization_interfaces: Dict[str,] = {
            key: value(**interface_init) for key, value in available_normalization_interfaces.items()
        }

        if self._internalPaths.root_storage_path is None:
            logger.critical("{self.name} launching modelling interface without a root path. Fallback to current directory")
            self.generate_root_path(Path("."))

        for comp in self._modelling_interfaces.values():
            comp.generate_root_path(self._internalPaths.root_storage_path)

        try: # Generate class to store the normalization parameters
            self._normalization_information = SpecNorm_Unit.load_from_disk(
                self._internalPaths.root_storage_path,
                filename=self.fname.split(".fits")[0],
                algo_name=self._internal_configs["NORMALIZATION_MODE"]
            )
        except custom_exceptions.NoDataError:
            logger.warning("Can't find previous normalization parameters on disk!")
            self._normalization_information = SpecNorm_Unit(frame_name=self.fname.split(".fits")[0],
                                                            algo_name=self._internal_configs["NORMALIZATION_MODE"]
                                                            )
            self._normalization_information.generate_root_path(self._internalPaths.root_storage_path)
        self.initialized_normalization_interface = True

    def normalize_spectra(self):
        """
        TODO: See if we need to paralelize this!

        Launch the normalization of the spectra, using the selected algorithm
        Returns
        -------

        """
        if not self._internal_configs["NORMALIZE_SPECTRA"]:
            logger.warning("<NORMALIZE_SPECTRA> option has been disabled by the user")
            return
        if self._already_normalized_data:
            logger.warning("{} is already normalized; Doing nothing!", self.name)
            return
        self.initialize_normalization_interface()

        norm_interface = self._normalization_interfaces[self._internal_configs["NORMALIZATION_MODE"]]
        # TODO: see if we want to parallelize this!

        for order in range(self.N_orders):
            wavelengths, flux, uncerts, mask = self.get_data_from_spectral_order(order,
                                                                                 include_invalid=True
                                                                                 )

            new_flux, new_uncerts, norm_keys = norm_interface.launch_normalization(wavelengths=wavelengths,
                                                                                   flux=flux,
                                                                                   uncertainties=uncerts,
                                                                                   loaded_info=self._normalization_information.get_norm_info_from_order(
                                                                                       order
                                                                                       )
                                                                                   )
            self.spectra[order] = new_flux
            self.uncertainties[order] = new_uncerts

            self._normalization_information.store_norm_info(order, norm_keys)

        self._already_normalized_data = True

    def trigger_data_storage(self, *args, **kwargs) -> NoReturn:
        super().trigger_data_storage(*args, **kwargs)
        self._normalization_information.trigger_data_storage()