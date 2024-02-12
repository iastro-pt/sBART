from pathlib import Path
from loguru import logger
from typing import NoReturn, Dict, Optional

import numpy as np
from SBART.utils.BASE import BASE
from SBART.utils.UserConfigs import DefaultValues, UserParam, ValueFromList, BooleanValue, PathValue

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
    NORMALIZATION_MODE               False          RASSINE                                    [1]
    ============================ ================ ================ ======================== ================

    Notes:

        [1] Name of the spectral normalizers, that are described in :mod:`SBART.spectral_normalization`

    *Note:* Also check the **User parameters** of the parent classes for further customization options of SBART

    """

    # TODO: confirm the kernels that we want to allow
    _default_params = BASE._default_params + DefaultValues(
        NORMALIZE_SPECTRA=UserParam(False, constraint=BooleanValue),
        NORMALIZATION_MODE=UserParam(
            "RASSINE", constraint=ValueFromList(list(available_normalization_interfaces.keys()))
        ),
        S1D_folder=UserParam(mandatory=False, constraint=PathValue, default_value=""),
        RASSINE_path=UserParam(mandatory=False, constraint=PathValue, default_value=""),
    )

    def __init__(self, **kwargs):
        self._default_params = self._default_params + Spectral_Normalization._default_params
        self.has_normalization_component = True
        super().__init__(**kwargs)

        if not self.has_spectrum_component:
            msg = "Can't add modelling component to class without a spectrum"
            logger.critical(msg)
            raise Exception(msg)

        self._already_normalized_data = False
        self._normalization_interfaces: Dict[str, NormalizationBase] = {}
        self._normalization_information: Optional[SpecNorm_Unit] = None

    def initialize_normalization_interface(self) -> NoReturn:
        """
        Initialize the normalization interface for the currently selected mode!
        Returns
        -------

        """
        key = self._internal_configs["NORMALIZATION_MODE"]

        if key in self._normalization_interfaces:
            return

        interface_init = {
            "obj_info": self.spectrum_information,
            "user_configs": self._internal_configs.get_user_configs(),
        }

        extra_info = {}
        interface_init["obj_info"]["S1D_name"] = self.get_S1D_name()
        interface_init["obj_info"]["frame_path"] = self.file_path
        interface_init["obj_info"]["Frame_instance"] = type(self)

        interface = available_normalization_interfaces[key]

        self._normalization_interfaces[key] = interface(**interface_init)

        if self._internalPaths.root_storage_path is None:
            logger.critical(
                f"{self.name} launching normalization interface without a root path. Fallback to current directory"
            )
            self.generate_root_path(Path("."))

        self._normalization_interfaces[key].generate_root_path(
            self._internalPaths.root_storage_path
        )

        current_frame_name = self.fname.split(".fits")[0]
        try:  # Generate class to store the normalization parameters
            self._normalization_information = SpecNorm_Unit.load_from_disk(
                self._internalPaths.root_storage_path,
                filename=current_frame_name,
                algo_name=self._internal_configs["NORMALIZATION_MODE"],
            )
        except custom_exceptions.NoDataError:
            logger.warning("Can't find previous normalization parameters on disk!")
            self._normalization_information = SpecNorm_Unit(
                frame_name=current_frame_name,
                algo_name=self._internal_configs["NORMALIZATION_MODE"],
            )
            self._normalization_information.generate_root_path(
                self._internalPaths.root_storage_path
            )

    def normalize_spectra(self):
        """
        TODO: See if we need to parallelize this!

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

        norm_interface = self._normalization_interfaces[
            self._internal_configs["NORMALIZATION_MODE"]
        ]
        if norm_interface.orderwise_application:
            self.trigger_orderwise_method(norm_interface)
        else:
            self.trigger_epochwise_method(norm_interface)

    def trigger_epochwise_method(self, norm_interface):
        name = "S1D"
        loaded_info = self._normalization_information.get_norm_info_from_order(name)

        wavelengths, flux, uncerts, _ = self.get_data_from_full_spectrum()

        new_waves, new_flux, new_uncert, norm_keys = norm_interface.launch_epochwise_normalization(
            wavelengths=wavelengths,
            flux=flux,
            uncertainties=uncerts,
            loaded_info=loaded_info,
        )
        self.wavelengths = new_waves.reshape(wavelengths.shape)
        self.spectra = new_flux.reshape(wavelengths.shape)
        self.uncertainties = new_uncert.reshape(wavelengths.shape)
        logger.warning("Epoch wise normalization is overriding the minimum SNR!")
        self._internal_configs["minimum_order_SNR"] = 0
        self.regenerate_order_status()
        self._normalization_information.store_norm_info(name, norm_keys)

        self._already_normalized_data = True
        # Trigger a new check of the data integrity, as we have just overloaded the entire
        # S2D spectrum. However, this ignores any kind of quality check!

    def trigger_orderwise_method(self, norm_interface):
        # TODO: see if we want to parallelize this!
        for order in range(self.N_orders):
            wavelengths, flux, uncerts, mask = self.get_data_from_spectral_order(
                order, include_invalid=True
            )

            mask_to_use = ~mask
            loaded_info = self._normalization_information.get_norm_info_from_order(order)

            new_flux, new_uncerts, norm_keys = norm_interface.launch_orderwise_normalization(
                wavelengths=wavelengths[mask_to_use],
                flux=flux[mask_to_use],
                uncertainties=uncerts[mask_to_use],
                loaded_info=loaded_info,
            )
            self.spectra[order][mask_to_use] = new_flux
            self.uncertainties[order][mask_to_use] = new_uncerts

            self._normalization_information.store_norm_info(order, norm_keys)

        self._already_normalized_data = True

    def trigger_data_storage(self, *args, **kwargs) -> NoReturn:
        super().trigger_data_storage(*args, **kwargs)
        if self._normalization_information is not None:
            self._normalization_information.trigger_data_storage()
