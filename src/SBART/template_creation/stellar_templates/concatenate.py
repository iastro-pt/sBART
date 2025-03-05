import time
import traceback
from multiprocessing import Process, Queue
from typing import Dict, Optional

import numpy as np
from loguru import logger
from tabletexifier import Table

from SBART.Base_Models.Frame import Frame
from SBART.Masks import Mask
from SBART.utils import custom_exceptions, open_buffer
from SBART.utils.concurrent_tools.close_interfaces import close_buffers, kill_workers
from SBART.utils.custom_exceptions import (
    BadOrderError,
    BadTemplateError,
    InvalidConfiguration,
)
from SBART.utils.RV_utilities.create_spectral_blocks import build_blocks
from SBART.utils.shift_spectra import remove_RVshift
from SBART.utils.status_codes import INTERNAL_ERROR, MISSING_DATA
from SBART.utils.units import convert_data, kilometer_second
from SBART.utils.UserConfigs import (
    DefaultValues,
    Positive_Value_Constraint,
    UserParam,
    ValueFromIterable,
)

from .Stellar_Template import StellarTemplate
from SBART.utils import choices


class ConcatenateStellar(StellarTemplate):
    """
    **User parameters:**

        This object doesn't introduce unique user parameters.

    *Note:* Check the **User parameters** of the parent classes for further customization options of this class

    """

    method_name = choices.STELLAR_CREATION_MODE.Concatenate.value

    _default_params = StellarTemplate._default_params + DefaultValues(
        ALIGNEMENT_RV_SOURCE=UserParam("DRS", constraint=ValueFromIterable(["DRS", "SBART"])),
        FLUX_threshold_for_template=UserParam(
            default_value=1,
            constraint=Positive_Value_Constraint,
            description="Flux threshold for masking the spectral template. Set to one to avoid possible numerical issues with near-zero values",
        ),
    )

    def __init__(self, subInst: str, user_configs: Optional[Dict] = None, loaded: bool = False):
        super().__init__(subInst=subInst, user_configs=user_configs, loaded=loaded)

        if not loaded:
            if self._internal_configs["MEMORY_SAVE_MODE"]:
                logger.warning(
                    "Stellar template creation will save RAM usage. This will result in multiple open/close "
                    "operations across the entire SBART pipeline! ",
                )

            self._found_error = False

    @custom_exceptions.ensure_invalid_template
    def create_stellar_template(self, dataClass, conditions=None) -> None:
        """Creating the stellar template"""
        # removal may change the first common wavelength; make sure
        try:
            super().create_stellar_template(dataClass, conditions)
        except custom_exceptions.StopComputationError:
            return

        instrument_information = dataClass.get_instrument_information()

        epoch_shape = instrument_information["array_size"]
        N_pix = epoch_shape[1]
        epoch_shape = (epoch_shape[0], epoch_shape[1] * len(self.frameIDs_to_use))
        self.array_size = epoch_shape
        # Create arrays of zeros in order to open in shared memory and change their values!
        self.spectra = np.zeros(epoch_shape)
        self.rejection_array = np.zeros((len(self.frameIDs_to_use), epoch_shape[0]))

        self.wavelengths = np.zeros(epoch_shape)
        self.uncertainties = np.zeros(epoch_shape)
        self.spectral_mask = None
        frame_RV_map = {  # construct map between the actual frames and the RVs. Done like this to allow being over-riden easily
            i: j for i, j in zip(self.frameIDs_to_use, self.sourceRVs)
        }

        full_mask = np.zeros(epoch_shape, dtype=bool)

        try:
            inst_info = dataClass.get_instrument_information()
            N_orders = inst_info["array_size"][0]

            for index, frameID in enumerate(self.frameIDs_to_use):
                wave_reference, flux, e, mask = dataClass.get_frame_arrays_by_ID(frameID)

                wavelengths = remove_RVshift(
                    wave_reference,
                    stellar_RV=convert_data(
                        frame_RV_map[frameID],
                        new_units=kilometer_second,
                        as_value=True,
                    ),
                )

                sl = slice(index * N_pix, (index + 1) * N_pix)
                for order in range(N_orders):
                    # Avoid near-one numerical
                    scale = 1000 / np.median(flux[order][~mask[order]])

                    self.wavelengths[order][sl] = wavelengths[order]
                    self.uncertainties[order][sl] = e[order] * scale
                    self.spectra[order][sl] = flux[order] * scale
                    full_mask[order][sl] = mask[order]

                if self._internal_configs["MEMORY_SAVE_MODE"]:
                    _ = dataClass.close_frame_by_ID(frameID)

            for order in range(N_orders):
                # Sort by wavelength inside the order
                inds = np.argsort(self.wavelengths[order])
                self.wavelengths[order] = self.wavelengths[order][inds]
                self.spectra[order] = self.spectra[order][inds]
                self.uncertainties[order] = self.uncertainties[order][inds]
                full_mask[order] = full_mask[order][inds]

            missing_points = np.where(self.wavelengths == 0)
            full_mask[missing_points] = True
            self.spectral_mask = Mask(full_mask, mask_type="binary")

            self.evaluate_bad_orders()
            self._finish_template_creation()

        except Exception as e:
            logger.opt(exception=True).critical("Stellar template creation failed due to: {}", e)
        finally:
            logger.info("Closing shared memory interfaces of the Stellar template")

    @property
    def RV_keyword(self) -> str:
        if self._internal_configs["ALIGNEMENT_RV_SOURCE"] == "SBART":
            RV_KW_start = "previous_SBART_RV"
        else:
            RV_KW_start = "DRS_RV"

        return RV_KW_start
