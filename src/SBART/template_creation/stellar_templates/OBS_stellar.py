from typing import Union

import numpy as np
from loguru import logger

from SBART.Masks import Mask
from SBART.utils import custom_exceptions

from .Stellar_Template import StellarTemplate


class OBS_Stellar(StellarTemplate):
    """
    Stellar template from the observation with the highest SNR (computed as a sum over all spectral orders)
    """

    method_name = "OBSERVATION"

    def __init__(self, subInst: str, user_configs: Union[None, dict] = None, loaded: bool = False):
        super().__init__(subInst=subInst, user_configs=user_configs, loaded=loaded)

        self._selected_frameID = None
        self._found_error = False

    @custom_exceptions.ensure_invalid_template
    def create_stellar_template(self, dataClass, conditions=None) -> None:
        """
        Creating the stellar template
        """
        # removal may change the first common wavelength; make sure
        try:
            super().create_stellar_template(dataClass, conditions)
        except custom_exceptions.StopComputationError:
            return

        logger.info("Searching for frameID with highest sum of orderwise SNRs")
        total_SNR = []
        for frameID in self.frameIDs_to_use:
            total_SNR.append(
                np.nansum(dataClass.get_KW_from_frameID(KW="orderwise_SNRs", frameID=frameID))
            )

        self._selected_frameID = self.frameIDs_to_use[np.argmax(total_SNR)]
        logger.info("Selected frameID={}", self._selected_frameID)
        wavelenghts, spectra, uncertainties, mask = dataClass.get_frame_arrays_by_ID(
            self._selected_frameID
        )

        self.wavelengths = wavelenghts
        self.spectra = spectra
        self.uncertainties = uncertainties

        self.spectral_mask = Mask(mask, mask_type="binary")

        self._finish_template_creation()
