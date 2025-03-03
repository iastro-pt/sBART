from typing import Union

import numpy as np
from astropy.io import fits
from loguru import logger

from SBART.Masks import Mask
from SBART.utils import custom_exceptions
from SBART.utils.UserConfigs import PathValue, UserParam, DefaultValues

from .Stellar_Template import StellarTemplate


class PHOENIX(StellarTemplate):
    """Stellar template from a PHOENIX template"""

    method_name = "PHOENIX"
    _default_params = StellarTemplate._default_params + DefaultValues(
        PHOENIX_FILE_PATH=UserParam("", constraint=PathValue, mandatory=False),
    )

    def __init__(
        self, subInst: str, user_configs: Union[None, dict] = None, loaded: bool = False
    ):
        super().__init__(subInst=subInst, user_configs=user_configs, loaded=loaded)
        self._found_error = False

    @custom_exceptions.ensure_invalid_template
    def create_stellar_template(self, dataClass, conditions=None) -> None:
        """Creating the stellar template."""
        # removal may change the first common wavelength; make sure
        try:
            super().create_stellar_template(dataClass, conditions)
        except custom_exceptions.StopComputationError:
            return

        logger.info("Searching for frameID with highest sum of orderwise SNRs")

        with fits.open(self._internal_configs["PHOENIX_FILE_PATH"]) as hdu:
            data = hdu[1].data

        logger.info("Selected frameID={}", self._selected_frameID)

        self.wavelengths = data["wavelengths"]
        self.spectra = data["flux"] * 10_000
        self.uncertainties = np.zeros(self.spectra.shape)
        mask = np.zeros_like(self.spectra, dtype=bool)
        self.spectral_mask = Mask(mask, mask_type="binary")

        self._finish_template_creation()

    def store_metrics(self): ...
