from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
from astropy.io import fits

from SBART.utils import choices, custom_exceptions
from SBART.utils.choices import DISK_SAVE_MODE
from SBART.utils.UserConfigs import (
    DefaultValues,
    UserParam,
    ValueFromDtype,
)

from .Telluric_Template import TelluricTemplate

if TYPE_CHECKING:
    from SBART.data_objects.DataClass import DataClass
    from SBART.utils.SBARTtypes import UI_DICT

RESOURCES_PATH = Path(__file__).parent.parent.parent / "resources"


class OHemissionTelluric(TelluricTemplate):
    """ """

    _default_params = TelluricTemplate._default_params + DefaultValues(
        SKYcalcPath=UserParam(
            default_value=None,
            constraint=ValueFromDtype((str, Path, type(None))),
            description=("Path to a fits file provided from ESO's skycalc tool for the correct MJD"),
        ),
    )

    method_name = choices.TELLURIC_CREATION_MODE.OHemission.value

    def __init__(
        self,
        subInst: str,
        user_configs: Optional[UI_DICT] = None,
        extension_mode: str = "lines",
        application_mode: str = "removal",
        loaded: bool = False,
    ):
        super().__init__(
            subInst=subInst,
            extension_mode=extension_mode,
            user_configs=user_configs,
            loaded=loaded,
            application_mode=application_mode,
        )

    @custom_exceptions.ensure_invalid_template
    def create_telluric_template(self, dataClass: DataClass, custom_frameID: Optional[int] = None) -> None:
        """Create a telluric template from a TelFit transmission spectra [1].

        The model is created for the date in which the reference observation was made.

        It estimates the continuum level and classifies each point that shows a
        decrease of 1% as a telluric line. Furthermore, it creates a window of
        6 points to each side of a detection, to attempt to pick up the wings of
        the telluric line.

        Parameters
        ----------
        dataClass:
            DataClass object
        custom_frameID :
            If Not None, does not search for the "optimal" frameID to use as a basis

        Returns
        -------
        numpy.ndarray
            Telluric (binary) spectrum, for the wavelengths present in the input
            array

        Notes
        -----
        [1] https://github.com/kgullikson88/Telluric-Fitter

        """
        try:
            super().create_telluric_template(dataClass, custom_frameID=custom_frameID)
        except custom_exceptions.StopComputationError:
            return

        with fits.open(self._internal_configs["SKYcalcPath"]) as hdu:
            datable = hdu[1].data

        wavelengths, tell_spectra = datable["lam"], datable["flux"]

        template = np.zeros_like(tell_spectra)
        template[np.where(tell_spectra != 0)] = 1
        self.template = template

        # ! no median filtering (might still be needed in the future)
        self._continuum_level = 1.0
        self.wavelengths = wavelengths * 10  # convert to the prevalent wavelength units

        self.transmittance_wavelengths, self.transmittance_spectra = (
            wavelengths,
            tell_spectra,
        )

        self.build_blocks()
        self._compute_wave_blocks()
        self._finish_template_creation()

    def store_metrics(self):
        super().store_metrics()
