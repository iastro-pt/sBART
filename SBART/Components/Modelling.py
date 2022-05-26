from pathlib import Path
from typing import NoReturn

from SBART.Base_Models.BASE import BASE
from SBART.utils.UserConfigs import (
    DefaultValues,
    UserParam,
    ValueFromList,
)

from SBART.spectral_modelling import GPSpecModel, ScipyInterpolSpecModel


class Spectral_Modelling(BASE):
    """

    Introduces, in a given object, the functionality to model and interpolate the stellar orders.
    In order to inherit from this class, it must also be a children of :class:`SBART.Components.SpectrumComponent.Spectrum`

    **User parameters:**

    ============================ ================ ================ ======================== ================
    Parameter name                 Mandatory      Default Value    Valid Values                 Comment
    ============================ ================ ================ ======================== ================
    INTERPOL_MODE                   False           splines              splines / GP           [1]
    ============================ ================ ================ ======================== ================

    - [1] This flag will select which algorithm we will use to interpolate the spectra. Depending on the selection,
    we might want to pass extra-parameters, as defined in:
        - splines: :class:`SBART.Components.scipy_interpol.ScipyInterpolSpecModel`
        - GP: :class:`SBART.Components.GPSectralmodel.GPSpecModel`


    *Note:* Also check the **User parameters** of the parent classes for further customization options of SBART

    """

    # TODO: confirm the kernels that we want to allow
    _default_params = BASE._default_params + DefaultValues(
        INTERPOL_MODE=UserParam("splines", constraint=ValueFromList(("splines", "GP")))
    )

    def __init__(self, **kwargs):
        self._default_params = self._default_params + Spectral_Modelling._default_params
        self.has_modelling_component = True
        super().__init__(**kwargs)

        if not self.has_spectrum_component:
            # TODO: ensure that it is safe to do this in here
            # TODO 1: won't this raise an Exception depending on the instantiation order???
            raise Exception("Can't add modelling component to class without a spectrum")

        interface_init = {"obj_info": self.spectrum_information,
                          "user_configs": kwargs["user_configs"]
                          }

        self._modelling_interfaces = {
            "GP": GPSpecModel(**interface_init),
            "splines": ScipyInterpolSpecModel(**interface_init)
        }

    def generate_root_path(self, storage_path: Path) -> NoReturn:
        super().generate_root_path(storage_path)
        for comp in self._modelling_interfaces.values():
            comp.generate_root_path(storage_path)

    @property
    def interpol_mode(self) -> str:
        return self._internal_configs["INTERPOL_MODE"]

    def interpolate_spectrum_to_wavelength(self, order):
        # TODO: implement this!
        return self._modelling_interfaces[self.interpol_mode]
