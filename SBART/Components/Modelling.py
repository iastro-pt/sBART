from typing import NoReturn
from SBART.Base_Models.BASE import BASE


class Spectral_Modelling(BASE):
    """

    Introduces, in a given object, the functionaitly to model and interpolate the stellar orders.
    In order to inherit from this class, it must also be a children of :class:`SBART.Components.SpectrumComponent.Spectrum`

    This is not yet implemented, so it is currently a placeholder that introduces no User parameters
    """

    def __init__(self, **kwargs):
        self._default_params = self._default_params + Spectral_Modelling._default_params
        self.has_modelling_component = True

        super().__init__(**kwargs)

    def load_previous_model_results_from_disk(self) -> NoReturn:
        raise NotImplementedError

    def generate_model_from_order(self, order: int) -> NoReturn:
        raise NotImplementedError

    def interpolate_spectrum_to_wavelength(self, order, new_wavelengths):
        raise NotImplementedError
