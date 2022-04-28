"""
Introduce extra functionalities in Classes.

A class that inherits from the Components will have new functionalities.

- :py:class:`~SBART.Components.SpectrumComponent.Spectrum`: Class will be able of storing spectral data (independently of format).
- :py:class:`~SBART.Components.Modelling.Spectral_Modelling`: Children will be able to model stellar spectra.
 Yet to be **implemented**
"""

from .Modelling import Spectral_Modelling
from .SpectrumComponent import Spectrum
