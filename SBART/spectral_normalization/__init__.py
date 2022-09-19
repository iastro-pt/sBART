"""
This sub-package implements the different methods that can be used to interpolate stellar spectra
to a new wavelength grid.

"""
from .alpha_shape_normalization import AlphaShape_normalization
from .polynomial_normalization import Polynomial_normalization

available_normalization_interfaces = {"Alpha-Shape": AlphaShape_normalization,
                                      "Poly-Norm": Polynomial_normalization
                                      }