"""
This sub-package implements the different methods that can be used to normalize stellar spectra

"""
from .polynomial_normalization import Polynomial_normalization
from .RASSINE_normalization import RASSINE_normalization

available_normalization_interfaces = {
    "Poly-Norm": Polynomial_normalization,
    "RASSINE": RASSINE_normalization,
}
