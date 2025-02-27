from enum import Enum, auto



class SPECTRA_INTERPOL_MODE(Enum):
    """Enumerator to represent the DISK save mode of a given SBART object."""

    SPLINES=1
    GP=2

class SPLINE_INTERPOL_MODE(Enum):
    """Enumerator to represent the DISK save mode of a given SBART object."""

    CUBIC_SPLINE = 1
    QUADRATIC_SPLINE = 2
    PCHIP = 3
    NEAREST = 4 
    RBF = 5
