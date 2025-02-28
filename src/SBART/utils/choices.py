from enum import Enum, auto


class DISK_SAVE_MODE(Enum):
    """Enumerator to represent the DISK save mode of a given SBART object."""

    DISABLED = 1

    BASIC = 2

    EXTREME = 3


class WORKING_MODE(Enum):
    ONE_SHOT = "ONE_SHOT"
    ROLLING = "ROLLING"


class TELLURIC_EXTENSION(Enum):
    LINES = "LINES"
    WINDOW = "WINDOW"


class SPECTRA_INTERPOL_MODE(Enum):
    """Enumerator to represent the DISK save mode of a given SBART object."""

    SPLINES = 1
    GP = 2


class SPLINE_INTERPOL_MODE(Enum):
    """Enumerator to represent the DISK save mode of a given SBART object."""

    CUBIC_SPLINE = 1
    QUADRATIC_SPLINE = 2
    PCHIP = 3
    NEAREST = 4
    RBF = 5
