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


class TELLURIC_CREATION_MODE(Enum):
    tapas = "tapas"
    telfit = "telfit"


class STELLAR_CREATION_MODE(Enum):
    Sum = "Sum"
    Concatenate = "Concatenate"
    OBSERVATION = "OBSERVATION"
    Median = "Median"
    PHOENIX = "PHOENIX"


class TELLURIC_APPLICATION_MODE(Enum):
    removal = "removal"
    correction = "correction"


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


class INTERPOLATION_ERR_PROP(Enum):
    """how should we propagate uncertainties"""

    interpolation = 1
    propagation = 2
    none = 3


class ORDER_REMOVAL_MODE(Enum):
    """how should we remove orders in the RV extraction"""

    per_subInstrument = 1
    GLOBAL = 2


class RV_EXTRACTION_MODE(Enum):
    """how should we remove orders in the RV extraction"""

    ORDER_WISE = 1
    EPOCH_WISE = 2
