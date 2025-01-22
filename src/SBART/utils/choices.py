from enum import Enum, auto

# class syntax


class DISK_SAVE_MODE(Enum):
    """Enumerator to represent the DISK save mode of a given SBART object."""

    DISABLED = 1

    BASIC = 2

    EXTREME = 3


class WORKING_MODE(Enum):
    ONE_SHOT = "ONE_SHOT"
    ROLLING = "ROLLING"
