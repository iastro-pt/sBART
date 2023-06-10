from pathlib import Path
from typing import Optional

from SBART import __version__


def build_filename(og_path: Path, filename, fmt, SBART_version: Optional[str] = None) -> str:
    """
    Standardize the filenames of sBART outputs

    Parameters
    ==============
    og_path : str
        path in which the file will be stored
    filename: str
        name of the file
    fmt: str
        format of the file, e.g. pickle, str
    """
    SBART_version = SBART_version if SBART_version is not None else __version__
    return (og_path / f"{filename}_TM_{SBART_version}.{fmt}").as_posix()
