import pathlib
from typing import Union

from SBART.utils.custom_exceptions import InvalidConfiguration, NoDataError


def ensure_path_from_input(
    input_path: Union[str, pathlib.Path], ensure_existence: bool = False
) -> pathlib.Path:
    """
    Ensure that a given path is a pathlib.Path object

    Parameters
    -----------------
    input_path: Union[str, pathlib.Path]
        User-provided path
    ensure_existence: bool
        If True, raises an Exception if the input_path does not exists. By default False

    Raises
    --------
    InvalidConfiguration
        If the path is neither a string nor a pathlib.Path object
    NoDataError
        If the ensure_existence flag is set to True and input_path does not exist
    """

    if isinstance(input_path, str):
        input_path = pathlib.Path(input_path)

    if not isinstance(input_path, pathlib.Path):
        raise InvalidConfiguration("The user-provided path is neither a string nor a pathlib.Path")

    if ensure_existence:
        if not input_path.exists():
            raise NoDataError(f"The user-provided path <{input_path}> does not exist")

    return input_path
