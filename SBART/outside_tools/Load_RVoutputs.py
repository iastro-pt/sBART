import pathlib

from SBART.data_objects.RV_outputs import RV_holder
from SBART.utils.paths_tools import find_latest_version


def find_RVoutputs(
    path, load_full_flags=False, load_work_pkgs=False, enable_logs: bool = True, SBART_version=None
):
    """Search, inside a given path, for the RV cube that comes from the latest version of S-BART

    Parameters
    ----------
    load_work_pkgs:
        Load outputs from the workers. By default, False, as they tend to be "heavy"
    load_full_flags:
        Load full information of all order-wise Flags. By default, False, as we don't always need this
    path : str
        Path in which the search will be done. If there are multiple cubes (from different S-BART versions), it will
        return the one from the most recent version
    enable_logs : bool, optional
        If True, it will print messages. By default, True

    Returns
    -------
    RV_holder
        Rv holder with the cubes loaded in!

    Raises
    ------
    FileNotFoundError
        If there is no RV cube stored in folder

    """

    found = False
    most_recent_version = find_latest_version(path, enable_logs=enable_logs)

    outs = RV_holder.load_from_disk(
        pathlib.Path(path),
        load_work_pkgs=load_work_pkgs,
        load_full_flags=load_full_flags,
        SBART_version=SBART_version if SBART_version is not None else most_recent_version,
    )
    return outs
