import os

from loguru import logger

from SBART import __version__, __version_info__


def find_latest_version(path, enable_logs: bool = True) -> str:
    """
    Search, inside a directory, for all files with SBART versions. Returns the latest version found on disk

    Parameters
    ----------
    path

    Returns
    -------

    """
    available_cubes = []
    versions_full = []
    version_sum = []

    for filename in os.listdir(path):
        if os.path.isdir(os.path.join(path, filename)) or "_" not in filename:
            continue

        available_cubes.append(filename)
        cube_ver = filename.split("_")[-1].split(".")[0]
        versions_full.append(cube_ver)
        version_sum.append(
            sum([i * j for i, j in zip([100, 10, 1], map(int, cube_ver.split("-")))])
        )
    try:
        highest_ver = max(version_sum)
    except ValueError as exc:  # no version number in the list
        raise FileNotFoundError(f"There are no SBART outputs in {path}") from exc

    if highest_ver != sum([i * j for i, j in zip([100, 10, 1], __version_info__)]) and enable_logs:
        logger.warning(
            "\tRV cube is not the most recently installed version ({}). Using data from {}".format(
                __version__, versions_full[version_sum.index(highest_ver)]
            )
        )

    return versions_full[version_sum.index(highest_ver)]


if __name__ == "__main__":
    print(
        find_latest_version(
            "/home/amiguel/seminar/validate_SBART/outputs/ESPRESSO/Suite_Small/RV_step"
        )
    )
