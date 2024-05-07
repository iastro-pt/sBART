import ujson as json
from pathlib import Path

from loguru import logger

from SBART.utils import build_filename, custom_exceptions
from SBART.utils.json_ready_converter import json_ready_converter


class MetaData:
    """
    Class to hold metadata information of the processed data.

    The Frames, when loading data from disk will collect information to work as MetaData, which will be used to provide an  option
    of avoiding the repeated computation of exactly the same dataset. This class provides an easy interface to hold such data,
    write and load it to disk, and compare equality between two MetaData objects.

    The data stored inside this object is divided by subInstruments, with each having a unique set of values, that is not shared with
    other subInstruments. Comparisons of equality are supported at the subInstrument level.

    """

    def __init__(self, data_to_hold=None):
        self.information = {} if data_to_hold is None else data_to_hold

    def subInstrument_comparison(self, other, subInstrument: str) -> bool:
        """
        Compare the Metadata of a given subInstrument. CHecks that must pass for equality:

        * The same keys are present in the two objects
        * A given key has the same value in the two objects

        Parameters
        ----------
        other:
            Other MetaData object
        subInstrument: str
            SubInstrument to compare

        Returns
        -------
        comparison_result: bool
            True if the MetaData matchess
        """
        equal = True

        try:
            for key, subInst_value in self.information[subInstrument].items():
                if subInst_value != other.information[subInstrument][key]:
                    equal = False
                    break
        except KeyError as e:
            equal = False

        return equal

    def store_json(self, path: Path):
        """
        Store the class as a json files

        Parameters
        ----------
        path:
            Path in which this object will be stored

        Returns
        -------

        """
        logger.debug("Storing Metadata to {}", path)

        storage_path = build_filename(path, "MetaData", fmt="json")

        info_to_store = {}
        for key, value in self.information.items():
            value = json_ready_converter(value)
            info_to_store[key] = value

        with open(storage_path, mode="w") as handle:
            json.dump(info_to_store, handle, indent=4)

    def add_info(self, key, info, subInstrument):
        """
        Add a new metric to be tracked, with the values being collected over all available frames.

        Parameters
        ----------
        key: str
            Name of the metric
        info: Union[str, Iterable]
            data to be stored
        subInstrument: str
            subInstrument to which the info belongs to

        Returns
        -------

        Raises
        ------
        TypeError
            If the info is not a list nor an iterable
        """
        if not isinstance(info, (str, tuple, list)):
            raise TypeError("info must be  str or list object")
        if subInstrument not in self.information:
            self.information[subInstrument] = {}

        # remove duplicates, but keep as a list (to avoid problems when storing to json)!
        self.information[subInstrument][key] = list(set(info))

    @classmethod
    def load_from_json(cls, path):
        storage_path = build_filename(
            path,
            "MetaData",
            fmt="json",
            SBART_version=None,  # we always want the latest version
        )
        try:
            with open(storage_path) as handle:
                information = json.load(handle)
        except FileNotFoundError:
            msg = "Failed to find metadata file in {}".format(storage_path)
            logger.warning(msg)
            raise custom_exceptions.StopComputationError(msg)

        return MetaData(information)


if __name__ == "__main__":
    A = MetaData()
    B = MetaData()
    A.add_info("a", [1, 2, 3], "dd")
    B.add_info("a", [2, 1], "dd")

    print(A.information)
    print(A.subInstrument_comparison(B, "dd"))
