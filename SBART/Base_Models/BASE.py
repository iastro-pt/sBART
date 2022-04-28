from pathlib import Path
from typing import Any, Dict, NoReturn, Optional

from loguru import logger

from SBART.utils import custom_exceptions
from SBART.utils.paths_tools.PathsHandler import Paths
from SBART.utils.status_codes import Flag, Status
from SBART.utils.types import UI_DICT, UI_PATH
from SBART.utils.UserConfigs import DefaultValues, InternalParameters


class BASE:
    """
    Base class, for almost all of SBART objects.

    Inheriting from this class brings a common interface for SBART's User parameters and handling of disk paths.

    """

    _object_type = ""
    _name = ""

    _default_params = DefaultValues()

    def __init__(
        self,
        user_configs: Optional[UI_DICT] = None,
        root_level_path: Optional[UI_PATH] = None,
        needed_folders: Optional[Dict[str, str]] = None,
        start_with_valid_status: bool = True,
    ):

        self._internal_configs = InternalParameters(self.name, self._default_params)

        if user_configs is None:
            user_configs = {}

        self._internalPaths = Paths(
            root_level_path=root_level_path, preconfigured_paths=needed_folders
        )

        self._internal_configs.receive_user_inputs(user_configs)
        self._needed_folders = needed_folders
        self._status = Status(assume_valid=start_with_valid_status)  # BY DEFAULT IT IS A VALID ONE!

    ###
    #   Data storage
    ###

    def trigger_data_storage(self, *args, **kwargs) -> NoReturn:
        logger.info("{} storing data to disk", self.name)
        pass

    def json_ready(self) -> Dict[str, Any]:
        """
        Concerts the current class into a json entry that can be easily saved/loaded to/from disk
        Returns
        -------

        """
        return {}

    def generate_root_path(self, storage_path: Path) -> NoReturn:
        """ """
        logger.debug("Setting root path of {} to {}", self.name, storage_path)
        self._internalPaths.add_root_path(storage_path)

    def add_relative_path(self, path_name, relative_structure):
        self._internalPaths.add_relative_path(path_name, relative_structure)

    ###
    #   Handling the status of the sBART objects
    ###

    def add_to_status(self, new_flag: Flag) -> NoReturn:
        self._status.store_flag(new_flag=new_flag)

    def _data_access_checks(self) -> NoReturn:
        """
        Ensure that the status of the sBART object is valid. This is a very broad check of validity that is overloaded
        in multiple places in the code
        Returns
        -------

        """
        if not self.is_valid:
            raise custom_exceptions.InvalidConfiguration(
                "Attempting to access data from sBART object that is not "
                "valid. Check previous log messages for further information: {}".format(
                    self._status
                )
            )

    def load_from_file(self, root_path, loading_path: str) -> None:
        self.generate_root_path(root_path)

    ###
    #  Properties
    ###
    def is_object_type(self, type_to_check: str) -> bool:
        """
        Check if this object is of a given type

        Parameters
        ----------
        type_to_check

        Returns
        -------

        """
        return self._object_type == type_to_check

    @property
    def is_valid(self) -> bool:
        return self._status.is_valid

    @property
    def name(self) -> str:
        return "{} - {}".format(self.__class__._object_type, self.__class__._name)

    @property
    def storage_name(self) -> str:
        return self.__class__._name

    @classmethod
    def control_parameters(cls):
        return cls._default_params.keys()
