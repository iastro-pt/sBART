from pathlib import Path
from typing import Dict, NoReturn, Optional, Union

from SBART.utils import custom_exceptions


class Paths:
    def __init__(
        self,
        root_level_path: Optional[Union[str, Path]] = None,
        preconfigured_paths: Optional[Dict[str, str]] = None,
    ):
        self._folder_mappings = preconfigured_paths if preconfigured_paths is not None else {}

        if isinstance(root_level_path, str):
            root_level_path = Path(root_level_path)

        self._root_path: Optional[Path] = root_level_path if root_level_path is not None else None

        # For lazy creation of the folders and to avoid multiple creation attempts
        self._constructed_folders = set()

    def add_relative_path(self, folder_KW: str, rel_path: str) -> NoReturn:
        """
        Add a relative (to self._root_path) folder, associated with a given Keyword
        Parameters
        ----------
        folder_KW
        rel_path

        Returns
        -------

        """
        self._folder_mappings[folder_KW] = rel_path

    def get_path_to(
        self, folder_tag: str, absolute: bool = True, as_posix: bool = True
    ) -> Union[str, Path]:
        if self._root_path is None:
            raise custom_exceptions.MissingRootPath(
                "Must provide the root level path before asking for other paths"
            )

        if folder_tag == "ROOT":
            out_path = self._root_path.absolute()
        else:
            out_path = self._folder_mappings[folder_tag]

            if absolute:
                out_path = self._root_path / out_path

        if folder_tag not in self._constructed_folders:
            out_path.mkdir(parents=True, exist_ok=True)
            self._constructed_folders.add(folder_tag)

        if as_posix:
            out_path = out_path.as_posix()

        return out_path

    def add_root_path(
        self, path: Union[str, Path], current_folder_name: Optional[str] = None
    ) -> NoReturn:
        if isinstance(path, str):
            path = Path(path)

        if current_folder_name is not None:
            path = path / current_folder_name

        # ensure that the folder exists
        path.mkdir(exist_ok=True, parents=True)

        self._root_path = path

    @property
    def root_storage_path(self) -> Path:
        return self._root_path


if __name__ == "__main__":
    handler = Paths(
        "/home/amiguel/seminar/teste_code_changes/CARMENES", {"testes": "testepaths/hkasdh"}
    )
