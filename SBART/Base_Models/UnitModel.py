from pathlib import Path
from typing import NoReturn

from SBART.Base_Models.BASE import BASE
from SBART.utils import custom_exceptions


class UnitModel(BASE):
    """
    Base unit.
    """

    # The Units will store data in the RV_cube/_content_name folder
    _content_name = "BASE"
    _name = "StorageUnit::"

    def __init__(self, frameIDs, N_orders, needed_folders=None):

        super().__init__(needed_folders=needed_folders)
        self.associated_frameIDs = frameIDs
        self.N_orders = N_orders

    def is_storage_type(self, store_type):
        return store_type in self._name

    def find_index_of(self, frameID):
        return self.associated_frameIDs.index(frameID)

    def generate_root_path(self, storage_path: Path) -> NoReturn:
        if isinstance(storage_path, str):
            storage_path = Path(storage_path)
        storage_path = storage_path / self._content_name
        super().generate_root_path(storage_path)

    @classmethod
    def load_from_disk(cls, root_path: Path):
        storage_path = root_path / cls._content_name
        if not storage_path.exists():
            raise custom_exceptions.NoDataError
