from pathlib import Path
from typing import NoReturn

from SBART.utils.BASE import BASE
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

    @classmethod
    def load_from_disk(cls, root_path: Path):
        if not root_path.exists():
            raise custom_exceptions.NoDataError
