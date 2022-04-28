import os
from typing import NoReturn, Union

from loguru import logger

from SBART.Components import Spectrum
from SBART.utils import custom_exceptions
from SBART.utils.UserConfigs import DefaultValues
from SBART.utils.status_codes import (
    CREATING_MODEL,
    FAILED_MODEL_CREATION,
    MISSING_DATA,
    SUCCESS,
)


class BaseTemplate(Spectrum):
    """
    Base Class for all Templates (both telluric and stellar). Implements validity checks, base routines for data storage and "outside"
    interface to request data

    **User parameters:**

        This object doesn't introduce unique user parameters.

    *Note:* Check the **User parameters** of the parent classes for further customization options of this class

    """

    _object_type = "Template"

    method_name = "Base"
    template_type = "Base"

    _default_params = Spectrum._default_params + DefaultValues()

    def __init__(self, subInst: str, user_configs: Union[None, dict], loaded=False):
        """
        Parameters
        ----------
        subInst
            sub-Instrument for which the template is going to be created
        user_configs
            Dictionary with the keys and values of the user parameters that have been described above
        loaded
            True if the template was loaded from disk.
        """

        # The .name property needs the subInstrument to be defined ASAP
        self._associated_subInst = subInst

        super().__init__(
            user_configs=user_configs,
            needed_folders={
                "metrics": "metrics",
                "fit_products": "data_products/fit_products",
                "RunTimeProds": "data_products/RunTimeProducts",
            },
            start_with_valid_status=True,
        )

        self._cached_info = None
        self._created_template = False
        self._proceed_with_creation = True

        self._loaded = loaded
        self.add_to_status(CREATING_MODEL)

    def trigger_data_storage(self, clobber: bool) -> NoReturn:
        """
        Check for validity of the template and, afterwards, trigger the data storage routine

        Parameters
        ----------
        clobber:
            If True, delete old disk files before attempting to store the new ones.

        Raises
        ------
        FailedStorage
            If the template was either loaded or not created
        """
        super().trigger_data_storage(clobber)

        if self.was_loaded:
            logger.debug(f"Loaded {self.template_type} template will not be saved to disk")
            raise custom_exceptions.FailedStorage

        if not self.is_valid:
            logger.info("The template was not created. Storing nothing to disk")
            raise custom_exceptions.FailedStorage

        self.store_template(clobber=clobber)

    def _base_checks_for_template_creation(self):
        if not self.is_valid:
            logger.warning("Template will not be created. Check previous error messages")
            raise custom_exceptions.StopComputationError

        if self.was_loaded:
            logger.warning("Loaded Template has nothing to compute.")
            raise custom_exceptions.StopComputationError

    def store_template(self, clobber: bool) -> None:
        """
        Handle deletion of old disk files + apply checks to see if we want to store data to disk

        Parameters
        ----------
        clobber

        Returns
        -------

        Raises
        -------
        FailedStorage
            If we attempt to remove the previous version of the template and it was not found on disk.
            TODO: is this a good idea? I would bet that it isn't.....
        """

        logger.info("Storing {} to disk", self.name)

        if clobber:
            logger.warning(
                "Deleting previous outputs stored in the folder <{}>",
                self._internalPaths.root_storage_path,
            )
            filename = f"{self.storage_name}_{self._associated_subInst}.fits"
            template_path = self._internalPaths.root_storage_path / filename

            try:
                os.remove(template_path)
            except FileNotFoundError:
                logger.warning("Previous template not found under the path <{}>", template_path)
                raise custom_exceptions.FailedStorage
        else:
            logger.warning("Disabled removal of old disk files!")

        if not self.is_valid:
            raise custom_exceptions.FailedStorage("Template is not valid. Not storing data to disk")

    def load_from_file(self, root_path, loading_path: str) -> None:
        """
        Interface to load a template from disk

        TODO: we don't really need the loading path... the root_path is enough for what we want to do...

        Parameters
        ----------
        root_path
            Root path for the template
        loading_path
            Path to the actual template

        Returns
        -------

        Raises
        ------
        NoDataError
            If we attempt to load from a path that does not exist
        """
        logger.info("Loading {} template from disk file:", self.__class__.template_type)
        logger.info("\t" + os.path.basename(loading_path))

        super().load_from_file(root_path, loading_path)
        if not os.path.exists(loading_path):
            logger.warning(
                "There is no {} template for {}} data",
                self.__class__.template_type,
                self._associated_subInst,
            )
            self.add_to_status(MISSING_DATA("No template stored on disk"))
            raise custom_exceptions.NoDataError

        self._loaded = True

    def _finish_template_creation(self):
        self.add_to_status(SUCCESS("Created {} template".format(self.template_type)))

    def is_type(self, type_to_test: str) -> bool:
        """
        Parameters
        ----------
        type_to_test
            Check if the template is of a given type (i.e. Stellar or Telluric)

        Returns
        -------
        result: bool
            True if the types match.
        """
        return self.__class__.template_type == type_to_test

    def mark_as_invalid(self) -> NoReturn:
        """
        Change the status of the template into a Failed State. This will make all of the future validity tests fail!
        Returns
        -------

        """
        self.add_to_status(FAILED_MODEL_CREATION)

    @property
    def was_loaded(self) -> bool:
        """

        Returns
        -------
        result: bool
            True if the template was loaded
        """
        return self._loaded

    @property
    def name(self) -> str:
        return f"{self.__class__.method_name}-{self.__class__.template_type} from {self._associated_subInst}"

    @property
    def storage_name(self) -> str:
        return f"{self.__class__.method_name}-{self.__class__.template_type}"

    def __repr__(self):
        return self.name
