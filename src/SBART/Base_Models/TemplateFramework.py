import os
from typing import List, Optional, Type

from astropy.io import fits
from loguru import logger

from SBART.utils.BASE import BASE
from SBART.Base_Models.Template_Model import BaseTemplate
from SBART.utils import custom_exceptions
from SBART.utils.types import UI_DICT, UI_PATH
from SBART.utils.UserConfigs import DefaultValues


class TemplateFramework(BASE):
    """
    Base Class for the Stellar and Telluric Models. This class is responsible for:

    * Trigger the creation of individual templates for each sub-Instrument
    * Trigerring the template's disk operations (saving and loading)
    * Delivering the templates that correspond to a given subInstrument

    **User parameters:**

        This class introduces no unique parameters

    """

    # as string to avoid circular dependency: https://www.python.org/dev/peps/pep-0484/#forward-references
    _object_type = "BASE"
    _name = "TemplateFramework"

    model_type = "Base"
    template_map = {}

    _default_params = BASE._default_params + DefaultValues()

    def __init__(
        self, mode: str, root_folder_path: UI_PATH, user_configs: Optional[UI_DICT] = None
    ):
        """
        Parameters
        ----------
        mode
            To be deprecated
        root_folder_path
            Root path to store the data products of sBART
        user_configs
            Configurations for this object, following the provided specifications
        """
        logger.info("Starting {} Model", self.__class__.model_type)
        super().__init__(
            needed_folders={"Stellar": "Stellar", "Telluric": "Telluric"}, user_configs=user_configs
        )
        self._internalPaths.add_root_path(root_folder_path, "templates")

        self.templates = {}

        self._valid_params = {}

    def request_data(self, subInstrument: str) -> Type[BaseTemplate]:
        """Return the template built for a given subInstrument

        Parameters
        ----------
        subInstrument
            subInstrument name

        Returns
        -------
        template
            The desired template

        Raises
        ---------
        BadTemplateError
            If the template was not created
        InvalidConfiguration
            If the subInstrument doesn't exist
        """
        try:
            if not self.templates[subInstrument].is_valid:
                msg = "Template was not created"
                logger.critical(msg)
                raise custom_exceptions.BadTemplateError(msg)
        except KeyError as exc:
            msg = "There is no {} template from  {}".format(self.name, subInstrument)
            logger.critical(msg)
            raise custom_exceptions.InvalidConfiguration(msg)

        return self.templates[subInstrument]

    def Generate_Model(
        self,
        dataClass,
        template_configs: dict,
        attempt_to_load: bool = False,
        store_templates: bool = True,
    ) -> None:
        """Generate a model for all subInstruments with data

        Parameters
        ----------
        dataClass : :class:`~SBART.data_objects.DataClass`
            DataClass with the observations
        template_configs:
            Dictionary that passes the user-configurations for the templates that will be created (following the specifications of the
            chosen template). Currently all sub-Instruments have to share the same configuration
        attempt_to_load : bool
            Before computing the templates, attempt to load them from the disk location in which it will be stored
        store_templates:
            If True, trigger the data storage routines after creating the templates

        """

        logger.debug("Starting the creation of {} models!", self.__class__.model_type)

        if attempt_to_load:
            logger.info("Attempting to load previous Templates from disk before creating them")
            try:
                self.load_templates_from_disk()
            except custom_exceptions.TemplateNotExistsError:
                logger.info("No templates to load from disk. Creating all from scratch")
        else:
            logger.info("Creating all templates from scratch")

        for subInst in dataClass.get_available_subInstruments():
            if subInst not in self.templates:
                self.templates[subInst] = None

        for subInst, prev_template in self.templates.items():
            if prev_template is not None:
                logger.debug(f"Template from {subInst} has already been loaded.")
                continue

            self.templates[subInst] = self._compute_template(
                data=dataClass, subInstrument=subInst, user_configs=template_configs
            )

            self.templates[subInst].generate_root_path(
                self._internalPaths.get_path_to(self.__class__.model_type, as_posix=False)
            )

        if store_templates:
            self.store_templates_to_disk()

    def load_templates_from_disk(self):
        """
        Load templates from disk.

        Parameters
        ----------
        """
        template_path = self._internalPaths.get_path_to(self.__class__.model_type)

        which = self._internal_configs["CREATION_MODE"]
        logger.debug("Searching in : {} for {}", template_path, which)
        available_templates = self._find_templates_from_disk(which=which)

        for temp_path in available_templates:
            temp_disk_name = os.path.basename(temp_path)

            temp_name = temp_disk_name.split("_")[0]
            temp_subInst = temp_disk_name.split("_")[-1].split(".fits")[0]

            template_header = fits.getheader(temp_path)
            config_dict = {}
            for key in self.__class__.template_map[temp_name].control_parameters():
                if "path" in key or "user_" in key:
                    continue
                # ! just ignore the FIT keys?
                if "FIT" in key:
                    continue
                config_dict[key] = template_header[f"HIERARCH {key}"]

            if self.is_type("Telluric"):
                config_dict["download_path"] = ""

            loaded_temp = self.__class__.template_map[temp_name](
                temp_subInst, loaded=True, user_configs=config_dict
            )

            try:
                loaded_temp.load_from_file(root_path=template_path, loading_path=temp_path)
            except custom_exceptions.NoDataError:
                pass
            self.templates[temp_subInst] = loaded_temp

    def _find_templates_from_disk(self, which: str) -> List[str]:
        """
        Search the storage disk location to find any templates that might have been stored in there

        Parameters
        ----------
        which : str
            type of template that we want to load (.e. Tapas, Telfit, Sum)

        Returns
        -------
        paths
            List of the paths found on disk [TODO: change from strings to pathlib.Path]

        Raises
        ------
        TemplateNotExistsError
            If it is not possible to find any stored template on the disk path
        """

        which = which.capitalize()
        loading_path = self._internalPaths.get_path_to(self.__class__.model_type, as_posix=True)
        logger.info(
            "Loading {} template of type {} from disk inside directory",
            self.__class__.model_type,
            which,
        )
        logger.info("\t" + loading_path)
        available_templates = [
            i for i in os.listdir(loading_path) if which in i and i.endswith("fits")
        ]
        logger.info(
            "Found {} available templates: {} of type {}",
            len(available_templates),
            available_templates,
            which,
        )

        if len(available_templates) == 0:
            logger.warning("Could not find template to load in {}".format(loading_path))
            raise custom_exceptions.TemplateNotExistsError()
        return [os.path.join(loading_path, i) for i in available_templates]

    def store_templates_to_disk(self, clobber: bool = False) -> None:
        """
        Trigger the data storage routine of all templates stored inside the Model

        Parameters
        ----------
        clobber : bool
            Whether to delete and re-write over previous outputs
        """
        storage_path = self._internalPaths.get_path_to(self.__class__.model_type, as_posix=True)
        logger.info(
            "Storing templates from <{}> under the directory",
            self.__class__.model_type,
        )
        logger.info("\t" + storage_path)

        for template in self.templates.values():
            template.trigger_data_storage(clobber=clobber)

    def is_type(self, to_check: str) -> bool:
        """
        Check if the Model is of a given type (i.e. Stellar or Telluric)

        Parameters
        ----------
        to_check
            Type to check the model against

        Returns
        -------
        bool
            Output of the comparison

        """
        return self.__class__.model_type == to_check

    def __repr__(self):
        return f"Model handling: {self.templates}"
