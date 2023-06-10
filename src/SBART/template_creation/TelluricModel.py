import os
from typing import List, Type

from loguru import logger

from SBART.Base_Models.TemplateFramework import TemplateFramework
from SBART.Base_Models.Template_Model import BaseTemplate
from SBART.utils.UserConfigs import DefaultValues, UserParam, ValueFromList
from SBART.utils.custom_exceptions import InvalidConfiguration, TemplateNotExistsError
from SBART.utils.types import UI_DICT, UI_PATH
from .telluric_templates.Telluric_Template import TelluricTemplate
from .telluric_templates.telluric_from_tapas import TapasTelluric
from .telluric_templates.telluric_from_telfit import TelfitTelluric


class TelluricModel(TemplateFramework):
    # noinspection LongLine
    """
    The TelluricModel is responsible for the creation of the telluric template for each sub-Instrument. This object
    supports the following user parameters:

    **User parameters:**

    ================================ ================ ================ ===================== ================
    Parameter name                      Mandatory      Default Value    Valid Values           Comment
    ================================ ================ ================ ===================== ================
    CREATION_MODE                       True            ----            tapas / telfit          Which telluric template to create [1]
    APPLICATION_MODE                    False           removal         removal / correction    How to use the template [2]
    EXTENSION_MODE                      False           lines           lines / window          How to account for BERV changes [3]
    ================================ ================ ================ ===================== ================


    - [1] Currently, SBART supports the usage of the  following templates:

            * "tapas" : :class:`~SBART.template_creation.telluric_templates.telluric_from_telfit.TelfitTelluric`
            * "telfit": :class:`~SBART.template_creation.telluric_templates.telluric_from_tapas.TapasTelluric`

    - [2] Either uses the template to correct or to remove the telluric templates. Possible values:

            * "removal" - the selected source of the transmittance spectra is used to build a binary template
            * "correction" - the template will be used to correct the stellar spectra. Currently, this is not yet supported

    - [3] The way with which we account for BERV variations after creating the "master" binary template. Can take the following values:

            * "lines" - It is shifted to match the BERV of all available observations and the N_obs templates are combined into a final binary template
            * "window" - We search the points in which the binary mask is set to 1. We then Doppler shift those wavelengths by the maximum BERV varation that we suffer during a year (~30 km/s).

    *Note:* Also check the **User parameters** of the parent classes for further customization options of SBART

    """
    model_type = "Telluric"

    template_map = {"Telfit": TelfitTelluric, "Tapas": TapasTelluric}
    _default_params = TemplateFramework._default_params + DefaultValues(
        CREATION_MODE=UserParam(
            None, constraint=ValueFromList(("tapas", "telfit")), mandatory=True
        ),
        APPLICATION_MODE=UserParam("removal", constraint=ValueFromList(("removal", "correction"))),
        EXTENSION_MODE=UserParam("lines", constraint=ValueFromList(("lines", "window"))),
    )

    def __init__(self, usage_mode: str, root_folder_path: UI_PATH, user_configs: UI_DICT):
        """
        Instantiation of the object:

        Parameters
        ----------
        usage_mode : str
            How to telluric templates will be applied to the data. If 'individual' -> template applied to its own subInstrument
            If 'merged', we use a template built from merging the templates from all subInstruments
        root_folder_path: Union [str, pathlib.Path]
            Path to the folder inside which SBART will store its outputs
        user_configs: Optional[Dict[str, Any]]
            Dictionary with the keys and values of the user parameters that have been described above
        """
        super().__init__(mode="", root_folder_path=root_folder_path, user_configs=user_configs)

        logger.info("Starting Telluric Model")

        # can be 'individual' -> template applied to its own subInstrument
        # or  'merged' and we use a template built from merging the templates from all subInstruments
        self._usage_mode = usage_mode

    def request_template(self, subInstrument: str) -> Type[BaseTemplate]:
        """
        Returns the template for a given sub-Instrument.
        Parameters
        ----------
        subInstrument: str
            "name" of the subInstrument

        Returns
        -------
            Requested telluric Template
        """

        logger.debug("Serving {} template to subInstrument {}", self._usage_mode, subInstrument)
        if self._usage_mode == "":
            return self.templates["merged"]
        if self._usage_mode == "individual":
            return self.templates[subInstrument]
        raise InvalidConfiguration()

    def Generate_Model(
        self,
        dataClass,
        telluric_configs: dict,
        force_computation: bool = False,
        store_templates: bool = True,
    ) -> None:
        """
        Generate a telluric model for all subInstruments with data, as defined in the parent implementation. Afterwards,
        allow to combine the telluric model of all sub-Instruments into a single telluric binary model which will
        then be used for all available observations.

        [**Warning**: the combination is yet to be implemented]

        Parameters
        ----------
        dataClass : :class:`~SBART.data_objects.DataClass`
            DataClass with the observations
        telluric_configs : dict
            Dictionary with the parameters needed to control the creation of the telluric template, following the
            specifications of the templates.
        force_computation : bool
            If False, it will attempt to lead the telluric template from disk before trying to compute them. If True,
            always compute telluric template, even if it exists
        """

        super().Generate_Model(
            dataClass=dataClass,
            template_configs=telluric_configs,
            attempt_to_load=not force_computation,
            store_templates=False,
        )

        if self._usage_mode == "merged":
            logger.info("Telluric model merging the templates from all epochs")
            self.merge_templates()

        if store_templates:
            self.store_templates_to_disk()

    def merge_templates(self) -> None:
        """
        Merge the telluric template of all sub-Instruments to create a master telluric binary template

        Raises
        -------
        NotImplementedError
            The method is yet to be implemented
        """
        logger.info("Merging templates to create a global one")
        raise NotImplementedError

    # Internal Usage:

    def _find_templates_from_disk(self, which: str) -> List[str]:

        which = which.capitalize()

        loading_path = self._internalPaths.get_path_to(self.__class__.model_type)
        logger.info("Loading {} template from disk inside directory", self.__class__.model_type)
        logger.info("\t" + loading_path)

        available_templates = []
        if not os.path.exists(loading_path):
            logger.warning("Could not find template to load in {}".format(loading_path))
            raise TemplateNotExistsError()

        for fname in os.listdir(loading_path):
            if (
                which in fname
                and self._internal_configs["EXTENSION_MODE"] in fname
                and fname.endswith("fits")
            ):
                available_templates.append(fname)
        logger.info(
            "Found {} available templates: {} of type {}",
            len(available_templates),
            available_templates,
            which,
        )
        if len(available_templates) == 0:
            logger.critical("Could not find templates to load!")
            raise TemplateNotExistsError()

        return [os.path.join(loading_path, i) for i in available_templates]

    def _compute_template(self, data, subInstrument: str, user_configs: dict) -> TelluricTemplate:

        creation_mode = self._internal_configs["CREATION_MODE"]
        logger.info("Using template of type: {}", creation_mode)

        if creation_mode == "none":
            pass
        elif creation_mode == "telfit":
            chosen_template = TelfitTelluric
        elif creation_mode == "tapas":
            chosen_template = TapasTelluric
        else:
            raise InvalidConfiguration()

        tell_template = chosen_template(
            subInst=subInstrument,
            user_configs=user_configs,
            extension_mode=self._internal_configs["EXTENSION_MODE"],
            application_mode=self._internal_configs["APPLICATION_MODE"],
        )

        tell_template.load_information_from_DataClass(data)
        if self.is_for_removal:
            tell_template.create_telluric_template(dataClass=data)
        else:
            logger.debug("Telluric template in removal mode. Fitting from inside dataClass")

        return tell_template

    @property
    def is_for_removal(self) -> bool:
        """
        Returns
        -------
        bool
            True if the template will be used to remove telluric features
        """
        return self._internal_configs["APPLICATION_MODE"] == "removal"

    @property
    def is_for_correction(self) -> bool:
        """

        Returns
        -------
        bool
            True if the template will be used to correct telluric features
        """
        return self._internal_configs["APPLICATION_MODE"] == "correction"
