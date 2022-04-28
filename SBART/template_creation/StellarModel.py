from pathlib import Path
from typing import Optional, Set

from loguru import logger

from SBART.Base_Models.TemplateFramework import TemplateFramework
from SBART.utils import custom_exceptions
from SBART.utils.custom_exceptions import BadTemplateError, NoDataError
from SBART.utils.spectral_conditions import ConditionModel, Empty_condition
from SBART.utils.status_codes import INTERNAL_ERROR
from SBART.utils.types import UI_DICT, UI_PATH
from SBART.utils.UserConfigs import (
    BooleanValue,
    DefaultValues,
    UserParam,
    ValueFromDtype,
    ValueFromList,
)

from .stellar_templates.OBS_stellar import OBS_Stellar
from .stellar_templates.sum_stellar import SumStellar


class StellarModel(TemplateFramework):
    """
    The StellarModel is responsible for the creation of the stellar template for each sub-Instrument, allowing the user to apply
    :py:mod:`~SBART.utils.spectral_conditions` to select the observations that will be in use for this process.

    This object supports the following user parameters:

    **User parameters:**

    ================================ ================ ================ ==================== ================
    Parameter name                      Mandatory      Default Value    Valid Values        Comment
    ================================ ================ ================ ==================== ================
    CREATION_MODE                       False              Sum         Sum, OBSERVATION      Type of stellar template to create  [1]
    ================================ ================ ================ ==================== ================

    - [1] Currently, SBART supports the usage of the  following stellar templates:

            * Sum : :class:`~SBART.template_creation.stellar_templates.sum_stellar.SumStellar`
            * OBS_Stellar: :class:`~SBART.template_creation.stellar_templates.OBS_stellar.OBS_Stellar`

    **Note:** Also check the **User parameters** of the parent classes for further customization options of SBART
    **Note:** All disk- and user-facing interactions are handled by the parent class

    """

    _object_type = "SpectralModel"
    _name = "Stellar"

    model_type = "Stellar"

    template_map = {"Sum": SumStellar, "OBSERVATION": OBS_Stellar}

    _default_params = TemplateFramework._default_params + DefaultValues(
        CREATION_MODE=UserParam("Sum", constraint=ValueFromList(list(template_map.keys()))),
        ALIGNEMENT_RV_SOURCE=UserParam("DRS", constraint=ValueFromList(["DRS", "SBART"])),
        PREVIOUS_SBART_PATH=UserParam("", constraint=ValueFromDtype((str, Path))),
        USE_MERGED_RVS=UserParam(False, constraint=BooleanValue),
    )

    def __init__(self, root_folder_path: UI_PATH, user_configs: Optional[UI_DICT] = None):
        """
        Instantiation of the object:

        Parameters
        ----------
        root_folder_path: Union [str, pathlib.Path]
            Path to the folder inside which SBART will store its outputs
        user_configs: Optional[Dict[str, Any]]
            Dictionary with the keys and values of the user parameters that have been described above
        """
        super().__init__(mode="", root_folder_path=root_folder_path, user_configs=user_configs)

        self._creation_conditions = Empty_condition()
        self.iteration_number = 0
        self.RV_source = None

    def Generate_Model(
        self,
        dataClass,
        template_configs: Optional[UI_DICT] = None,
        conditions: Optional[ConditionModel] = None,
        force_computation: bool = False,
        store_templates=True,
    ) -> None:
        """
        Apply the spectral conditions to decide which observations to use. Then, returns to the model generation as defined in the parent implementation.

        Parameters
        ----------
        dataClass : [type]
            DataClass with the observations
        template_configs : dict
            Dictionary with the parameters needed to control the creation of the template. Each one has its own
            set of requirements
        conditions: None, Condition
            Either None (to not select the OBS that will be used) or :py:mod:`~SBART.utils.spectral_conditions` to restrict
            the observations in use.
        force_computation: bool
            If True, recompute the stellar templates, even if they exist on disk. By default False
        store_templates: bool
            If True [default], store the templates to disk

        Notes
        -------

        * The conditions that are passed to the StellarModel are **only** used for the creation of the stellar template. This will **not** reject observations from the RV extraction
        """

        if conditions is not None:
            logger.info("Applying conditions to creation of stellar template")
            self._creation_conditions = conditions

        self.RV_source = self._internal_configs["ALIGNEMENT_RV_SOURCE"]

        if self._internal_configs["ALIGNEMENT_RV_SOURCE"] == "SBART":
            logger.info(
                "{} using {} RVs as the source for stellar template creation",
                self.name,
                self._internal_configs["ALIGNEMENT_RV_SOURCE"],
            )

            try:
                try:
                    self.iteration_number = int(
                        self._internal_configs["PREVIOUS_SBART_PATH"].split("Iteration_")[1]
                    )
                    self.RV_source = self._internal_configs["PREVIOUS_SBART_PATH"].parent.stem
                except IndexError:
                    logger.critical(
                        "Provided SBART path does not follow current conventions of having a folder for each "
                        "iteration"
                    )

                dataClass.load_previous_SBART_results(
                    self._internal_configs["PREVIOUS_SBART_PATH"],
                    use_merged_cube=self._internal_configs["USE_MERGED_RVS"],
                )

            except custom_exceptions.InvalidConfiguration:
                self.add_to_status(INTERNAL_ERROR)
                logger.critical("SBART RV loading routine failed. Stopping template creation")
                return

            self.add_relative_path("Stellar", "Stellar/Iteration_{}".format(self.iteration_number))

        super().Generate_Model(
            dataClass=dataClass,
            template_configs=template_configs,
            attempt_to_load=not force_computation,
            store_templates=False,
        )

        for subInst, temp in self.templates.items():
            temp.update_RV_source_info(
                iteration_number=self.iteration_number,
                RV_source=self.RV_source,
                merged_source=self._internal_configs["USE_MERGED_RVS"],
            )

        if store_templates:
            self.store_templates_to_disk()

    def _compute_template(self, data, subInstrument: str, user_configs: dict):

        chosen_template = self.template_map[self._internal_configs["CREATION_MODE"]]
        stellar_template = chosen_template(subInst=subInstrument, user_configs=user_configs)

        try:
            stellar_template.create_stellar_template(
                dataClass=data, conditions=self._creation_conditions
            )
        except NoDataError as exc:
            logger.info(
                "{} has no available data. The template will be created as an array of zeros",
                subInstrument,
            )

        return stellar_template

    def get_orders_to_skip(self, subInst: str) -> Set[int]:

        if subInst == "all":
            bad_orders = set()
            for temp in self.templates.values():
                if not temp.is_valid:
                    logger.critical("Invalid template <{}> does not have orders to skip", temp)
                    continue
                bad_orders.union(temp.bad_orders)
        else:
            if not self.templates[subInst].is_valid:
                raise BadTemplateError
            bad_orders = self.templates[subInst].bad_orders

        return bad_orders

    def load_templates_from_disk(self) -> None:
        """Currently we only have one type of stellar template -> no need for the user to specify it


        Parameters
        ----------
        path : str
            [description]
        """
        super().load_templates_from_disk()

    @property
    def RV_keyword(self) -> str:
        if self._internal_configs["ALIGNEMENT_RV_SOURCE"] == "SBART":
            RV_KW_start = "previous_SBART_RV"
        else:
            RV_KW_start = "DRS_RV"

        return RV_KW_start
