"""Handles the creation of the stellar models
"""

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

from .stellar_templates.median_stellar import MedianStellar
from .stellar_templates.OBS_stellar import OBS_Stellar
from .stellar_templates.sum_stellar import SumStellar


class StellarModel(TemplateFramework):
    """The StellarModel is responsible for the creation of the stellar template for each sub-Instrument, allowing the user to apply
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

    template_map = {"Sum": SumStellar, "OBSERVATION": OBS_Stellar, "Median": MedianStellar}

    _default_params = TemplateFramework._default_params + DefaultValues(
        CREATION_MODE=UserParam("Sum", constraint=ValueFromList(list(template_map.keys()))),
        ALIGNEMENT_RV_SOURCE=UserParam("DRS", constraint=ValueFromList(["DRS", "SBART"])),
        PREVIOUS_SBART_PATH=UserParam("", constraint=ValueFromDtype((str, Path))),
        USE_MERGED_RVS=UserParam(False, constraint=BooleanValue),
    )

    def __init__(self, root_folder_path: UI_PATH, user_configs: Optional[UI_DICT] = None):
        """Instantiation of the object:

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
        """Apply the spectral conditions to decide which observations to use. Then, returns to the model generation as defined in the parent implementation.

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
        -----
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
            pth = Path(self._internal_configs["PREVIOUS_SBART_PATH"])
            if not pth.exists():
                msg = "Previous SBART path does not exist!"
                logger.critical(msg)
                raise custom_exceptions.InvalidConfiguration(msg)

            try:
                # Attempt to go up to 4 folders up to search for Iteration folders for RVs
                for _ in range(4):
                    pth = pth.parent
                    if "Iteration" in pth.stem:
                        iter_number = int(pth.stem.split("_")[-1]) + 1
                    else:
                        continue
                    self.iteration_number = iter_number

                    self.RV_source = Path(self._internal_configs["PREVIOUS_SBART_PATH"]).parent.stem
                    break
                else:
                    msg = "Couldn't find iteration number from user-provided previous sbart path"
                    logger.critical(msg)
                    raise custom_exceptions.InvalidConfiguration(msg)
                logger.info(f"Found data from previous sBART runs, starting Iteration {iter_number}")

                dataClass.load_previous_SBART_results(
                    self._internal_configs["PREVIOUS_SBART_PATH"],
                    use_merged_cube=self._internal_configs["USE_MERGED_RVS"],
                )

            except custom_exceptions.InvalidConfiguration as e:
                self.add_to_status(INTERNAL_ERROR)
                logger.exception("SBART RV loading routine failed. Stopping template creation")
                raise e
        else:
            logger.info("Using CCF RVs as the basis for the creation of the stellar models")

        self.add_relative_path("Stellar", f"Stellar/Iteration_{self.iteration_number}")

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

    def update_interpol_properties(self, new_properties):
        for subInst, temp in self.templates.items():
            temp.set_interpolation_properties(new_properties)

    def get_interpol_modes(self) -> Set[str]:
        return set(temp.interpol_mode for temp in self.templates.values())

    def _compute_template(self, data, subInstrument: str, user_configs: dict):
        chosen_template = self.template_map[self._internal_configs["CREATION_MODE"]]
        key = "ALIGNEMENT_RV_SOURCE"
        if key in user_configs:
            logger.warning(f"Key <{key}> from Stellar Model over-riding the one from the template configs")
        user_configs[key] = self._internal_configs[key]
        stellar_template = chosen_template(subInst=subInstrument, user_configs=user_configs)

        try:
            stellar_template.create_stellar_template(dataClass=data, conditions=self._creation_conditions)
        except NoDataError:
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
