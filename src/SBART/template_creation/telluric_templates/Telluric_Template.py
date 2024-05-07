import warnings
from pathlib import Path
from typing import List, NoReturn, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.time import Time
from loguru import logger

from SBART import __version__
from SBART.Base_Models.Template_Model import BaseTemplate
from SBART.ModelParameters import Model
from SBART.utils import custom_exceptions
from SBART.utils.RV_utilities.create_spectral_blocks import build_blocks
from SBART.utils.UserConfigs import (
    BooleanValue,
    DefaultValues,
    UserParam,
    ValueInInterval,
)
from SBART.utils.custom_exceptions import NoDataError
from SBART.utils.shift_spectra import apply_BERV_correction
from SBART.utils.spectral_conditions import KEYWORD_condition, Empty_condition
from SBART.utils.status_codes import DISK_LOADED_DATA, MISSING_DATA, SUCCESS
from SBART.utils.telluric_utilities.compute_overlaps_blocks import find_overlaps
from SBART.utils.units import kilometer_second


class TelluricTemplate(BaseTemplate):
    """
    BaseClass of a telluric template, with all the necessary functionalities.

    Inherits from the BaseTemplate, similarly to the stellar templates. The telluric template does not
    have the same shape as the S2D spectra. As we only need the wavelength regions in which it has
    detected a feature, we can preserve the original shape of the transmittance spectra.

    **User parameters:**

    ================================ ================ ================ ================ ================
    Parameter name                      Mandatory      Default Value    Valid Values    Comment
    ================================ ================ ================ ================ ================
    continuum_percentage_drop           False              1            Inside [0, 100]  [1]
    force_download                      False              False         boolean         [2]
    ================================ ================ ================ ================ ================

    - [1]: Minimum drop (in relation to continuum level) needed to flag a point as a telluric feature
    - [2]: Force the download of all necessary data products, even if not necessary

    *Note:* Also check the **User parameters** of the parent classes for further customization options of SBART

    """

    _name = "Telluric"
    _default_params = BaseTemplate._default_params + DefaultValues(
        continuum_percentage_drop=UserParam(
            1, constraint=ValueInInterval([0, 100], include_edges=True)
        ),
        force_download=UserParam(False, constraint=BooleanValue),
    )
    template_type = "Telluric"
    method_name = "Base"

    def __init__(
        self,
        subInst: str,
        user_configs: Union[None, dict] = None,
        extension_mode: str = "lines",
        application_mode: str = "removal",
        loaded: bool = False,
    ):
        """

        Parameters
        ----------
        extension_mode : str, optional
            How to handle with Earth motion during the year. If 'lines', compute the template in each possible location
            for the available observations and only remove the. If "window", each telluric feature will be extended by a
            region equal to the maximum BERV. By default: 'lines'

        """
        super().__init__(subInst, user_configs, loaded)

        self._associated_BERV = 0

        self._application_mode = application_mode
        self._extension_mode = extension_mode
        self._masked_wavelengths = []
        self._computed_wave_blocks = False

        self.transmittance_wavelengths = None
        self.transmittance_spectra = None
        self._continuum_level = None

        self._fitModel = Model(params_of_model=[])
        self._BERVS = []
        self.MAXBERV = None
        self._reference_frameID = None  # used for telluric removal!

        if self.for_feature_correction:
            # Not the best way of doing this, but ....
            self.add_to_status(SUCCESS)

        self._loaded_dataclass_info = False

        self.template = None

        self._metric_selection_conditions = Empty_condition()

    # Workaround to avoid calling spectra to the binary telluric template (to make things internally consistent)
    @property
    def spectra(self):
        return self.template

    @spectra.setter
    def spectra(self, updated_value):
        self.internal_val = updated_value

    def get_data_from_spectral_order(self, order: int, include_invalid: bool = False):
        uncertainties = self.uncertainties[order] if self.for_feature_correction else None
        return self.wavelengths[order], self.template[order], uncertainties

    def get_data_from_full_spectrum(self):
        uncertainties = self.uncertainties if self.for_feature_correction else None
        return self.wavelengths, self.template, uncertainties

    #####

    def _generate_model_parameters(self, dataClass):
        logger.debug("Generating the parameters of the model applied in {}", self.name)
        self._fitModel.generate_priors(dataClass)

    def load_information_from_DataClass(self, dataClass) -> None:
        """
        Load the necessary information from the dataClass to generate the models. Loads:
            - BERV information (for the telluric removal)
            - Generates the model information (i.e. populates priors and bounds of self._fitModel

        Parameters
        ----------
        dataClass

        Returns
        -------

        """
        logger.debug("Template {} loading data from the dataClass", self.name)

        self.BERVS, self.MAXBERV = self._load_BERV_info(dataClass)

        if not self.is_valid:
            return

        self._generate_model_parameters(dataClass)
        self._loaded_dataclass_info = True

    # Internal Data loading routines
    def _load_BERV_info(self, DataClass):
        """Load the BERV values and the MaxBERV of all observations from the subInstrument associated with this template!

        Parameters
        ----------
        DataClass : [type]
            [description]

        Returns
        -------
        Tuple[List[float], List[float]]
            [description]
        """
        if not self.is_valid:
            return [], np.nan * kilometer_second

        if self._associated_subInst not in DataClass.get_subInstruments_with_valid_frames():
            logger.warning(
                "{} has no valid observations. Not computing telluric template",
                self._associated_subInst,
            )
            self.add_to_status(MISSING_DATA)
            return [], []

        BERVS = DataClass.collect_KW_observations(
            KW="BERV", subInstruments=[self._associated_subInst], include_invalid=False
        )
        max_bervs = DataClass.collect_KW_observations(
            KW="MAX_BERV",
            subInstruments=[self._associated_subInst],
            include_invalid=False,
        )

        # It seems that numpy does not like lists of astropy.units elements
        unitless_max_bervs = [i.value for i in max_bervs]
        return BERVS, max_bervs[np.argmax(unitless_max_bervs)]

    def _search_reference_frame(self, dataclass: "DataClass") -> Union[int, float]:
        """
        Select the frame that will be used to construct the telluric template.
        By default, select the one with the highest relative humidity. If that
        keyword is not loaded, then uses the one with the highest airmass. If
        there are no valid frames in the associated subINstrument, returns
        np.nan

        Parameters
        ----------
        dataclass : DataClass
            [description]

        Returns
        -------
        int
            [description]
        """
        try:
            valid_frame_ids = dataclass.get_frameIDs_from_subInst(self._associated_subInst)
        except NoDataError as exc:
            msg = "{} has no valid observations. Not computing telluric template".format(
                self._associated_subInst
            )

            logger.warning(msg)
            self.add_to_status(MISSING_DATA(msg))
            return np.nan

        # Placing upper limit of temperature at 50ºC
        self._metric_selection_conditions += KEYWORD_condition(
            "ambient_temperature", [[None, 323.15]]
        )

        if self.__class__.method_name.lower() == "telfit":
            # 1 December 2014, because no GDAS profile for telfit
            # ! may blow up if it removes all observations :(
            # add condition so that the reference observation is more than week
            # ago, to guarantee the GDAS profile already exists
            one_week_ago = int(Time.now().jd - 7)
            self._metric_selection_conditions += KEYWORD_condition("BJD", [[2453340, one_week_ago]])

        logger.debug("Using Relative humidity as the selection criterion for reference observation")
        metric_to_select = dataclass.collect_KW_observations(
            KW="relative_humidity",
            subInstruments=[self._associated_subInst],
            include_invalid=False,
            conditions=self._metric_selection_conditions,
        )

        # due to the conditions, some elements may be None
        metric_to_select = [-1 if m is None else m for m in metric_to_select]

        if not np.isfinite(metric_to_select[0]):
            logger.warning(
                "Relative humidity keyword was not loaded. Using airmass to select the reference observation for {}",
                self._associated_subInst,
            )
            metric_to_select = dataclass.collect_KW_observations(
                KW="airmass",
                subInstruments=[self._associated_subInst],
                include_invalid=False,
                conditions=self._metric_selection_conditions,
            )

            # due to the conditions, some elements may be None
            metric_to_select = [-1 if m is None else m for m in metric_to_select]

        chosen_frameID = valid_frame_ids[np.argmax(metric_to_select)]
        self._associated_BERV = dataclass.get_frame_by_ID(chosen_frameID).get_KW_value("BERV")

        logger.info(
            "Telluric Template from {} using {} as the reference",
            self._associated_subInst,
            dataclass.get_frame_by_ID(chosen_frameID),
        )

        return chosen_frameID

    #########################################
    #  Creation of the telluric binary mask #
    #########################################

    def create_telluric_template(self, dataClass, custom_frameID: Optional[int] = None) -> None:
        logger.info(
            "Starting creation of {} template from {}",
            self.__class__.template_type,
            self._associated_subInst,
        )

        if custom_frameID is not None:
            logger.info(
                "Creation of Telluric Template is based on a custom frameID: {}", custom_frameID
            )
            self._reference_frameID = custom_frameID
        else:
            self._reference_frameID = self._search_reference_frame(dataClass)

        if self.for_feature_correction:
            raise custom_exceptions.InvalidConfiguration(
                "No need to generate binary mask for a correction model"
            )

        self._base_checks_for_template_creation()

        if not self._loaded_dataclass_info:
            raise custom_exceptions.InvalidConfiguration(
                f"{ self.name} did not load dataClass information"
            )

        logger.info(
            "Starting telluric template creation, with reference ID = {}",
            self._reference_frameID,
        )

        self._fitModel.disable_full_model()

    ###
    #   transmittance post-processing
    ###
    def _compute_wave_blocks(self) -> None:
        """Go from the binary template to a list of wavelength regions in which the template is non-zero.
        If the template was not computed, add nothing and raise a warning.
        """
        if not self.is_valid:
            msg = "Creating list of wavelength region (with tellurics) from a non-computed Telluric template. No telluric features identified"
            logger.warning(msg)
            raise custom_exceptions.InternalError(msg)

        # TODO:  optimize the list of blocked features! It will have a very bad scaling with N_{obs}
        logger.info("Creating list of blocked features due to tellurics")
        if not self.was_loaded:
            logger.info("Extending telluric features with the mode: <{}>", self._extension_mode)

        indexes = build_blocks(np.where(self.template != 0))
        updated_block = []
        for telluric_block in indexes:
            interval = self.wavelengths[telluric_block]
            # first overlap search, to take advantage of the smaller list size in here (when compared against the "global" one)
            updated_block.extend(
                find_overlaps(self._extend_detections([interval[0], interval[-1]]))
            )

        self._masked_wavelengths = find_overlaps(updated_block)
        self._computed_wave_blocks = True

    def _extend_detections(self, telluric_block: List[list]) -> List[List[float]]:
        """Extend each block of telluric detection based on the self._extension_mode that was selected by the user

        Parameters
        ----------
        telluric_block : List[list]
            Start and end of a wavelength block that has a telluric feature!

        Returns
        -------
        List[List[float]]
            Updated position of the feature
        """
        if self.was_loaded:
            return [telluric_block]

        updated_block = []
        if self._extension_mode == "lines":
            for berv_value in self.BERVS:
                berv = berv_value.to(kilometer_second).value
                lowest_wavelength = apply_BERV_correction(telluric_block[0], BERV=berv)
                highest_wavelength = apply_BERV_correction(telluric_block[1], BERV=berv)
                updated_block.append([lowest_wavelength, highest_wavelength])

        elif self._extension_mode == "window":
            berv = self.MAXBERV.to(kilometer_second).value
            lowest_wavelength = apply_BERV_correction(telluric_block[0], BERV=-berv)
            highest_wavelength = apply_BERV_correction(telluric_block[1], BERV=berv)
            updated_block.append([lowest_wavelength, highest_wavelength])

        return updated_block

    def create_binary_template(self, continuum_level) -> None:
        """Transform the transmittance spectra into a binary template, based on percentage deviations from the continuum level.

        Parameters
        ----------
        continuum_level : np.ndarray()
            Continuum level that is estimated by the children classes
        """

        logger.info("Converting from transmittance spectra to binary mask!")

        if not self.for_feature_removal:
            logger.warning(
                "Telluric Template will not be used to remove spectra. No need to create binary mask"
            )
            return
        # Find a decrease of 1% in relation to the continuum level; Positive
        # gains (against the continuum value) are not considered as tellurics
        percentages = (self.transmittance_spectra - continuum_level) / continuum_level
        telluric_indexes = np.where(
            percentages < -self._internal_configs["continuum_percentage_drop"] / 100
        )

        # We want a binary template
        telluric_mask = np.zeros_like(self.transmittance_spectra, dtype=int)
        telluric_mask[telluric_indexes] = 1

        self.template = telluric_mask

        self._compute_wave_blocks()

    #######################################
    #  Outside access to the properties   #
    #######################################

    @property
    def contaminated_regions(self) -> list:
        if not self.for_feature_removal:
            raise custom_exceptions.InvalidConfiguration(
                "{} is not a template constructed for telluric removal!"
            )

        if not self._computed_wave_blocks:
            logger.debug("No previous computation of wavelength blocks. Doing it now!")
            self._compute_wave_blocks()
        return self._masked_wavelengths

    @property
    def storage_name(self) -> str:
        return f"{self.__class__.method_name}_{self._extension_mode}_{self.__class__.template_type}"

    @property
    def for_feature_removal(self) -> bool:
        return self._application_mode == "removal"

    @property
    def for_feature_correction(self) -> bool:
        return self._application_mode == "correction"

    def store_metrics(self):
        ...

    def trigger_data_storage(self, *args, **kwargs) -> NoReturn:
        try:
            super().trigger_data_storage(*args, **kwargs)
            self.store_metrics()
        except custom_exceptions.FailedStorage:
            return

    #################################
    #           Data storage        #
    #################################

    def store_template(self, clobber: bool) -> NoReturn:
        if self.for_feature_correction:
            logger.info(
                "{} is on correction mode. Not storing conventional data products to disk",
                self.name,
            )
            return

        try:
            super().store_template(clobber)
        except custom_exceptions.FailedStorage:
            return

        header = fits.Header()

        header["TYPE"] = self.__class__.template_type
        header["subInst"] = self._associated_subInst
        header["VERSION"] = __version__
        header["IS_VALID"] = self.is_valid

        for key, config_val in self._internal_configs.items():
            if "path" in key or "user_" in key or isinstance(config_val, (list, tuple)):
                continue

            if "FIT" in key:
                continue

            header["HIERARCH {}".format(key)] = config_val
        hdu = fits.PrimaryHDU(data=[], header=header)

        hdus_cubes = [hdu]

        hdu_wave = fits.ImageHDU(data=self.wavelengths, header=header, name="Wave")
        complete_template = np.zeros(self.template.shape)
        for pair in self._masked_wavelengths:
            indexes = np.where(
                np.logical_and(self.wavelengths >= pair[0], self.wavelengths <= pair[1])
            )
            complete_template[indexes] = 1

        hdu_temp = fits.ImageHDU(data=complete_template, header=header, name="Temp")

        for val in [hdu_wave, hdu_temp]:
            hdus_cubes.append(val)

        hdu_transWave = fits.ImageHDU(
            data=self.transmittance_wavelengths, header=header, name="TRANSMIT_WAVE"
        )
        hdu_transSpec = fits.ImageHDU(
            data=self.transmittance_spectra, header=header, name="TRANSMIT_SPECTRA"
        )
        hdus_cubes.extend([hdu_transWave, hdu_transSpec])

        hdul = fits.HDUList(hdus_cubes)

        filename = f"{self.storage_name}_{self._associated_subInst}.fits"
        logger.debug("Storing template to {}", self._internalPaths.root_storage_path / filename)
        hdul.writeto(self._internalPaths.root_storage_path / filename, overwrite=True)

        metrics_path = self._internalPaths.get_path_to("metrics", as_posix=False)

        fig, axis = plt.subplots()
        axis.plot(self.transmittance_wavelengths, self.transmittance_spectra)
        axis.set_xlabel(r"Wavelength [$\AA$]")
        axis.set_ylabel("Transmittance")
        fig.savefig(metrics_path / f"transmittance_{self._associated_subInst}.png")
        plt.close(fig)

    def load_from_file(self, root_path: Path, loading_path: str) -> None:
        """
        TODO: save and load the actual flag to disk!
        Parameters
        ----------
        root_path
        loading_path

        Returns
        -------

        """
        super().load_from_file(root_path, loading_path)

        with fits.open(loading_path) as hdulist:
            if hdulist[1].header.get("VERSION", "") != __version__:
                logger.warning(
                    "Loaded template was not created under the current SBART version. Possible problems may arise"
                )
            self._associated_subInst = hdulist["Wave"].header["subInst"]

            waves = hdulist["Wave"].data
            template = hdulist["Temp"].data

            self.transmittance_wavelengths = hdulist["TRANSMIT_WAVE"].data
            self.transmittance_spectra = hdulist["TRANSMIT_SPECTRA"].data

            self.template = template
            self.wavelengths = waves

        self.add_to_status(DISK_LOADED_DATA(f"Loaded data from {loading_path}"))

    def _finish_template_creation(self):
        self.create_binary_template(self._continuum_level)

        super()._finish_template_creation()
