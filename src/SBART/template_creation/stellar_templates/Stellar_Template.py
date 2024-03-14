import ujson as json
from pathlib import Path
from typing import NoReturn, Union, Optional

import numpy as np
from astropy.io import fits
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from SBART.Base_Models.Frame import Frame
from SBART import __version__
from SBART.Base_Models.Template_Model import BaseTemplate
from SBART.Components import Spectral_Modelling
from SBART.Masks import Mask
from SBART.utils import build_filename, custom_exceptions
from SBART.utils.UserConfigs import (
    BooleanValue,
    DefaultValues,
    IntegerValue,
    UserParam,
    Positive_Value_Constraint,
)
from SBART.utils.concurrent_tools.create_shared_arr import create_shared_array
from SBART.utils.custom_exceptions import (
    NoDataError,
)
from SBART.utils.status_codes import HIGH_CONTAMINATION, MISSING_DATA, OrderStatus
from SBART.utils.shift_spectra import apply_RVshift, remove_RVshift
from SBART.utils.units import convert_data, kilometer_second


class StellarTemplate(BaseTemplate, Spectral_Modelling):
    """
    This object is the parent class of all StellarTemplates, implementing a common set of user configurations, data storage
    operations, among others.

    **User parameters:**

    ========================= ================ ================ ================================= ================
    Parameter name               Mandatory      Default Value               Valid Values            Comment
    ========================= ================ ================ ================================= ================
    NUMBER_WORKERS                 False           1            Integer >= 0                       [2]
    MEMORY_SAVE_MODE               False           False           boolean                         [3]
    MINIMUM_NUMBER_OBS             False           3            Integer >= 0                       [4]
    ========================= ================ ================ ================================= ================

    [1] - How to propagate the spectral uncertainties
    [2] - Number of jobs at once
    [3] - Save RAM by clearing the frame's S2D arrays from memory after using them
    [4] - Minimum number of  **valid** observations needed to proceed with template creation

    .. note::
       This class also uses the User parameters defined by the :class:`~SBART.Components.Modelling.Spectral_Modelling`
    class

    .. note::
        Also check the **User parameters** of the parent classes for further customization options of SBART

    """

    # Note: the root_path of the stellarTemplate is the templates/Stellar folder
    _name = "Stellar"

    _default_params = BaseTemplate._default_params + DefaultValues(
        NUMBER_WORKERS=UserParam(1, IntegerValue + Positive_Value_Constraint),
        MEMORY_SAVE_MODE=UserParam(
            False, constraint=BooleanValue
        ),  # if True, close the S2D files after using them!
        MINIMUM_NUMBER_OBS=UserParam(
            3, constraint=IntegerValue
        ),  # minimum number of OBS to create stellar template
    )

    template_type = "Stellar"
    method_name = "Base"

    def __init__(self, subInst: str, user_configs: Union[None, dict] = None, loaded: bool = False):
        super().__init__(subInst, user_configs, loaded)

        self.rejection_array = None

        self.frameIDs_to_use = []

        self._in_shared_mem = False
        self.shm = {}
        self.package_pool = None
        self.output_pool = None
        self.target_name = None
        self._reference_frameID = None
        self._reference_filepath = None
        self.used_fpaths = []
        self._conditions = None

        self._rejection_flags_map = {}

        # Count the current iteration number; Used when improving the stellar template
        self._iter_number: int = 0
        self._RV_source = None
        self._merged_source = None

    #################################
    #           Template creation   #
    #################################

    def create_stellar_template(self, dataClass, conditions) -> None:
        """
        Trigger the start of the creatoin of the stellar template

        Parameters
        ----------
        dataClass : :class:`~SBART.data_objects.DataClass.DataClass`
            Data
        conditions: :py:mod:`~SBART.utils.spectral_conditions`
            Apply conditions over the frames that will be used to create the stellar template

        Raises
        ------
        StopComputationError
            If the computations is triggered by some event

        """
        logger.info(
            "Starting creation of {} template from {}",
            self.__class__.template_type,
            self._associated_subInst,
        )

        array_size = dataClass.get_instrument_information()["array_size"]
        self.array_size = array_size
        self._OrderStatus = OrderStatus(array_size[0])

        try:
            self.frameIDs_to_use = dataClass.get_frameIDs_from_subInst(self._associated_subInst)
        except NoDataError as exc:
            logger.critical(
                "{} has no valid observations. Not computing {} template",
                self._associated_subInst,
                self.__class__.template_type,
            )
            self.add_to_status(
                MISSING_DATA("No valid observations from {}".format(self._associated_subInst))
            )

        self._base_checks_for_template_creation()

        self._conditions = conditions

        if conditions is None:
            logger.info(
                "No conditions for the creation of the stellar template. Using all available frames from {}",
                self._associated_subInst,
            )
            IDS_to_use = self.frameIDs_to_use
        else:
            logger.info(
                "Evaluating spectral conditions to select the valid observations from {}",
                self._associated_subInst,
            )
            IDS_to_use = []
            for frameID in self.frameIDs_to_use:
                keep, flags = conditions.evaluate(dataClass.get_frame_by_ID(frameID))
                if keep:
                    IDS_to_use.append(frameID)
                else:
                    logger.info(
                        "FrameID {} rejected from stellar template due to {}",
                        frameID,
                        flags,
                    )
                    self._rejection_flags_map[
                        dataClass.get_filename_from_frameID(frameID, full_path=True)
                    ] = flags

            if len(IDS_to_use) == 0:
                msg = (
                    "subInstrument {} has no available observations after applying the spectral conditions. Stellar "
                    "template creation will halt".format(self._associated_subInst)
                )

                logger.warning(msg)
                self.add_to_status(MISSING_DATA(msg))

        self._base_checks_for_template_creation()

        self.frameIDs_to_use = IDS_to_use

        if len(self.frameIDs_to_use) < self._internal_configs["MINIMUM_NUMBER_OBS"]:
            logger.critical(
                "Construction of stellar template from {} using less observations ({}) than the limit ({})",
                self._associated_subInst,
                len(self.frameIDs_to_use),
                self._internal_configs["MINIMUM_NUMBER_OBS"],
            )
            self.add_to_status(
                MISSING_DATA(
                    "Can't create stellar template with less than {} observations".format(
                        self._internal_configs["MINIMUM_NUMBER_OBS"]
                    )
                )
            )

        self._base_checks_for_template_creation()

        # TODO: ensure that they all observations are consistent!
        first_frame = dataClass.get_frame_by_ID(self.frameIDs_to_use[0])
        self.is_blaze_corrected = first_frame.check_if_data_correction_enabled("is_blaze_corrected")
        self.was_telluric_corrected = first_frame.check_if_data_correction_enabled(
            "was_telluric_corrected"
        )
        self.is_BERV_corrected = first_frame.check_if_data_correction_enabled("is_BERV_corrected")
        self.flux_atmos_balance_corrected = first_frame.check_if_data_correction_enabled(
            "flux_atmos_balance_corrected"
        )
        self.flux_dispersion_balance_corrected = first_frame.check_if_data_correction_enabled(
            "flux_dispersion_balance_corrected"
        )

    def add_new_frame_to_template(self, frame: Frame):
        """
        Allow to inject a new observation into a pre-existing model. This base function checks for
        a match on the different flux corrections and ensures that the loaded Flag is set to False,
        so that it is possible to update the disk products afterwards.

        Parameters
        ----------
        frame: Frame
            A new frame to inject into the stellar template

        Returns
        -------

        Raises
        -------
        custom_exceptions.InvalidConfiguration:
            If the flux corrections of the Frame do not match those from the stellar template

        """

        logger.info("Adding new frame to pre-existing stellar template. Updating model!")
        self._loaded = False

        keep = True
        for name, val1, val2 in [
            (
                "Blaze",
                self.is_blaze_corrected,
                frame.check_if_data_correction_enabled("is_blaze_corrected"),
            ),
            (
                "Telluric",
                self.was_telluric_corrected,
                frame.check_if_data_correction_enabled("was_telluric_corrected"),
            ),
            (
                "BERV_corrected",
                self.is_BERV_corrected,
                frame.check_if_data_correction_enabled("is_BERV_corrected"),
            ),
            (
                "Flux atmos balance",
                self.flux_atmos_balance_corrected,
                frame.check_if_data_correction_enabled("flux_atmos_balance_corrected"),
            ),
            (
                "flux_dispersion_balance_corrected",
                self.flux_dispersion_balance_corrected,
                frame.check_if_data_correction_enabled("flux_dispersion_balance_corrected"),
            ),
            ("sub-Instrument", self.sub_instrument, frame.sub_instrument),
        ]:
            if val1 != val2:
                keep = False
                logger.warning(
                    f"Template-frame corrections are different: {name} - template: {val1} - Frame: {val2}"
                )

        if not keep:
            msg = f"New frame does not match the corrections from the stellar template"
            logger.critical(msg)
            raise custom_exceptions.InvalidConfiguration(msg)

    def evaluate_bad_orders(self) -> None:
        logger.info("Computing orders with too many points masked")
        entire_mask = self.spectral_mask.get_custom_mask()
        N_pixels_per_order = self.pixels_per_order
        order_cutoff = N_pixels_per_order - 100

        for order in range(self.N_orders):
            order_mask = entire_mask[order]
            bad_pixels = np.sum(order_mask)
            if bad_pixels > order_cutoff:
                msg = "Stellar template rejecting order {} due to having more than {}/{} pixels masked".format(
                    order, order_cutoff, N_pixels_per_order
                )
                logger.warning(msg)

                self._OrderStatus.add_flag_to_order(order, HIGH_CONTAMINATION(msg))

    #################################
    #           Data access        #
    #################################

    def check_if_used_frameID(self, frameID: int) -> bool:
        """
        Parameters
        ----------
        frameID:
            ID of a given observations

        Returns
        -------
        used: bool
            True if the frame was used for the construction of the template

        """
        return frameID in self.frameIDs_to_use

    #################################
    #           Data storage        #
    #################################
    def trigger_data_storage(self, clobber) -> NoReturn:
        try:
            super().trigger_data_storage(clobber)
            self._store_json_information()
            self.store_metrics()
        except custom_exceptions.FailedStorage:
            return

    def _store_json_information(self):
        logger.info("Storing template flags to disk")
        storage_path = self._internalPaths.get_path_to("RunTimeProds", as_posix=False)

        detailedFlags_storage = build_filename(
            storage_path,
            f"DetailedFlags_{self._associated_subInst}",
            fmt="json",
        )
        self._OrderStatus.store_as_json(detailedFlags_storage)

        miscInfo = build_filename(
            storage_path,
            f"miscInfo_{self._associated_subInst}",
            fmt="json",
        )

        # Note: when adding things here we must use as a key the exact name of the parameter...
        # Note2: carefull with the datatypes that are stored in here...
        storage_information = {
            "frameIDs_to_use": self.frameIDs_to_use,
            "used_fpaths": [i.as_posix() for i in self.used_fpaths],
            "is_BERV_corrected": self.is_BERV_corrected,
            "is_blaze_corrected": self.is_blaze_corrected,
            "was_telluric_corrected": self.was_telluric_corrected,
            "flux_dispersion_balance_corrected": self.flux_dispersion_balance_corrected,
            "flux_atmos_balance_corrected": self.flux_atmos_balance_corrected,
            "_reference_frameID": self._reference_frameID,
            "_reference_filepath": self._reference_filepath,
        }

        with open(miscInfo, mode="w") as file:
            json.dump(storage_information, file, indent=4)

    def store_template(self, clobber: bool) -> None:
        # TODO: store the frameIDS that were used!
        super().store_template(clobber)

        header = fits.Header()

        header["TYPE"] = self.__class__.template_type
        header["subInst"] = self._associated_subInst
        header["VERSION"] = __version__

        for key, config_val in self._internal_configs.items():
            header["HIERARCH {}".format(key)] = config_val
        hdu = fits.PrimaryHDU(data=[], header=header)

        hdus_cubes = [hdu]

        hdu_wave = fits.ImageHDU(data=self.wavelengths, header=header, name="WAVE")
        hdu_temp = fits.ImageHDU(data=self.spectra, header=header, name="TEMP")

        mask = self.spectral_mask.get_custom_mask().astype(int)

        hdu_mask = fits.ImageHDU(data=mask, header=header, name="MASK")
        hdu_uncerts = fits.ImageHDU(data=self.uncertainties, header=header, name="UNCERTAINTIES")

        for val in [hdu_wave, hdu_temp, hdu_mask, hdu_uncerts]:
            hdus_cubes.append(val)

        hdul = fits.HDUList(hdus_cubes)

        filename = f"{self.storage_name}_{self._associated_subInst}.fits"
        hdul.writeto(self._internalPaths.root_storage_path / filename, overwrite=True)

        filename = f"{self.storage_name}_{self._associated_subInst}_inputs.txt"

        logger.debug("Storing used filepaths to disk")
        with open(self._internalPaths.root_storage_path / filename, mode="w") as to_write:
            if self._conditions is not None:
                self._conditions.write_to_disk(to_write)

            to_write.write(self._internal_configs.text_pretty_description(indent_level=0))

            to_write.write("\n\nRejected files (N = {})".format(len(self._rejection_flags_map)))
            for filepath, flags in self._rejection_flags_map.items():
                to_write.write("\n\t{}:".format(filepath))
                for flag in flags:
                    to_write.write(f"\n\t\t{flag}")

            to_write.write("\n\nEpochs in use (Total = {}): \n".format(len(self.frameIDs_to_use)))
            for path in self.used_fpaths:
                to_write.write(f"\n{path}")

    def store_metrics(self):
        metrics_path = self._internalPaths.get_path_to("metrics", as_posix=False)
        mask = self.spectral_mask.get_custom_mask().astype(int)

        N_order, N_pixel = self.spectra.shape
        fig, ax = plt.subplots(1, 1)
        custom_cmap = lambda x: "green" if x < 0.2 else "orange" if x < 0.6 else "red"
        for order in range(N_order):
            order_mask = mask[order]
            value = np.sum(order_mask) / N_pixel
            ax.scatter(order, value, color=custom_cmap(value))
        ax.set_xlabel("Order number")
        ax.set_ylabel("% pixels rejected")
        fig.savefig(metrics_path / f"Template_rejection_percentage_{self._associated_subInst}.png")
        plt.close(fig)

        c = [
            "palegreen",
            "green",
            "darkgreen",
            "orange",
            "chocolate",
            "salmon",
            "red",
            "darkred",
            "purple",
            "black",
        ]
        v = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1]
        l = list(zip(v, c))
        cmap = LinearSegmentedColormap.from_list("rg", l, N=256)

        hori_dire = range(self.rejection_array.shape[1] + 1)
        verti_dire = range(self.rejection_array.shape[0] + 1)

        fig, ax = plt.subplots(figsize=(20, 10), constrained_layout=True)
        data = ax.pcolormesh(
            hori_dire,
            verti_dire,
            self.rejection_array,
            vmin=0,
            vmax=1,
            cmap=cmap,
            shading="auto",
            edgecolors="w",
        )
        fig.colorbar(data)
        ax.set_xlabel("Order")
        ax.set_ylabel("FrameID")
        ax.set_xticks(hori_dire[::10])

        ax.set_yticks(np.asarray(verti_dire)[:-1] + 0.5, labels=self.frameIDs_to_use)
        # ax.set_yticks(verti_dire, minor=False, labels=frameIDs)

        ax.set_ylim([0, np.max(verti_dire)])
        ax.set_xlim([0, np.max(hori_dire)])

        new_array = np.c_[self.frameIDs_to_use, self.rejection_array]

        new_array = np.r_[[[np.nan] + list(hori_dire)[:-1]], new_array]

        np.savetxt(
            metrics_path / f"order_pixel_rejections_{self._associated_subInst}.txt",
            new_array,
            fmt="%.3f",
        )
        fig.savefig(metrics_path / f"order_pixel_rejection_{self._associated_subInst}.pdf")
        plt.close(fig)

    def load_from_file(self, root_path, loading_path: str) -> None:
        super().load_from_file(root_path, loading_path)

        with fits.open(loading_path) as hdulist:
            if hdulist[1].header.get("VERSION", "") != __version__:
                logger.warning("Loaded template was not created under the current SBART version")
            self._associated_subInst = hdulist["WAVE"].header["subInst"]

            self.spectra = hdulist["TEMP"].data
            self.wavelengths = hdulist["WAVE"].data
            self.uncertainties = hdulist["UNCERTAINTIES"].data

            mask = hdulist["MASK"].data.astype(bool)
            self.spectral_mask = Mask(initial_mask=mask)

        self.array_size = self.spectra.shape

        storage_path = self._internalPaths.get_path_to("RunTimeProds", as_posix=False)

        detailedFlags_storage = build_filename(
            storage_path,
            f"DetailedFlags_{self._associated_subInst}",
            fmt="json",
        )

        self._OrderStatus = OrderStatus.load_from_json(detailedFlags_storage)

        miscInfo = build_filename(
            storage_path,
            f"miscInfo_{self._associated_subInst}",
            fmt="json",
        )

        with open(miscInfo) as file:
            json_info = json.load(file)

        json_info["used_fpaths"] = list([Path(i) for i in json_info["used_fpaths"]])

        for key, value in json_info.items():
            setattr(self, key, value)

    # Handle shared memory
    def convert_to_shared_mem(self):
        if self._in_shared_mem:
            # Avoid opening multiple arrays!
            logger.info(f"{self.__class__.name} already in shared memory")
            return
        logger.info("Putting the stellar template in shared memory")
        self._in_shared_mem = True

        array_of_zeros = np.zeros(self.wavelengths.shape)
        buffer_info, shared_uncerts = create_shared_array(array_of_zeros)
        self.shm["template_errors"] = buffer_info
        uncertainties = shared_uncerts

        buffer_info, shared_temp = create_shared_array(array_of_zeros)
        self.shm["template"] = buffer_info
        template = shared_temp

        buffer_info, shared_wave = create_shared_array(array_of_zeros)
        self.shm["template_counts"] = buffer_info
        counts = shared_wave

        buffer_info, shared_wave = create_shared_array(self.wavelengths)
        self.shm["template_wavelength"] = buffer_info
        wavelengths = shared_wave

        return wavelengths, template, uncertainties, counts

    def cleanup_shared_memory(self) -> None:
        logger.debug("Cleaning the shared memory interfaces from the Stellar template creation")
        self._close_workers()
        self._close_queues()
        self._close_shared_memory_arrays()

    def _close_workers(self) -> None:
        logger.debug("{} closing the workers", self.name)
        for _ in range(self._internal_configs["NUMBER_WORKERS"]):
            self.package_pool.put(np.nan)

    def _close_shared_memory_arrays(self):
        """Close the shared memory arrays"""
        logger.debug("{} closing the shared memory array", self.name)

        self._in_shared_mem = False
        for mem_block in self.shm.values():
            mem_block[0].close()
            mem_block[0].unlink()

        self.shm = {}

    def _close_queues(self):
        """Close the communication queues"""
        logger.debug("{} closing the communication queues", self.name)

        self.package_pool.close()
        self.output_pool.close()

        self.package_pool = None
        self.output_pool = None

    @property
    def storage_name(self) -> str:
        """
        Return the storage name for the stellar templates, which will include the name of the method,
        the RV source for the aligmenet
        Returns
        -------

        """
        base_name = f"{self.__class__.method_name}_{self._RV_source}"

        if self._merged_source is not None:
            base_name += "_"
            base_name += "merged" if self._merged_source else "individual"

        return base_name

    def interpolate_spectrum_to_wavelength(
        self,
        order,
        new_wavelengths,
        shift_RV_by,
        RV_shift_mode,
        remove_OB: Optional[Frame] = None,
        include_invalid=False,
    ):
        """
        Override of the interpolation algorithm to allow the rejection of OBs
        Parameters
        ----------
        order
        new_wavelengths
        shift_RV_by
        RV_shift_mode
        remove_OB
        include_invalid

        Returns
        -------

        """
        self.initialize_modelling_interface()

        wavelength, flux, uncertainties, mask = self.get_data_from_spectral_order(
            order, include_invalid
        )
        desired_inds = ~mask

        og_lambda, og_spectra, og_errs = (
            wavelength[desired_inds],
            flux[desired_inds],
            uncertainties[desired_inds],
        )

        if remove_OB is not None:  # TODO: check that this is working as expected
            if remove_OB.frameID in self.frameIDs_to_use:
                ccf_rv = convert_data(
                    remove_OB.get_KW_value(self.RV_keyword),
                    new_units=kilometer_second,
                    as_value=True,
                )
                new_forder, new_ferror = remove_OB.interpolate_spectrum_to_wavelength(
                    new_wavelengths=wavelength,
                    order=order,
                    shift_RV_by=ccf_rv,
                    RV_shift_mode="remove",
                )
                self.get_data_from_spectral_order(order, include_invalid)

                og_spectra = og_spectra - new_forder / len(self.frameIDs_to_use)
                og_errs = np.sqrt(og_errs**2 - new_ferror**2)

        if RV_shift_mode == "apply":
            shift_function = apply_RVshift
        elif RV_shift_mode == "remove":
            shift_function = remove_RVshift
        else:
            raise custom_exceptions.InvalidConfiguration("Unknown mode")

        og_lambda = shift_function(wave=og_lambda, stellar_RV=shift_RV_by)

        try:
            new_flux, new_errors = self.interpolation_interface.interpolate_spectrum_to_wavelength(
                og_lambda=og_lambda,
                og_spectra=og_spectra,
                og_err=og_errs,
                new_wavelengths=new_wavelengths,
                order=order,
            )
        except custom_exceptions.StopComputationError as exc:
            logger.critical("Interpolation of {} has failed", self.name)
            raise exc

        return np.asarray(new_flux), np.asarray(new_errors)

    @property
    def iteration_number(self) -> int:
        return self._iter_number

    def update_RV_source_info(self, iteration_number: int, RV_source: str, merged_source: bool):
        self._iter_number = iteration_number
        self._RV_source = RV_source
        self._merged_source = merged_source

    @property
    def RV_keyword(self):
        return "DRS_RV"
