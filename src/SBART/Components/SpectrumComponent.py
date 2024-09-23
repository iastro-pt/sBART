from typing import Set

import numpy as np
from loguru import logger

from SBART.utils import custom_exceptions
from SBART.utils.BASE import BASE
from SBART.utils.shift_spectra import (
    apply_approximated_BERV_correction,
    apply_BERV_correction,
    remove_approximated_BERV_correction,
    remove_BERV_correction,
)
from SBART.utils.status_codes import OrderStatus
from SBART.utils.types import RV_measurement
from SBART.utils.units import kilometer_second


class Spectrum(BASE):
    """
    Allow an SBART object to hold spectral data, providing a common interface for it. The goal of this class is to be a parent of
    the Frames and the templates (both stellar and telluric).

    This class introduces new attributes and methods into the children classes with the goal of proving a unifying framework.
    Not only does it allow to store the flux information, but it also created a pixel-wise mask to reject "bad" pixels.

    For more information we refer to the :class:`~SBART.Masks.Mask_class.Mask` class

    """

    def __init__(self, **kwargs):
        # If True, the object will never close its data to save resources
        self._never_close = False

        self._default_params = self._default_params
        self.has_spectrum_component = True

        super().__init__(**kwargs)

        self.qual_data = None  # error flags
        self.spectra = None  # S2D/S1D data
        self.wavelengths = None  # wavelengths in vacuum
        self.uncertainties = None  # Flux errors
        self.spectral_mask = None  # to be determined if I want this here or not .....
        self._blaze_function = None

        self.flux_atmos_balance_corrected = False
        self.is_blaze_corrected = False
        self.flux_dispersion_balance_corrected = False
        self.is_skysub = False
        self.is_BERV_corrected = False

        self._OrderStatus = None
        self.regenerate_order_status()

        # If True, then the data was loaded from disk. Otherwise, it still needs to be loaded in!
        self._spectrum_has_data_on_memory = False

        self.was_telluric_corrected = False

    def update_uncertainties(self, new_values):
        """
        Allow to update the uncertainty values, which allows for manual SNR changes

        Parameters
        ----------
        new_values
            Numpy array with the new uncertainties
        Returns
        -------

        """
        # self.uncertainties = full(self.spectra.shape, 200)
        if self.spectra.shape != new_values.shape:
            raise custom_exceptions.InvalidConfiguration(
                "The new uncertainties don't have the same size as the spectra"
            )
        self.uncertainties = new_values

    def regenerate_order_status(self):
        logger.warning(f"Resetting order status of {self.name}")
        try:
            if self.array_size is not None:
                self._OrderStatus = OrderStatus(self.N_orders)
        except AttributeError:
            pass

    def check_if_data_correction_enabled(self, property_name) -> bool:
        """
        If we attempt to access the correction state from the outside (before opening the S2D arrays), we will
        get a wrong value (of False), as the corrections are yet to be applied.
        Thus, we check if the desired property is currently given as a input argument. If it is, return its
        value, otherwise default to the class property values!
        """
        kw_map = {
            "flux_atmos_balance_corrected": "apply_FluxCorr",
            "flux_dispersion_balance_corrected": "apply_FluxBalance_Norm",
            "is_skysub": "is__skysub",
        }
        if property_name == "is_BERV_corrected":
            # Every frame will be berv corrected (if this is not done by the DRS)
            return True

        if property_name in [
            "was_telluric_corrected",
            "is_blaze_corrected",
            "is_skysub",
        ]:
            return getattr(self, property_name)

        if property_name not in kw_map:
            raise custom_exceptions.InternalError(
                "Searching for a data correction that is not available ({})",
                property_name,
            )
        try:
            return self._internal_configs[kw_map[property_name]]
        except KeyError:
            return getattr(self, property_name)

    def trigger_data_storage(self, *args, **kwargs):
        super().trigger_data_storage(args, kwargs)
        # Store whatever

    def apply_BERV_correction(self, BERV_value: RV_measurement) -> None:
        """
        If it hasn't been done before, apply the BERV correction to the wavelength solution of this frame.

        Parameters
        ----------
        BERV_value


        Returns
        -------

        """
        if self.is_BERV_corrected:
            return
        berv = BERV_value.to(kilometer_second).value
        berv_func = (
            apply_approximated_BERV_correction
            if self.use_approximated_BERV_correction
            else apply_BERV_correction
        )
        self.wavelengths = berv_func(self.wavelengths, berv)
        self.is_BERV_corrected = True

    def remove_BERV_correction(self, BERV_value: RV_measurement):
        """
        Remove the BERV correction from a given observation

        Parameters
        ----------
        BERV_value

        Returns
        -------

        """
        if not self.is_BERV_corrected:
            return
        berv = BERV_value.to(kilometer_second).value
        berv_func = (
            remove_approximated_BERV_correction
            if self.use_approximated_BERV_correction
            else remove_BERV_correction
        )

        self.wavelengths = berv_func(self.wavelengths, berv)
        self.is_BERV_corrected = False

    def apply_telluric_correction(self, wavelengths, model, model_uncertainty):
        """

        Divide the spectra by a telluric correction model, without really accounting for model uncertainties.
        This shouldn't be used in the current "state" ....

        Parameters
        ----------
        wavelengths
        model
        model_uncertainty

        Returns
        -------

        """
        if self.was_telluric_corrected:
            logger.warning(
                "Attempting to correct telluric features of previously corrected data. Doing nothing"
            )
            return

        if model.shape != self.spectra.shape:
            raise custom_exceptions.InvalidConfiguration(
                "Telluric correction model does not have the same shape as the S2D"
            )
        self.wavelengths = self.wavelengths / wavelengths
        self.spectra = self.spectra / model

        # TODO: actually account for uncertainties in the model
        self.uncertainties = self.uncertainties / model_uncertainty

    def _compute_BLAZE(self):
        """
        Estimate the BLAZE function by dividing BLAZE-corrected and BLAZE-uncorrected spectra. A children class must
        implement this, as normally don't have the paths to the two files

        Returns
        -------

        """
        raise NotImplementedError(
            "{} does not have a BLAZE computation tool".format(self.name)
        )

    def get_BLAZE_function(self):
        """
        Return the blaze function. If it is not available, attempt to compute it!

        Returns
        -------

        """
        logger.debug("{} retrieving Blaze function")
        if self._blaze_function is None:
            self._compute_BLAZE()

        return self._blaze_function

    def get_data_from_spectral_order(self, order: int, include_invalid: bool = False):
        """
        Retrieve a single order from the S2D matrix

        Parameters
        ----------
        order : int
            Order to retrive
        include_invalid: bool
            If False, raise exception when attempting to access data from bad order
        Returns
        -------
        np.ndarray
            wavelengths
        np.ndarray
            flux
        np.ndarray
            uncertainties
        np.ndarray
            Binary mask of the pixels
        """

        self._data_access_checks()
        if order in self.bad_orders and not include_invalid:
            raise custom_exceptions.BadOrderError(f"{order=} is invalid!")

        return (
            self.wavelengths[order],
            self.spectra[order],
            self.uncertainties[order],
            self.spectral_mask.get_custom_mask()[order],
        )

    def get_data_from_full_spectrum(self):
        """
        Retrieve the entire spectra.
        If we are working with S2D data: send the [N_orders, N_pixels] matrix
        If we are working with S1D data: send a single N_pixels 1-D array with the relevant information

        Returns
        ---------
        np.ndarray
            wavelengths
        np.ndarray
            Spectra
        np.ndarray
            Uncertainties
        np.ndarray
            Spectral mask, a binary mask which is set to one on the  **pixels to be discarded** and zero otherwise.

        """
        self._data_access_checks()
        if self.is_S2D:
            return (
                self.wavelengths,
                self.spectra,
                self.uncertainties,
                self.spectral_mask.get_custom_mask(),
            )
        elif self.is_S1D:
            # The S1D file is stored as a S2D with only one order!
            return (
                self.wavelengths,
                self.spectra,
                self.uncertainties,
                self.spectral_mask.get_custom_mask(),
            )

    def scale_spectra(self, factor: float):
        self._data_access_checks()
        logger.info("Scaling up frame!")
        self.spectra *= factor
        self.uncertainties *= factor

    def set_frame_as_Zscore(self):
        """
        Re-defining the frame as one with zero mean and unit-variance (z-score)
        Returns
        -------

        """
        logger.info("Setting up frame as a Zscore!")
        for order in range(self.N_orders):
            _, flux, _, mask = self.get_data_from_spectral_order(
                order=order, include_invalid=True
            )
            valid_mask = ~mask
            mean, std = np.mean(flux[valid_mask]), np.std(flux[valid_mask])
            self.spectra = (self.spectra - mean) / std

    def close_arrays(self):
        """
        Close the arrays that are currently open in memory. Next time we try to access them, the disk file will be re-opened.
        Saves RAM at the cost of more I/O operations

        """
        if self._never_close:
            logger.warning("Frame has been set to never close its arrays!")
            return
        self._spectrum_has_data_on_memory = False

        self.qual_data = None
        self.spectra = None
        self.wavelengths = None
        self.uncertainties = None

    @property
    def valid_orders(self):
        return (i for i in range(self.N_orders) if i not in self.bad_orders)

    @property
    def bad_orders(self) -> Set[int]:
        return self._OrderStatus.bad_orders

    @property
    def OrderStatus(self):
        """
        Return the Status of the entire Frame

        Returns
        -------
        OrderStatus
        """
        DeprecationWarning("Use 'OrderWiseStatus' instead of this method")
        return self.OrderWiseStatus

    @property
    def OrderWiseStatus(self) -> OrderStatus:
        """
        Returns the OrderStatus of the entire observation

        Returns
        -------
        OrderStatus

        """
        return self._OrderStatus

    @property
    def spectrum_information(self):
        return {
            "N_orders": self.N_orders,
            "object_type": self._object_type,
            "blaze_corrected": self.is_blaze_corrected,
            "flux_atmos_balance_corrected": self.flux_atmos_balance_corrected,
            "flux_dispersion_balance_corrected": self.flux_dispersion_balance_corrected,
            "telluric_corrected": self.was_telluric_corrected,
        }

    @property
    def N_orders(self) -> int:
        """
        Returns
        -------
        int
            Number of orders in the array
        """
        return self.array_size[0]

    @property
    def pixels_per_order(self) -> int:
        """
        Returns
        -------
        int
            Number of pixels in each order
        """
        return self.array_size[1]

    @property
    def is_open(self) -> bool:
        """
        Returns
        -------
        bool
            True if it has the arrays loaded on memory
        """
        return self._spectrum_has_data_on_memory
