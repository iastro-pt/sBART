from typing import NoReturn, Set

from loguru import logger

from SBART.Base_Models.BASE import BASE
from SBART.utils import custom_exceptions
from SBART.utils.shift_spectra import apply_BERV_correction
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
        self._default_params = self._default_params
        self.has_spectrum_component = True

        super().__init__(**kwargs)

        self.qual_data = None  # error flags
        self.spectra = None  # S2D/S1D data
        self.wavelengths = None  # wavelengths in vacuum
        self.uncertainties = None  # Flux errors
        self.spectral_mask = None  # to be determined if I want this here or not .....
        self._blaze_function = None

        self.is_blaze_corrected = False
        self.is_BERV_corrected = False

        self._OrderStatus = None
        try:
            if self.array_size is not None:
                self._OrderStatus = OrderStatus(self.N_orders)
        except AttributeError:
            pass

        # If True, then the data was loaded from disk. Otherwise, it still needs to be loaded in!
        self._spectrum_has_data_on_memory = False

        self.was_telluric_corrected = False

    def trigger_data_storage(self, *args, **kwargs) -> NoReturn:
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
        self.wavelengths = apply_BERV_correction(self.wavelengths, berv)
        self.is_BERV_corrected = True

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
        raise NotImplementedError("{} does not have a BLAZE computation tool".format(self.name))

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
            raise custom_exceptions.BadOrderError()

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

    def close_arrays(self) -> NoReturn:
        """
        Close the arrays that are currently open in memory. Next time we try to access them, the disk file will be re-opened.
        Saves RAM at the cost of more I/O operations

        """
        self._spectrum_has_data_on_memory = False

        self.qual_data = None
        self.spectra = None
        self.wavelengths = None
        self.uncertainties = None

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
        return self.OrderWiseStatusr

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
