from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from astropy.units.format import fits

from SBART.utils import custom_exceptions
from loguru import logger

from SBART.spectral_normalization.normalization_base import NormalizationBase
from SBART.utils.UserConfigs import (
    DefaultValues,
    PathValue,
    UserParam,
    BooleanValue
)


class RASSINE_normalization(NormalizationBase):
    """

    **User parameters:**


    *Note:* Also check the **User parameters** of the parent classes for further customization options of SBART

    """

    _default_params = NormalizationBase._default_params + \
                      DefaultValues(S1D_folder=UserParam(mandatory=False, constraint=PathValue, default_value=""),
                                    )
    _name = "RASSINE"

    orderwise_application = False

    def __init__(self, obj_info, user_configs):
        super().__init__(obj_info=obj_info,
                         user_configs=user_configs,
                         needed_folders={"RASSINE_IN": "_Storage/RASSINE_inputs"}
                         )
        if obj_info["is_S2D"] and "S1D_folder" not in user_configs:
            raise custom_exceptions.InvalidConfiguration("Must provide the S1D folder when using S2D files")

    def _prepare_Rassine_run(self, wavelengths, flux, uncertainties):
        logger.info("Preparing text files for RASSINE application")
        if self._spec_info["is_S2D"]:
            S1D_path = self._internal_configs["S1D_folder"] / self._spec_info["S1D_name"]
            temp_configs = deepcopy(self._internal_configs.get_user_configs())
            temp_configs["spectra_format"] = "S1D"
            # open a temporary frame to retrieve the S1D data!
            new_frame = self._spec_info["Frame_instance"](file_path=S1D_path,
                                                          user_configs=temp_configs,
                                                          )
            wavelengths, flux, uncertainties, _ = new_frame.get_data_from_full_spectrum()

        # Concatenate the arrays for RASSINE
        arr = np.c_[wavelengths[0], flux[0], uncertainties[0]]

        filename = self._spec_info["S1D_name"]
        filename = filename.replace("fits", "txt")
        logger.debug(f'Storing RASSINE input data to {self._internalPaths.get_path_to("RASSINE_IN", as_posix=False) / filename}')
        np.savetxt(self._internalPaths.get_path_to("RASSINE_IN", as_posix=False) / filename,
                   arr
                   )

    def run_RASSINE(self, wavelengths, flux, uncertainties):
        self._prepare_Rassine_run(wavelengths, flux, uncertainties)
        # TODO: check the commands to launch RASSINE

    def _fit_epochwise_normalization(self, wavelengths, flux, uncertainties):
        # TODO: think about SNR problems that might arise within SBART if this goes through without adding an offset
        # TODO: search for the S1D files
        s1d_sol, s1d_sci, s1d_err = np.zeros(wavelengths.shape), np.zeros(wavelengths.shape), np.zeros(wavelengths.shape)

        ## Build txt file inputs

        self.run_RASSINE(wavelengths, flux, uncertainties)

        if self._spec_info["is_S2D"]:
            # This must be done on the RASSINE outputs!
            return
            # If the input frame is a S2D file, then we re-arrange the S1D file to fit the expected "shape"
            # of the S2D files!

            reconstructed_S2D = np.zeros(wavelengths.shape)
            reconstructed_wavelengths = np.zeros(wavelengths.shape)
            reconstructed_uncertainties = np.zeros(wavelengths.shape)

            order_number = 0
            order_size = wavelengths[0].size
            to_break = False

            while not to_break:
                start_order = order_size * order_number
                end_order = start_order + order_size
                if end_order >= s1d_sol.size:
                    to_break = True
                    end_order = s1d_sol.size

                slice_size = end_order - start_order
                reconstructed_wavelengths[order_number] = np.pad(s1d_sol[start_order:end_order], (0, order_size - slice_size))
                reconstructed_S2D[order_number] = np.pad(s1d_sci[start_order:end_order], (0, order_size - slice_size))
                reconstructed_uncertainties[order_number] = np.pad(s1d_err[start_order:end_order], (0, order_size - slice_size))
                order_number += 1
            # The "new" orders that don't have any information will have a flux of zero. Thus, they will be deemed to
            # be invalid during the mask creation process (that is re-launched after this routine is done)

        # TODO: missing the parameters that will be cached!
        return reconstructed_wavelengths, reconstructed_S2D, reconstructed_uncertainties, {}

    def _apply_epoch_normalization(self, wavelengths, flux, uncertainties, extra_info, **kwargs):
        super()._apply_epoch_normalization(wavelengths, flux, uncertainties, extra_info, **kwargs)

        return flux, uncertainties

    def _normalization_sanity_checks(self):
        # TODO: check this, maybe we will be limited to BLAZE-corrected spectra!
        logger.debug("{} does not apply any sanity check on the data!")
