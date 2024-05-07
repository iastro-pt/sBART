import pickle
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import NoReturn
from scipy.interpolate import CubicSpline

import matplotlib.pyplot as plt
import numpy as np
from astropy.units.format import fits

from SBART.utils import custom_exceptions
from loguru import logger

from SBART.spectral_normalization.normalization_base import NormalizationBase
from SBART.utils.UserConfigs import DefaultValues, PathValue, UserParam, BooleanValue


class RASSINE_normalization(NormalizationBase):
    """
    Uses RASSINE to normalize the stellar spectra.

    **Description:**

    Works with either S1D or S2D spectra, with a different behaviour on both cases:

    1) With S1D data:
        - Simple division of the stellar spectra with the continuum model. The continuum is interpolated (cubic spline)
    2) With S2D data:
        - Loads the S1D file from disk, applying the process from 1). Then, it divides the S1D spectra in chunks of "N_{order}" pixels to recreate a S2D spectra. This will re-trigger all the order masking procedures and remove all previous rejections

    **Name of the normalizer**: RASSINE

    **User parameters:**

    ====================== ================ ================ ======================== ================
    Parameter name             Mandatory      Default Value    Valid Values                Comment
    ====================== ================ ================ ======================== ================
        S1D_folder          False               ---             str, Path               [1]
        RASSINE_path        True                ---             str, Path               [2]
    ====================== ================ ================ ======================== ================


    Notes:
        [1] Folder in which the S1D files will be stored (if the input is a S2D spectra)

        [2] Path to a local clone of the modified RASSINE (git@github.com:Kamuish/Rassine_modified.git)

        [3] Also check the **User parameters** of the parent classes for further customization options of SBART


    Disk products:

        This method will create the output folder _Storage/RASSINE, where we can find:

        1) inputs/*.csv - RASSINE inputs
        2) outputs/*.png - Plot with the result of the continuum and normalization
        3) outputs/*.p - Rassine outputs
        4) *.json - Storage of params to use on next application


    **Example:**

    .. code-block:: python

        f = ESPRESSO(main_path / file_start,
                     user_configs={"NORMALIZE_SPECTRA": True,
                                   "NORMALIZATION_MODE": "RASSINE"
                                   "S1D_folder": "/home/foo/S1D_folder/target/",
                                   "RASSINE_path": "/home/foo/tools/Rassine_modified"
                                   }
             )

        f.normalize_spectra()
        f.trigger_data_storage()

    """

    _default_params = NormalizationBase._default_params + DefaultValues(
        S1D_folder=UserParam(mandatory=False, constraint=PathValue, default_value=""),
        RASSINE_path=UserParam(mandatory=True, constraint=PathValue),
    )

    _name = "RASSINE"

    orderwise_application = False

    def __init__(self, obj_info, user_configs):
        super().__init__(
            obj_info=obj_info,
            user_configs=user_configs,
            needed_folders={
                "RASSINE_IN": "_Storage/SpecNorm/RASSINE/inputs",
                "RASSINE_OUT": "_Storage/SpecNorm/RASSINE/outputs",
            },
        )
        if obj_info["is_S2D"] and "S1D_folder" not in user_configs:
            raise custom_exceptions.InvalidConfiguration(
                "Must provide the S1D folder when using S2D files"
            )

    def _get_S1D_data(self, wavelengths, flux, uncertainties):
        if self._spec_info["is_S2D"]:
            S1D_path = self._internal_configs["S1D_folder"] / self._spec_info["S1D_name"]
            temp_configs = deepcopy(self._internal_configs.get_user_configs())
            temp_configs["spectra_format"] = "S1D"
            # open a temporary frame to retrieve the S1D data!
            new_frame = self._spec_info["Frame_instance"](
                file_path=S1D_path,
                user_configs=temp_configs,
            )
            wavelengths, flux, uncertainties, _ = new_frame.get_data_from_full_spectrum()

        return (
            wavelengths[0],
            flux[0],
            uncertainties[0],
        )  # the S1D file is considered to be a very "large" order

    def _prepare_Rassine_run(self, wavelengths, flux, uncertainties):
        logger.info("Preparing text files for RASSINE application")

        wavelengths, flux, uncertainties = self._get_S1D_data(wavelengths, flux, uncertainties)

        # Concatenate the arrays for RASSINE
        arr = np.c_[wavelengths, flux]

        filename = self._spec_info["S1D_name"]
        filename = filename.replace("fits", "csv")
        logger.debug(
            f'Storing RASSINE input data to {self._internalPaths.get_path_to("RASSINE_IN", as_posix=False) / filename}'
        )
        # Ensure the format that is expected by RASSINE
        np.savetxt(
            self._internalPaths.get_path_to("RASSINE_IN", as_posix=False) / filename,
            arr,
            header="wave,flux",
            delimiter=",",
            comments="",
        )

        ## Update the Rassine config file!
        rassine_config = (
            """import os 
cwd = os.getcwd()

# =============================================================================
#  ENTRIES
# ==========================================================================
"""
            + f'spectrum_name = \'{(self._internalPaths.get_path_to("RASSINE_IN", as_posix=False) / filename).absolute()}\''
            + f'\noutput_dir  = \'{self._internalPaths.get_path_to("RASSINE_OUT", as_posix=False).absolute()}\'  '
            + """
synthetic_spectrum = False   # True if working with a noisy-free synthetic spectra 
anchor_file = ''             # Put a RASSINE output file that will fix the value of the 7 parameters to the same value than in the anchor file

column_wave = 'wave'
column_flux = 'flux'

float_precision = 'float64' # float precision for the output products wavelength grid

#general initial parameters

par_stretching = 'auto_0.5'     # stretch the x and y axes ratio ('auto' available)                            <--- PARAMETER 1
par_vicinity = 7                # half-window to find a local maxima                                           
                                
par_smoothing_box = 6           # half-window of the box used to smooth (1 => no smoothing, 'auto' available)  <--- PARAMETER 2
par_smoothing_kernel = 'savgol' # 'rectangular','gaussian','savgol' if a value is specified in smoothig_kernel
                                # 'erf','hat_exp' if 'auto' is given in smoothing box                     
                   
par_fwhm = 'auto'               # FWHM of the CCF in km/s ('auto' available)                                   <--- PARAMETER 3
CCF_mask = 'master'             # only needed if par_fwhm is in 'auto'
RV_sys = 0                      # RV systemic in kms, only needed if par_fwhm is in 'auto' and CCF different of 'master'
mask_telluric = [[6275,6330],   # a list of left and right borders to eliminate from the mask of the CCF
                 [6470,6577],   # only if CCF = 'master' and par_fwhm = 'auto'
                 [6866,8000]] 

par_R = 'auto'             # minimum radius of the rolling pin in angstrom ('auto' available)                  <--- PARAMETER 4
par_Rmax = 'auto'          # maximum radius of the rolling pin in angstrom ('auto' available)                  <--- PARAMETER 5
par_reg_nu = 'poly_1.0'    # penality-radius law                                                               <--- PARAMETER 6
                           # poly_d (d the degree of the polynome x**d)
                           # or sigmoid_c_s where c is the center and s the steepness

denoising_dist = 5      # half window of the area used to average the number of point around the local max for the continuum
count_cut_lim = 3       # number of border cut in automatic mode (put at least 3 if Automatic mode)
count_out_lim = 1       # number of outliers clipping in automatic mode (put at least 1 if Automatic mode)


interpolation = 'cubic' # define the interpolation for the continuum displayed in the subproducts            
                        # note that at the end a cubic and linear interpolation are saved in 'output' regardless this value

feedback = False        # run the code without graphical feedback and interactions with the sphinx (only wishable if lot of spectra)     
only_print_end = False  # only print in the console the confirmation of RASSINE ending
plot_end = False        # display the final product in the graphic
save_last_plot = False  # save the last graphical output (final output)


outputs_interpolation_saved = 'all' # to only save a specific continuum (output files are lighter), either 'linear','cubic' or 'all'
outputs_denoising_saved = 'undenoised'        # to only save a specific continuum (output files are lighter), either 'denoised','undenoised' or 'all'

light_version = True    # to save only the vital output


config = {'spectrum_name':spectrum_name,
          'synthetic_spectrum':synthetic_spectrum,
          'output_dir':output_dir,
          'anchor_file':anchor_file,
          'column_wave':column_wave,
          'column_flux':column_flux,
          'axes_stretching':par_stretching,
          'vicinity_local_max':par_vicinity,
          'smoothing_box':par_smoothing_box,
          'smoothing_kernel':par_smoothing_kernel,
          'fwhm_ccf':par_fwhm,
          'CCF_mask':CCF_mask,
          'RV_sys':RV_sys,
          'mask_telluric':mask_telluric,
          'min_radius':par_R,
          'max_radius':par_Rmax,
          'model_penality_radius':par_reg_nu,
          'denoising_dist':denoising_dist,
          'interpol':interpolation,
          'number_of_cut':count_cut_lim,
          'number_of_cut_outliers':count_out_lim,
          'float_precision':float_precision,
          'feedback':feedback,
          'only_print_end':only_print_end,
          'plot_end':plot_end,
          'save_last_plot':save_last_plot,
          'outputs_interpolation_save':outputs_interpolation_saved,
          'outputs_denoising_save':outputs_denoising_saved,
          'light_file':light_version,
          'speedup':1}                     
        """
        )
        with open(
            Path(self._internal_configs["RASSINE_path"]) / "Rassine_config.py", mode="w"
        ) as file:
            file.write(rassine_config)

    def run_RASSINE(self, wavelengths, flux, uncertainties):
        self._prepare_Rassine_run(wavelengths, flux, uncertainties)
        # TODO: check the commands to launch RASSINE
        logger.info("Launching RASSINE")

        subprocess.run(["python", Path(self._internal_configs["RASSINE_path"]) / "Rassine.py"])

        logger.info("RASSINE has finished running")

    def _fit_epochwise_normalization(self, wavelengths, flux, uncertainties):
        super()._fit_epochwise_normalization(wavelengths, flux, uncertainties)

        self.run_RASSINE(wavelengths, flux, uncertainties)
        filename = self._spec_info["S1D_name"].replace(".fits", "")

        output_path = (
            self._internalPaths.get_path_to("RASSINE_OUT", as_posix=False) / f"RASSINE_{filename}.p"
        )

        # TODO: missing the parameters that will be cached!
        params_to_store = {"RASSINE_OUT_FOLDER": output_path.as_posix()}

        return *self._apply_epoch_normalization(
            wavelengths, flux, uncertainties, **params_to_store
        ), params_to_store

    def _apply_epoch_normalization(self, wavelengths, flux, uncertainties, **kwargs):
        super()._apply_epoch_normalization(wavelengths, flux, uncertainties, **kwargs)
        logger.info(f"Applying normalization to epoch!")
        og_shape = wavelengths.shape

        wavelengths, flux, uncertainties = self._get_S1D_data(wavelengths, flux, uncertainties)

        # TODO: think about SNR problems that might arise within SBART if this goes through without adding an offset

        rassine_out_path = kwargs["RASSINE_OUT_FOLDER"]
        with open(rassine_out_path, mode="rb") as file:
            rass_products = pickle.load(file)

        cont_solution = rass_products["output"]["continuum_cubic"]

        CSplineInterpolator = CubicSpline(rass_products["wave"], cont_solution)
        cont_solution = CSplineInterpolator(wavelengths)

        # Ensure that we are not interpolating outside the grid!
        # In principle, this should not be a problem, as the grid **should** be large enough to
        # contain the entire wavelength solution
        cont_solution[
            np.logical_or(
                wavelengths < rass_products["wave"][0], wavelengths > rass_products["wave"][-1]
            )
        ] = np.nan
        self.plot_rassine_products(wavelengths, flux, uncertainties, rass_products, cont_solution)

        flux /= cont_solution
        uncertainties /= cont_solution

        return wavelengths, flux, uncertainties

    def plot_rassine_products(
        self, wavelength, flux, uncert, rass_products, interpolated_cont
    ) -> NoReturn:
        """
        Plot the end result of the continuum normalization

        Parameters
        ----------
        wavelength
        flux
        uncert
        rass_products

        Returns
        -------

        """
        if not self._ran_normalization_fit:
            return
        logger.debug("Plotting rassine products")
        fig, axis = plt.subplots(nrows=2, figsize=(8, 6), sharex=True)
        figs_to_close = [fig]

        axis[0].plot(wavelength, flux, color="black")
        axis[0].plot(wavelength, interpolated_cont, color="blue")
        axis[1].plot(wavelength, flux / interpolated_cont, color="black")
        axis[0].plot(rass_products["wave"], rass_products["output"]["continuum_cubic"], color="red")
        axis[1].set_xlabel(r"$\lambda [\AA]$")
        axis[0].set_ylabel("Flux")
        axis[1].set_ylabel("Normalized flux")

        filename = self._spec_info["S1D_name"].replace(".fits", ".png")
        fig.savefig(self._internalPaths.get_path_to("RASSINE_OUT", as_posix=False) / filename)
        for fig in figs_to_close:
            plt.close(fig)

    def _normalization_sanity_checks(self):
        # TODO: check this, maybe we will be limited to BLAZE-corrected spectra!
        logger.debug("{} does not apply any sanity check on the data!")
        if self._spec_info["is_S2D"]:
            raise custom_exceptions.InvalidConfiguration("Can't normalize S2D spectra")
