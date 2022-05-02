import copy
import json
import traceback
from pathlib import Path
from typing import Callable, NoReturn

import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
import tinygp

jax.config.update("jax_enable_x64", True)
from loguru import logger

from SBART.Base_Models.BASE import BASE
from SBART.ModelParameters import JaxComponent, Model
from SBART.utils import custom_exceptions
from SBART.utils.paths_tools import build_filename
from SBART.utils.status_codes import CONVERGENCE_FAIL, INTERNAL_ERROR, SUCCESS
from SBART.utils.UserConfigs import (
    BooleanValue,
    DefaultValues,
    UserParam,
    ValueFromList,
)

global kern_type


class Spectral_Modelling(BASE):
    """

    Introduces, in a given object, the functionaitly to model and interpolate the stellar orders.
    In order to inherit from this class, it must also be a children of :class:`SBART.Components.SpectrumComponent.Spectrum`

    **User parameters:**

    ============================ ================ ================ ======================== ================
    Parameter name                 Mandatory      Default Value    Valid Values                 Comment
    ============================ ================ ================ ======================== ================
    GP_KERNEL                       False           Matern-5_2      Matern-5_2; Matern-3_2      Which kernel to use
    FORCE_MODEL_GENERATION          False           False           boolean                     If True, don't use disk-stored data
    POSTERIOR_CHARACTERIZATION      False           minimize        minimize/MCMC               How to explore posterior dist
    ============================ ================ ================ ======================== ================

    *Note:* Also check the **User parameters** of the parent classes for further customization options of SBART

    """

    # TODO: confirm the kernels that we want to allow
    _default_params = BASE._default_params + DefaultValues(
        GP_KERNEL=UserParam("Matern-5_2", constraint=ValueFromList(["Matern-5_2", "Matern-3_2"])),
        FORCE_MODEL_GENERATION=UserParam(False, constraint=BooleanValue),
        POSTERIOR_CHARACTERIZATION=UserParam(
            "minimize", constraint=ValueFromList(["minimize", "MCMC"])
        ),
    )

    def __init__(self, **kwargs):
        self._default_params = self._default_params + Spectral_Modelling._default_params
        self.has_modelling_component = True

        # Middleman "attack" to inject the needed folders for this component
        if "needed_folders" not in kwargs or kwargs["needed_folders"] is None:
            kwargs["needed_folders"] = {}

        # TODO: This might bring problems with the root level! take care...
        possible_folders = {}
        for kern in self._default_params["GP_KERNEL"].existing_constraints.available_options:
            possible_folders[f"modelling_information_{kern}"] = f"_SpecModelParams/{kern}"

        kwargs["needed_folders"] = {
            **kwargs["needed_folders"],
            **possible_folders,
        }

        super().__init__(**kwargs)

        # Going to treat the orders as frameIDs to use the Model class!
        self._modelling_parameters = Model(
            params_of_model=[
                JaxComponent(
                    "log_scale",
                    initial_guess=jnp.log(0.001),
                    bounds=[None, None],
                    default_enabled=True,
                ),
                JaxComponent(
                    "log_amplitude",
                    initial_guess=jnp.log(1),
                    bounds=[None, None],
                    default_enabled=True,
                ),
                JaxComponent(
                    "log_mean", initial_guess=jnp.log(1), bounds=[None, None], default_enabled=True
                ),
            ]
        )

        # Avoid multiple loads of disk information
        self._loaded_disk_model = False

        # Avoid multiple calls to disk loading if the file does not exist
        self._attempted_to_load_disk_model: bool = False

        if not self.has_spectrum_component:
            # TODO: ensure that it is safe to do this in here
            # TODO 1: won't this raise an Exception depending on the instantiation order???
            raise Exception("Can't add modelling component to class without a spectrum")

    def _get_model_storage_filename(self) -> str:
        """
        Construct the storage filename for the model parameters for this observation!
        Returns
        -------

        """
        if self.is_object_type("Frame"):
            filename_start = self.bare_fname
        elif self.is_object_type("Template"):
            filename_start = "Template"
        else:
            raise custom_exceptions.InvalidConfiguration(
                "Spectral modelling can't save results for {}", self._object_type
            )

        output_folder = self._internalPaths.get_path_to(
            f"modelling_information_{self._internal_configs['GP_KERNEL']}", as_posix=False
        )
        full_fname = build_filename(output_folder, f"{filename_start}_GP_model", "json")
        return full_fname

    def _init_model(self):
        for order in range(self.N_orders):
            self._modelling_parameters.generate_prior_from_frameID(order)

    def load_previous_model_results_from_disk(self) -> NoReturn:
        """
        Load the json-stored disk data

        Returns
        -------

        """
        logger.warning("{} {}", self._loaded_disk_model, self._attempted_to_load_disk_model)
        if self._loaded_disk_model or self._attempted_to_load_disk_model:
            return

        self._attempted_to_load_disk_model = True
        self._init_model()
        logger.debug("Searching for previous model on disk")

        try:
            storage_name = self._get_model_storage_filename()
        except custom_exceptions.MissingRootPath:
            logger.debug("Missing Root path information. Giving up on loading data")
            raise custom_exceptions.NoDataError

        try:
            loaded_model = Model.load_from_json(storage_name, component_to_use=JaxComponent)
            self._loaded_disk_model = True
            self._modelling_parameters = loaded_model
        except FileNotFoundError:
            self._loaded_disk_model = False
            logger.debug("Failed to find disk model")
            raise custom_exceptions.NoDataError

    def _store_model_to_disk(self) -> NoReturn:
        """
        Store the fit parameters to disk, to avoid multiple computations in the future

        Returns
        -------

        """
        if not self._modelling_parameters.has_results_stored:
            return
        logger.info("Storing parameters of GP models to disk")

        full_fname = self._get_model_storage_filename()

        self._modelling_parameters.save_to_json_file(full_fname)

        logger.debug("Finished storage of spectral model")

    def generate_model_from_order(self, order: int) -> NoReturn:
        """
        Fit the stellar spectrum from a given order. If it has already been recomputed (or if it has previously failed)
        does nothing

        Parameters
        ----------
        order

        Returns
        -------

        """
        if not self.is_blaze_corrected:
            msg = "Currently the mean function can't handle the blaze spectrum"
            logger.critical(msg)
            raise NotImplementedError(msg)

        if not self._internal_configs["FORCE_MODEL_GENERATION"]:
            try:
                if not self._attempted_to_load_disk_model:
                    self.load_previous_model_results_from_disk()
            except custom_exceptions.NoDataError:
                pass

        if self._modelling_parameters.has_valid_identifier_results(order):
            logger.info("Parameters already exist on memory. Skipping")
            return

        try:
            solution_array, result_flag = self._launch_GP_fit(order=order)
        except Exception as e:
            msg = "Unknown error found when fitting GP: {}".format(
                traceback.print_tb(e.__traceback__)
            )
            logger.critical(msg)
            result_flag = INTERNAL_ERROR(msg)
            solution_array = [np.nan for _ in self._modelling_parameters.get_enabled_params()]

        self._modelling_parameters.store_frameID_results(
            order, result_vector=solution_array, result_flag=result_flag
        )

    def interpolate_spectrum_to_wavelength(self, order, new_wavelengths):
        """
        Interpolate the order of this spectrum to a given wavelength, using a GP. If the GP fit is yet to be done,
        then it is done beforehand.

        Parameters
        ----------
        order
            Spectral order to interpolate
        new_wavelengths
            New wavelength solution, for which we want to interpolate the spectrum to

        Returns
        -------
        mu
            Mean prediction
        sigma
            Model uncertainty

        Raises
        --------
        NoConvergenceError
            If the fit for this order failed
        """

        self.generate_model_from_order(order)
        fit_results = self._modelling_parameters.get_fit_results_from_frameID(order)
        param_names = self._modelling_parameters.get_enabled_params()

        optimal_combinations = {i: j for i, j in zip(param_names, fit_results)}
        wavelengths, fluxes, uncertainties, mask = self.get_data_from_spectral_order(order=order)
        order_mask = ~mask
        data_dict = {
            "XX_data": jnp.asarray(wavelengths[order_mask]),
            "YY_variance": jnp.asarray(uncertainties[order_mask] ** 2),
        }

        gp_object = build_gp(optimal_combinations, **data_dict)
        _, cond = gp_object.condition(fluxes[order_mask], X_test=new_wavelengths)

        mu = cond.loc
        std = np.sqrt(cond.variance)
        return mu, std

    def trigger_data_storage(self, *args, **kwargs) -> NoReturn:
        super().trigger_data_storage(args, kwargs)

        self._store_model_to_disk()

    def _launch_GP_fit(self, order):

        initial_params, bounds = self._modelling_parameters.generate_optimizer_inputs(
            order, rv_units=None
        )
        param_names = self._modelling_parameters.get_enabled_params()

        result_flag = SUCCESS
        initial_guess = {i: j for i, j in zip(param_names, initial_params)}
        wavelengths, fluxes, uncertainties, mask = self.get_data_from_spectral_order(order=order)
        order_mask = ~mask
        data_dict = {
            "XX_data": jnp.asarray(wavelengths[order_mask]),
            "YY_data": jnp.asarray(fluxes[order_mask]),
            "YY_variance": jnp.asarray(uncertainties[order_mask] ** 2),
        }

        global kern_type

        kern_type = self._internal_configs["GP_KERNEL"]

        self._modelling_parameters.update_params_initial_guesses(
            frameID=order,
            guesses={
                "log_mean": jnp.log(np.mean(fluxes[order_mask])),
                "log_amplitude": jnp.log(np.max(fluxes[order_mask]) - np.min(fluxes[order_mask])),
            },
        )

        if self._internal_configs["POSTERIOR_CHARACTERIZATION"] == "minimize":
            solver = jaxopt.ScipyMinimize(
                fun=loss, options={"maxiter": 1000, "disp": True}, method="BFGS", tol=1e-10
            )
            print(initial_guess, "\n --..-- \n", data_dict)
            soln = solver.run(jax.tree_map(jnp.asarray, initial_guess), **data_dict)
            print(soln)
            solution_array = [soln.params[name] for name in param_names]
            [print(np.exp(i)) for i in solution_array]

        elif self._internal_configs["POSTERIOR_CHARACTERIZATON"] == "MCMC":
            raise NotImplementedError

        # TODO: evaluate convergence and raise warnings/errors if needed

        return solution_array, result_flag


def generate_kernel(amplitude, length_scale):
    """
    Build a tinygp quasiseperable kernel with the provided amplitude and length_scale

    Parameters
    ----------
    amplitude
        Amplitude parameter
    length_scale
        Kernel's lenght scale
    kernel_type:
        Type of kernel, following the **User Parameters** of the Modelling class.
    Returns
    -------

    """
    global kern_type

    if kern_type == "Matern-5_2":
        cov_structure = tinygp.kernels.quasisep.Matern52(scale=length_scale)
    elif kern_type == "Matern-3_2":
        cov_structure = tinygp.kernels.quasisep.Matern32(scale=length_scale)
    else:
        raise Exception
    return amplitude * cov_structure
    # Building the tinyGP model for minimization


def build_gp(
    params,
    XX_data,
    YY_variance,
):
    kernel = generate_kernel(
        amplitude=jnp.exp(params["log_amplitude"]), length_scale=jnp.exp(params["log_scale"])
    )

    return tinygp.GaussianProcess(
        kernel,
        XX_data,
        diag=YY_variance,
        mean=jnp.exp(params["log_mean"]),
    )


@jax.jit
def loss(params, XX_data, YY_data, YY_variance):
    # TODO: understand if we can pass args to this function!
    gp = build_gp(params, XX_data, YY_variance)
    return -gp.log_probability(YY_data)
