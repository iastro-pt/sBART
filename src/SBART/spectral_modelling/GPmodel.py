import traceback
from typing import NoReturn
from functools import partial
import numpy as np

from loguru import logger

from SBART.ModelParameters.Parameter import JaxComponent
from SBART.utils import custom_exceptions
from SBART.utils.paths_tools import build_filename

try:
    import jax, jaxopt
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    import tinygp

    MISSING_TINYGP = False
except ImportError:
    MISSING_TINYGP = True
    # To avoid issues with possible needs of jax
    import numpy as jnp

from SBART.ModelParameters import Model

from SBART.utils.UserConfigs import (
    BooleanValue,
    DefaultValues,
    UserParam,
    ValueFromList,
    Positive_Value_Constraint,
)

from SBART.utils.status_codes import INTERNAL_ERROR, SUCCESS


from SBART.spectral_modelling.modelling_base import ModellingBase


class GPSpecModel(ModellingBase):
    """

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
    _default_params = ModellingBase._default_params + DefaultValues(
        GP_KERNEL=UserParam("Matern-5_2", constraint=ValueFromList(["Matern-5_2", "Matern-3_2"])),
        FORCE_MODEL_GENERATION=UserParam(False, constraint=BooleanValue),
        POSTERIOR_CHARACTERIZATION=UserParam(
            "minimize", constraint=ValueFromList(["minimize", "MCMC"])
        ),
        OPTIMIZATION_MAX_ITER=UserParam(1000, constraint=Positive_Value_Constraint),
    )

    def __init__(self, obj_info, user_configs):
        possible_folders = {}
        for kern in self._default_params["GP_KERNEL"].existing_constraints.available_options:
            possible_folders[f"modelling_information_{kern}"] = f"_SpecModelParams/{kern}"

        super().__init__(
            obj_info=obj_info, user_configs=user_configs, needed_folders=possible_folders
        )

        # Going to treat the orders as frameIDs to use the Model class!

        params_of_model = [
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
        for parameter in params_of_model:
            self._modelling_parameters.add_extra_param(parameter)

        # Ensuring that we initialize the model again, as we added new parameters!
        self._init_model()

    def _check_dependencies(self):
        if MISSING_TINYGP:
            raise custom_exceptions.InternalError("Missing the custom tinygp installation for GP")

    def _get_model_storage_filename(self) -> str:
        """
        Construct the storage filename for the model parameters for this observation!
        Returns
        -------

        """
        filename_start = super()._get_model_storage_filename()

        output_folder = self._internalPaths.get_path_to(
            f"modelling_information_{self._internal_configs['GP_KERNEL']}", as_posix=False
        )
        full_fname = build_filename(output_folder, f"{filename_start}_GP_model", "json")
        return full_fname

    def load_previous_model_results_from_disk(self, model_component_in_use):
        model_component_in_use = JaxComponent
        # TODO: ensure that loaded information is in accordance with the current parameters!
        return super().load_previous_model_results_from_disk(model_component_in_use)

    def generate_model_from_order(
        self, og_lambda, og_spectra, og_err, new_wavelengths, order
    ) -> NoReturn:
        """
        Fit the stellar spectrum from a given order. If it has already been recomputed (or if it has previously failed)
        does nothing

        Parameters
        ----------
        order

        Returns
        -------

        """
        self._check_dependencies()
        if not self.object_info["blaze_corrected"]:
            msg = "Currently the GP mean function can't handle the blaze spectrum"
            logger.critical(msg)
            raise NotImplementedError(msg)

        try:
            super().generate_model_from_order(order=order)
        except custom_exceptions.AlreadyLoaded:
            return

        try:
            solution_array, result_flag = self._launch_GP_fit(
                og_lambda, og_spectra, og_err, new_wavelengths, order
            )
        except Exception as e:
            msg = "Unknown error found when fitting GP to order {}: {}".format(
                order, traceback.print_tb(e.__traceback__)
            )
            logger.critical(msg)
            result_flag = INTERNAL_ERROR(msg)
            solution_array = [np.nan for _ in self._modelling_parameters.get_enabled_params()]

            raise custom_exceptions.StopComputationError("Unknown error encounterd") from e

        self._modelling_parameters.store_frameID_results(
            order, result_vector=solution_array, result_flag=result_flag
        )

    def _store_model_to_disk(self) -> NoReturn:
        # TODO: store information related with the GP parameters!
        return super()._store_model_to_disk()

    def interpolate_spectrum_to_wavelength(
        self, og_lambda, og_spectra, og_err, new_wavelengths, order
    ):
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

        import time

        t0 = time.time()

        t1 = time.time()
        self.generate_model_from_order(og_lambda, og_spectra, og_err, new_wavelengths, order)

        kern_type = self._internal_configs["GP_KERNEL"]

        try:
            fit_results = self._modelling_parameters.get_fit_results_from_frameID(order)
        except custom_exceptions.NoConvergenceError as exc:
            logger.critical(
                "Can't interpolate wavelengths from order that has not achieved convergence"
            )
            raise exc

        param_names = self._modelling_parameters.get_enabled_params()

        optimal_combinations = {i: j for i, j in zip(param_names, fit_results)}
        data_dict = {
            "XX_data": jnp.asarray(og_lambda),
            "YY_variance": jnp.asarray(og_err**2),
            "kern_type": kern_type,
        }

        t1 = time.time()
        gp_object = build_gp(optimal_combinations, **data_dict)
        _, cond = gp_object.condition(og_spectra, X_test=new_wavelengths)

        mu = cond.loc
        std = np.sqrt(cond.variance)
        return mu, std

    def _launch_GP_fit(self, og_lambda, og_spectra, og_err, new_wavelengths, order):
        initial_params, bounds = self._modelling_parameters.generate_optimizer_inputs(
            order, rv_units=None
        )
        param_names = self._modelling_parameters.get_enabled_params()

        result_flag = SUCCESS
        initial_guess = {i: j for i, j in zip(param_names, initial_params)}
        data_dict = {
            "XX_data": jnp.asarray(og_lambda),
            "YY_data": jnp.asarray(og_spectra),
            "YY_variance": jnp.asarray(og_err**2),
        }

        self._modelling_parameters.update_params_initial_guesses(
            frameID=order,
            guesses={
                "log_mean": jnp.log(jnp.mean(og_spectra)),
                "log_amplitude": jnp.log(jnp.max(og_spectra) - jnp.min(og_spectra)),
            },
        )

        loss_opt = partial(loss, kern_type=self._internal_configs["GP_KERNEL"])

        if self._internal_configs["POSTERIOR_CHARACTERIZATION"] == "minimize":
            solver = jaxopt.ScipyMinimize(
                fun=loss_opt,
                options={"maxiter": self._internal_configs["OPTIMIZATION_MAX_ITER"], "disp": True},
                method="BFGS",
                tol=1e-10,
            )
            # print(initial_guess, "\n --..-- \n", data_dict)
            soln = solver.run(jax.tree_map(jnp.asarray, initial_guess), **data_dict)
            # print(soln)
            solution_array = [soln.params[name] for name in param_names]
            # [print(np.exp(i)) for i in solution_array]

        elif self._internal_configs["POSTERIOR_CHARACTERIZATON"] == "MCMC":
            raise NotImplementedError("MCMC characterization is not supported")

        # TODO: evaluate convergence and raise warnings/errors if needed

        return solution_array, result_flag


def generate_kernel(amplitude, length_scale, kern_type):
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
    if kern_type == "Matern-5_2":
        cov_structure = tinygp.kernels.quasisep.Matern52
    elif kern_type == "Matern-3_2":
        cov_structure = tinygp.kernels.quasisep.Matern32
    else:
        raise Exception
    return amplitude * cov_structure(scale=length_scale)
    # Building the tinyGP model for minimization


def build_gp(params, XX_data, YY_variance, kern_type):
    kernel = generate_kernel(
        amplitude=jnp.exp(params["log_amplitude"]),
        length_scale=jnp.exp(params["log_scale"]),
        kern_type=kern_type,
    )

    return tinygp.GaussianProcess(
        kernel,
        XX_data,
        diag=YY_variance,
        mean=jnp.exp(params["log_mean"]),
    )


if not MISSING_TINYGP:

    @partial(jax.jit, static_argnames=("kern_type",))
    def loss(params, XX_data, YY_data, YY_variance, kern_type):
        # TODO: understand if we can pass args to this function!
        gp = build_gp(params, XX_data, YY_variance, kern_type)
        return -gp.log_probability(YY_data)
