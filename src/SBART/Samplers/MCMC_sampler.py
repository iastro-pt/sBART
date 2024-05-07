"""
Implements posterior characterization with MCMC algorithm
"""

from typing import Any, Dict, List, Optional, Tuple

import emcee
import numpy as np
from loguru import logger

from SBART.utils import custom_exceptions, meter_second, status_codes
from SBART.utils.math_tools import check_variation_inside_interval
from SBART.utils.status_codes import SUCCESS, Flag, WARNING
from SBART.utils.UserConfigs import DefaultValues, NumericValue, UserParam
from SBART.utils.work_packages import Package

from .SbartBaseSampler import SbartBaseSampler


def log_prior(theta, param_limits):
    """
    Uniform priors, with the limits passed by the Sampler
    Parameters
    ----------
    theta
    param_limits

    Returns
    -------

    """
    for entry_index, param_value in enumerate(theta):
        if param_limits[entry_index][0] > param_value or param_value > param_limits[entry_index][1]:
            return -np.inf
    return 0.0


def log_probability(theta, RV_limits, internal_function, target_args):
    lp = log_prior(theta, RV_limits)
    if not np.isfinite(lp):
        return -np.inf

    # tge likelihood returns the negative
    return lp + -1 * internal_function(theta, *target_args)


def estimate_RV_from_chains(sampler, burn_in: int, mean_list: List[float], std_list: List[float]):
    chains = sampler.get_chain(discard=burn_in, flat=True)
    mean, std = np.mean(chains), np.std(chains)
    mean_list.append(mean)
    std_list.append(std)
    return mean_list, std_list


class MCMC_sampler(SbartBaseSampler):
    """
    Explore the semi-Bayesian model posterior distribution with an MCMC routine (using emcee)

    **User parameters:**

    ================================ ================ ================ ================ ================
    Parameter name                      Mandatory      Default Value    Valid Values    Comment
    ================================ ================ ================ ================ ================
    MAX_ITERATIONS                      False           1000            Positive int     Maximum number of iterations for MCMC
    ensemble_moves                      False           None            ------           emcee Ensemble moves
    N_walkers                           False           4               Positive int     Number of walkers
    ================================ ================ ================ ================ ================

    *Note:* Also check the **User parameters** of the parent classes for further customization options of SBART
    """

    _name = "MCMC"

    _default_params = SbartBaseSampler._default_params + DefaultValues(
        MAX_ITERATIONS=UserParam(1000, constraint=NumericValue),
        ensemble_moves=UserParam(None),  # nwalkers=3, ensemble_moves=emcee.moves.GaussianMove(0.1)
        N_walkers=UserParam(4, constraint=NumericValue),
    )

    def __init__(self, RV_step, rv_prior: list, user_configs: Optional[Dict[str, Any]] = None):
        """
        Explore the posterior distribution with MCMC
        """
        super().__init__(
            mode="epoch-wise",
            RV_step=RV_step,
            RV_window=rv_prior,
            user_configs=user_configs,
            sampler_folders={
                "chains": "chains",
                "metrics": "metrics",  # TODO: store things in the metrics folder!
            },
        )

        self._ndim = 1

        self._RV_var_oscillation_criteria = 0.02

    ###
    #
    ###

    def optimize(self, target, target_kwargs: dict) -> Tuple[Package, Flag]:
        out_pkg = Package(
            ("RV", "RV_uncertainty", "autocorr_evolution", "RV_evolution", "RV_ERR_evolution")
        )

        params_to_use = self.model_params.get_enabled_params()
        ndim = len(params_to_use)

        initial_guesses, bounds = self.model_params.generate_optimizer_inputs(
            frameID=target_kwargs["run_information"]["frameID"], rv_units=meter_second
        )

        # TODO: validate this
        starting_pos = [initial_guesses] + 0.1 * np.random.randn(
            self._internal_configs["N_walkers"], ndim
        )

        if self.mode == "order-wise":
            internal_func = self.apply_orderwise
        elif self.mode == "epoch-wise":
            logger.debug("Initial guesses: {}", initial_guesses)
            logger.debug("Param bounds: {}", bounds)
            internal_func = self.apply_epochwise
            out_pkg["frameID"] = target_kwargs["run_information"]["frameID"]
        else:
            raise custom_exceptions.InvalidConfiguration(
                "Sampler mode <> does not exist", self.mode
            )
        args = (target, target_kwargs) if self.mode == "order-wise" else (target_kwargs,)

        sampler = emcee.EnsembleSampler(
            self._internal_configs["N_walkers"],
            ndim,
            log_probability,
            moves=[(self._internal_configs["ensemble_moves"], 1)],
            args=([bounds, internal_func, args]),
        )

        sampler, order_status, out_pkg, header_info = self.apply_MCMC(
            sampler=sampler, starting_pos=starting_pos, output_pkg=out_pkg
        )

        self.store_metrics(sampler=sampler, target_KWARGS=target_kwargs, header_info=header_info)

        if self.mode == "epoch-wise":
            target_kwargs["run_information"]["target_specific_configs"]["compute_metrics"] = True
            target_kwargs["run_information"]["target_specific_configs"]["weighted"] = True
            model_misspec, log_likelihood, orders = internal_func(
                out_pkg["RV"].value, target_kwargs
            )

        else:
            target_kwargs["compute_metrics"] = True
            target_kwargs["weighted"] = True
            _, model_misspec = internal_func(out_pkg["RV"].value, target, target_kwargs)

        out_pkg["FluxModel_misspecification"] = model_misspec

        return out_pkg, order_status

    def apply_MCMC(self, sampler, starting_pos, output_pkg: Package):
        posterior_mean = []
        posterior_std = []
        burn_in = 0
        mean = 0
        std = 0
        autocorr = 0
        MCMC_status = SUCCESS

        header_info = {}
        autocorrelation_evolution = [np.inf]

        RV_converged = False
        BurnIn_converged = False
        reject_obs = False

        for _ in sampler.sample(
            starting_pos, iterations=self._internal_configs["MAX_ITERATIONS"], progress=False
        ):
            # 40 initial samples to have a more robust autocorrelation time estimate
            if sampler.iteration < 50 or sampler.iteration % 10 != 0:
                continue
            if not BurnIn_converged:  # searching for stable estimation of autocorrelation time
                autocorr = sampler.get_autocorr_time(tol=0)

                # Variation in relation to previous measurement
                autocorr_var = (autocorrelation_evolution[-1] - autocorr) / autocorr

                # search for 1% change and 50 autocorr samples
                if np.abs(autocorr_var) < 0.01 and sampler.iteration > 50 * autocorr:
                    burn_in = int(10 * autocorr)
                    posterior_mean, posterior_std = estimate_RV_from_chains(
                        sampler=sampler,
                        burn_in=burn_in,
                        mean_list=posterior_mean,
                        std_list=posterior_std,
                    )
                    BurnIn_converged = True

                autocorrelation_evolution.append(autocorr)

            else:
                posterior_mean, posterior_std = estimate_RV_from_chains(
                    sampler=sampler,
                    burn_in=burn_in,
                    mean_list=posterior_mean,
                    std_list=posterior_std,
                )

                logger.info(
                    f"\tEvaluating state of posterior - iter : {sampler.iteration} - mean: {mean} - std : {std}"
                )

                # find oscillations smaller than 2% in regard to last estimate
                if check_variation_inside_interval(
                    posterior_mean[-2], posterior_mean[-1], self._RV_var_oscillation_criteria
                ) and check_variation_inside_interval(
                    posterior_std[-2], posterior_std[-1], self._RV_var_oscillation_criteria
                ):
                    RV_converged = True
                    break

        # This will store the actual burn-in used for the RV estimation
        # It can change if the BURN-IN itself has not found, leading to not searching for RV
        eff_BurnIN = burn_in

        if not RV_converged:
            if not BurnIn_converged:
                eff_BurnIN = int(10 * autocorr)
                logger.warning(
                    "Failed to achieve burn-in convergence. Using burn_in equal to 10X last autocorr: {}",
                    eff_BurnIN,
                )

                if eff_BurnIN > 0.8 * self._internal_configs["MAX_ITERATIONS"]:
                    logger.critical(
                        "Less than 20% of samples available after discarding burn-in. Rejecting OBS"
                    )
                    reject_obs = True

                if not reject_obs:
                    posterior_mean, posterior_std = estimate_RV_from_chains(
                        sampler=sampler,
                        burn_in=eff_BurnIN,
                        mean_list=posterior_mean,
                        std_list=posterior_std,
                    )
                    MCMC_status = WARNING("MCMC did not converge!")
                    # TODO: raise warning in the RV txt output!

        if reject_obs:
            MCMC_status = status_codes.CONVERGENCE_FAIL
            RV = np.nan * meter_second
            uncert = np.nan * meter_second
        else:
            RV = posterior_mean[-1] * meter_second
            uncert = posterior_std[-1] * meter_second

        header_info["RV_converged"] = RV_converged
        header_info["Burn-In converged"] = BurnIn_converged
        header_info["Exploration Burn-in"] = burn_in
        header_info["Effective Burn-in"] = eff_BurnIN
        header_info["N_iter"] = sampler.iteration

        output_pkg["RV_evolution"] = posterior_mean
        output_pkg["RV_ERR_evolution"] = posterior_std

        output_pkg["status"] = MCMC_status

        output_pkg["RV"] = RV
        output_pkg["RV_uncertainty"] = uncert

        return sampler, MCMC_status, output_pkg, header_info

    def store_metrics(self, sampler, target_KWARGS: dict, header_info: Dict[str, Any]):
        ###
        #   Build header
        ###
        frameID = target_KWARGS["run_information"]["frameID"]
        rel_path = "individual_subInst" if not self.is_merged_subInst else "merged_subInst"
        base_path = (
            self._internalPaths.root_storage_path
            / rel_path
            / target_KWARGS["run_information"]["subInst"]
        )
        header = ["General information:"]
        for header_KW, KW_val in header_info.items():
            header.append(f"\n\t{header_KW} : {KW_val}")
        header.append("\nChains:\n")
        if self.mode == "order-wise":
            order = target_KWARGS["current_order"]
            fname = base_path / f"frame_{frameID}__order_{order}.txt"
        else:
            fname = base_path / f"frame_{frameID}.txt"

        np.savetxt(
            fname=fname.as_posix(),
            X=sampler.get_chain(discard=0, flat=True),
            header="".join(header),
        )
