import numpy as np

from SBART.utils.choices import DISK_SAVE_MODE
from SBART.utils.RV_utilities import compute_DLW, ensure_valid_RV, build_blocks
from SBART.utils.RV_utilities.continuum_fit import match_continuum_levels
from SBART.utils.shift_spectra import apply_RVshift, SPEED_OF_LIGHT
from scipy.optimize import minimize_scalar
from functools import partial

from ASTRA.template_creation.stellar_templates import Stellar_Template


def target(n_iterations, init_guess, **kwargs):
    """Metric function for the chi-squared minimization.

    Parameters
    ----------
    params
    kwargs

    Returns
    -------

    """
    # compute the RVs
    StellarTemplate: Stellar_Template = kwargs["StellarTemplate"]

    trimmed_template = kwargs["template_wave"]

    current_wavelength = kwargs["spectra_wave"]
    spectra = kwargs["spectra"]

    rv_guess = init_guess[0]
    for iter_num in range(n_iterations):
        wave_spectra_starframe = apply_RVshift(trimmed_template, rv_guess / 1000)

        indexes = np.where(
            np.logical_and(
                current_wavelength >= wave_spectra_starframe[0],
                current_wavelength <= wave_spectra_starframe[-1],
            ),
        )
        (
            interpolated_template,
            interpol_errors,
        ) = StellarTemplate.interpolate_spectrum_to_wavelength(
            order=kwargs["current_order"],
            RV_shift_mode="apply",
            shift_RV_by=rv_guess / 1000,
            new_wavelengths=current_wavelength[indexes],
            include_invalid=False,
        )

        normalized_template, normalized_uncerts, coefs, residuals, cont_model = match_continuum_levels(
            current_wavelength,
            spectra[indexes],
            interpolated_template,
            indexes,
            continuum_type=kwargs["worker_configs"]["CONTINUUM_FIT_TYPE"],
            fit_degree=kwargs["worker_configs"]["CONTINUUM_FIT_POLY_DEGREE"],
            template_uncertainties=interpol_errors,
            get_continuum_model=True,
        )

        raw_wave, raw_temp, raw_e, raw_m = StellarTemplate.get_data_from_spectral_order(
            order=kwargs["current_order"],
        )
        raw_wave = apply_RVshift(raw_wave, rv_guess / 1000)
        norm = cont_model(raw_wave, coefs)
        raw_temp *= norm

        blocks = build_blocks(np.where(~raw_m))
        # Account for gaps in the template, computing the gradient by chunks
        template_grad = np.zeros_like(raw_m, dtype=np.float64)
        template_grad_2 = np.zeros_like(raw_m, dtype=np.float64)
        for block in blocks:
            if len(block) < 4:
                continue
            template_grad[block] = np.gradient(raw_temp[block]) / np.gradient(raw_wave[block])
            template_grad_2[block] = np.gradient(template_grad[block]) / np.gradient(raw_wave[block])

        # # Interpolate the gradient to the wavelength solution of the spectra
        template_grad, template_grad_err = StellarTemplate.interpolation_interface.interpolate_spectrum_to_wavelength(
            og_lambda=raw_wave,
            og_spectra=template_grad,
            og_err=np.ones_like(template_grad),
            new_wavelengths=current_wavelength[indexes],
            order=kwargs["current_order"],
        )
        template_grad_2, template_grad_2_err = (
            StellarTemplate.interpolation_interface.interpolate_spectrum_to_wavelength(
                og_lambda=raw_wave,
                og_spectra=template_grad_2,
                og_err=np.ones_like(template_grad_2),
                new_wavelengths=current_wavelength[indexes],
                order=kwargs["current_order"],
            )
        )

        # template_grad_1 = np.gradient(normalized_template) / np.gradient(current_wavelength)
        final_uncertainties = 1 / (kwargs["squared_spectra_uncerts"][indexes] + normalized_uncerts**2)

        result = minimize_scalar(
            fun=partial(
                _estimate_RV,
                xx=current_wavelength[indexes],
                yy=spectra[indexes],
                template=normalized_template,
                grad=template_grad,
                grad_2=template_grad_2,
                final_uncertainties=final_uncertainties,
                use_extra_term=kwargs["USE_TAYLOR_TERM_2"],
            ),
            # tol=1e-30,
            bounds=[-500, 500],
            options={"maxiter": 50000},
        )
        rv_guess += result.x

        rv_err = 1000 / np.sqrt(
            np.sum(
                1
                / (
                    np.sqrt(kwargs["squared_spectra_uncerts"][indexes])
                    / (template_grad * current_wavelength[indexes] / SPEED_OF_LIGHT)
                )
                ** 2
            )
        )

    if kwargs.get("get_minimum_information", False):
        # This will be triggered when the sampler sends a request to get more information
        # of the different metrics for the optimal RV solution
        try:
            dlw, dlw_err = compute_DLW(
                spec_wave=current_wavelength[indexes],
                spec_flux=spectra[indexes],
                spec_variance=kwargs["squared_spectra_uncerts"][indexes],
                temp_flux=interpolated_template,
                temp_variance=normalized_uncerts**2,
                spectra_binary_mask=indexes,
            )
        except:
            dlw, dlw_err = np.nan, np.nan

        data_out = {
            "poly_params": coefs,
            "DLW": dlw,
            "DLW_ERR": dlw_err,
            # "Pred_RV": pred_velocity * 1000,  # send information in m/s
            # "Pred_RV_precision": pred_velocity_error * 1000,  # send information in m/s
            # "quality": quality,
            # "pix_sum_in_template": pix_sum_in_template,
        }

        if kwargs["SAVE_DISK_SPACE"] == DISK_SAVE_MODE.DISABLED:
            data_out["flux_division_residuals"] = residuals

        return data_out

    return rv_guess, rv_err


def _estimate_RV(RV, xx, yy, template, grad, grad_2, final_uncertainties, use_extra_term):
    RV /= 1000
    if use_extra_term:
        extra_term = 0.5 * grad_2 * (xx**2) / (SPEED_OF_LIGHT**2)
    else:
        extra_term = 0

    flux_res = yy - (template - grad * xx * RV / SPEED_OF_LIGHT - extra_term * RV**2)
    return np.sum(final_uncertainties * (flux_res) ** 2)
