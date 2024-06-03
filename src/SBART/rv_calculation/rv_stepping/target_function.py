import numpy as np

from SBART.utils.RV_utilities import ensure_valid_RV, compute_DLW
from SBART.utils.RV_utilities.continuum_fit import match_continuum_levels
from SBART.utils.shift_spectra import apply_RVshift, SPEED_OF_LIGHT
from SBART.utils.math_tools.numerical_derivatives import first_numerical_derivative


def target(params, **kwargs):
    """
    Metric function for the chi-squared minimization.

    Parameters
    ----------
    params
    kwargs

    Returns
    -------

    """
    # compute the RVs
    StellarTemplate = kwargs["StellarTemplate"]

    trimmed_template = kwargs["template_wave"]

    # RV_step is in m/s, converting to kilometer
    ensure_valid_RV(params, kwargs["Effective_RV_Limits"])

    tentative_RV_shift = params / 1000
    wave_spectra_starframe = apply_RVshift(trimmed_template, tentative_RV_shift)

    current_wavelength = kwargs["spectra_wave"]
    spectra = kwargs["spectra"]

    indexes = np.where(
        np.logical_and(
            current_wavelength >= wave_spectra_starframe[0],
            current_wavelength <= wave_spectra_starframe[-1],
        )
    )

    interpolated_template, interpol_errors = StellarTemplate.interpolate_spectrum_to_wavelength(
        order=kwargs["current_order"],
        RV_shift_mode="apply",
        shift_RV_by=tentative_RV_shift,
        new_wavelengths=current_wavelength[indexes],
        include_invalid=False,
        remove_OB=kwargs["frame"],
    )

    normalized_template, normalized_uncerts, coefs, residuals = match_continuum_levels(
        current_wavelength,
        spectra[indexes],
        interpolated_template,
        indexes,
        continuum_type=kwargs["worker_configs"]["CONTINUUM_FIT_TYPE"],
        fit_degree=kwargs["worker_configs"]["CONTINUUM_FIT_POLY_DEGREE"],
        template_uncertainties=interpol_errors,
    )

    final_uncertainties = 1 / (kwargs["squared_spectra_uncerts"][indexes] + normalized_uncerts**2)

    chi_squared_val = np.sum(final_uncertainties * (spectra[indexes] - normalized_template) ** 2)

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
            )
        except:
            dlw, dlw_err = np.nan, np.nan

        # template_derivative, deriv_error = first_numerical_derivative(
        #     wavelengths=current_wavelength,
        #     flux=normalized_template,
        #     uncertainties=normalized_uncerts,
        # )
        # weights = (current_wavelength * template_derivative) ** 2 / (
        #     kwargs["squared_spectra_uncerts"][indexes] + normalized_uncerts**2
        # )

        # pix_sum_in_template = np.sum(normalized_template)
        # quality = np.sqrt(np.sum(weights)) / np.sqrt(pix_sum_in_template)

        # # In km/s !!!!
        # pred_velocity_error = SPEED_OF_LIGHT / (quality * np.sqrt(pix_sum_in_template))

        # res = spectra[indexes] - normalized_template
        # pred_velocity = (
        #     SPEED_OF_LIGHT
        #     * np.sum(
        #         res
        #         * np.sqrt(
        #             weights / (kwargs["squared_spectra_uncerts"][indexes] + normalized_uncerts**2)
        #         )
        #     )
        #     / np.sum(weights)
        # )

        # if kwargs["current_order"] == 100 and 0:
        #     print(type(pred_velocity))
        #     print(pred_velocity)
        #     import matplotlib.pyplot as plt

        #     fig, axis = plt.subplots(3)
        #     axis[0].scatter(current_wavelength, normalized_template)
        #     axis[1].scatter(current_wavelength, template_derivative)
        #     (
        #         axis[1].scatter(
        #             current_wavelength,
        #             np.gradient(current_wavelength, normalized_template),
        #             color="red",
        #         ),
        #     )

        #     axis[2].scatter(
        #         current_wavelength,
        #         np.gradient(current_wavelength, normalized_template) - template_derivative,
        #         color="red",
        #     )
        #     plt.show()

        data_out = {
            "poly_params": coefs,
            "DLW": dlw,
            "DLW_ERR": dlw_err,
            # "Pred_RV": pred_velocity * 1000,  # send information in m/s
            # "Pred_RV_precision": pred_velocity_error * 1000,  # send information in m/s
            # "quality": quality,
            # "pix_sum_in_template": pix_sum_in_template,
        }

        if not kwargs["SAVE_DISK_SPACE"]:
            data_out["flux_division_residuals"] = residuals

        return data_out

    return chi_squared_val
