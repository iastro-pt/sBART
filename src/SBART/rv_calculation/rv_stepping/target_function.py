import numpy as np

from SBART.utils.RV_utilities import ensure_valid_RV, compute_DLW
from SBART.utils.RV_utilities.continuum_fit import match_continuum_levels
from SBART.utils.shift_spectra import apply_RVshift


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

        data_out = {"poly_params": coefs, "DLW": dlw, "DLW_ERR": dlw_err}

        if not kwargs["SAVE_DISK_SPACE"]:
            data_out["flux_division_residuals"] = residuals

        return data_out

    return chi_squared_val
