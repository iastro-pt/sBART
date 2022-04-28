import matplotlib.pyplot as plt
import numpy as np

from SBART.utils.RV_utilities import ensure_valid_RV
from SBART.utils.RV_utilities.continuum_fit import fit_continuum_level
from SBART.utils.shift_spectra import apply_RVshift, interpolate_data


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
    trimmed_template = kwargs["template_wave"]
    template = kwargs["template"]

    # RV_step is in m/s, converting to kilometer
    ensure_valid_RV(params, kwargs["Effective_RV_Limits"])

    tentative_RV_shift = params / 1000
    wave_spectra_starframe = apply_RVshift(trimmed_template, tentative_RV_shift)

    current_wavelength = kwargs["spectra_wave"]
    spectra = kwargs["spectra"]
    interpolated_template, interpol_errors, indexes = interpolate_data(
        original_lambda=wave_spectra_starframe,
        original_spectrum=template,
        original_errors=kwargs["template_uncerts"],
        new_lambda=current_wavelength,
        lower_limit=wave_spectra_starframe[0],
        upper_limit=wave_spectra_starframe[-1],
        propagate_interpol_errors=kwargs["interpol_prop_type"],
        interpol_cores=kwargs["N_cores_propagation"],
    )

    # not entirely sure, but the spectra and the wavelengths go with different formats.....
    coefs, _, residuals, chosen_trend = fit_continuum_level(
        current_wavelength,
        spectra[indexes],
        interpolated_template,
        indexes,
        fit_degree=kwargs["worker_configs"]["CONTINUUM_FIT_POLY_DEGREE"],
    )

    normalizer = chosen_trend(current_wavelength[indexes], *coefs)

    if kwargs["make_plot"]:
        plt.plot(
            apply_RVshift(trimmed_template, kwargs["Effective_RV_Limits"][0] / 1000),
            template,
            color="blue",
        )
        plt.plot(
            apply_RVshift(trimmed_template, kwargs["Effective_RV_Limits"][1] / 1000),
            template,
            color="red",
        )

        plt.plot(current_wavelength, spectra, color="black")
        plt.show()
        print("here!!")
        plt.figure()
        plt.plot(
            current_wavelength[indexes],
            spectra[indexes] / interpolated_template,
            color="red",
        )
        plt.plot(current_wavelength[indexes], normalizer, color="black")
        final_uncertainties = 1 / (
            kwargs["squared_spectra_uncerts"][indexes] + interpol_errors ** 2
        )

        plt.figure()
        plt.plot(current_wavelength[indexes], np.sqrt(1 / final_uncertainties))
        if 1:
            fig, axis = plt.subplots(2, 1, sharex=True)
            plt.title("Normal comparison")
            axis[1].plot(
                current_wavelength[indexes],
                final_uncertainties * (spectra[indexes] - normalizer * interpolated_template) ** 2,
            )
            axis[0].plot(current_wavelength[indexes], spectra[indexes], color="black")
            axis[0].plot(wave_spectra_starframe, template, color="blue")
            axis[0].plot(
                current_wavelength[indexes],
                normalizer * interpolated_template,
                color="red",
            )

        # plt.plot(current_wavelength, spectra, color = 'black')
        plt.show()
    final_uncertainties = 1 / (kwargs["squared_spectra_uncerts"][indexes] + interpol_errors ** 2)

    chi_squared_val = np.sum(
        final_uncertainties * (spectra[indexes] - normalizer * interpolated_template) ** 2
    )

    if kwargs.get("get_minimum_information", False):
        data_out = {
            "poly_params": coefs,
            "flux_division_residuals": residuals,
        }
        return data_out

    return chi_squared_val
