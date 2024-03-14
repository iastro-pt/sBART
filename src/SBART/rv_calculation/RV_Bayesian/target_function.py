"""

"""
import numpy as np

from SBART.utils import second_term
from SBART.utils.RV_utilities import ensure_valid_RV
from SBART.utils.RV_utilities.continuum_fit import match_continuum_levels
from SBART.utils.math_tools.build_polynomial import evaluate_polynomial
from SBART.utils.shift_spectra import apply_RVshift


def SBART_target(params, **kwargs):
    # for the singleRV method, we use a different minimizer!
    if isinstance(params, (int, float)):
        RV_shift = params
    else:
        RV_shift = params[0]
    ensure_valid_RV(RV_shift, kwargs["Effective_RV_Limits"])
    # RV shift comes in meter second, covert to km/s
    RV_shift = RV_shift / 1000
    if kwargs["include_jitter"]:
        jitter = params[1]
    else:
        jitter = 0

    squared_jitter = jitter**2

    if kwargs["chromatic_trend"] != "none":
        # if the jitter is not included, the chromatic polynomial starts at index 1
        # Otherwise, it starts at index 2
        poly_params = params[1 + kwargs["include_jitter"] :]
    else:
        poly_params = [0]

    # interpolate template to spectra wavelengths
    if kwargs["chromatic_trend"] == "PixelWise":
        # TODO: apply the PixelWise computation here
        polynomial_contribution = 0
        print("there is no PixelWise trend!!!!!!!!!!!!")
        pass
    elif kwargs["chromatic_trend"] == "OrderWise":
        polynomial_contribution = evaluate_polynomial(poly_params, central_wavelength)
    else:
        polynomial_contribution = 0

    trimmed_template = kwargs["template_wave"]
    StellarTemplate = kwargs["StellarTemplate"]

    wave_spectra_starframe = apply_RVshift(trimmed_template, RV_shift)

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
        shift_RV_by=RV_shift,
        new_wavelengths=current_wavelength[indexes],
        include_invalid=False,
    )
    if kwargs["current_order"] == 59 and 0:
        # plt.plot(current_wavelength[indexes], interpolated_template)
        # plt.plot(current_wavelength[indexes], spectra[indexes], color = 'black')
        path = "/home/amiguel/work/automated_runs/ESPRESSO/SBART_OUT_data/GJ54.1/"
        temp_name = kwargs["name"]
        np.savetxt(path + "spectra.npy", np.c_[current_wavelength[indexes], spectra[indexes]])
        np.savetxt(
            path + temp_name + ".npy", np.c_[current_wavelength[indexes], interpolated_template]
        )
        print("Stored data to file!!!", RV_shift)

    ratio_transposed = (
        spectra[indexes] / interpolated_template
    )  # by definition the data is a N*1 matrix. the data is in a 1*N format

    # compute the log marginal likelihood for the order

    m = 2  # rank of H is always 2.
    N = ratio_transposed.size

    data = ratio_transposed

    # error propagation from the template and spectra
    # template not assumed to be noise free

    diag = (
        kwargs["squared_spectra_uncerts"][indexes] + interpol_errors**2 + squared_jitter
    ) / interpolated_template**2

    # Build H matrix
    H = np.ones((2, N))
    H[1, :] = current_wavelength[indexes]

    # Calculate the "A" term, inverse and determinant
    a_00 = np.sum(H[0] ** 2 / diag)
    a_01 = np.sum(H[0] * H[1] / diag)
    a_11 = np.sum(H[1] ** 2 / diag)

    det_A = a_00 * a_11 - a_01**2
    alpha_00 = a_11 / det_A
    alpha_01 = -a_01 / det_A
    alpha_11 = a_00 / det_A
    # Calculate C
    second_value = second_term.matrix_dot(H, diag, data, alpha_00, alpha_01, alpha_11)

    # Build the different terms of the marginal likelihood
    first_term = -0.5 * np.sum(np.square(data) / diag)
    order_value = (
        first_term
        + second_value
        - 0.5 * np.sum(np.log(diag))
        - 0.5 * np.log(det_A)
        - 0.5 * (N - m) * np.log(2 * np.pi)
    )

    weight = 1 if not kwargs["weighted"] else interpolated_template.size

    if kwargs["weighted"] and 0:
        # COmputation of the expected information. Ignore for now !
        expected_info[order] = (
            np.sum(
                0.5
                * (1 + np.log(2 * np.pi * (uncerts_trimmed[indexes] / interpolated_template) ** 2))
            )
            / weight
        )

    if kwargs["compute_metrics"]:
        # Flux model miss-specification
        # Use the expected value for the parameters of the polynomial

        normalized_template, coefs, residuals = match_continuum_levels(
            current_wavelength,
            spectra[indexes],
            interpolated_template,
            indexes,
            continuum_type="paper",
            fit_degree=1,
        )

        if not kwargs.get("SAVE_DISK_SPACE", False):
            misspec_metric = (spectra[indexes] - normalized_template) / np.sqrt(
                kwargs["squared_spectra_uncerts"][indexes] + interpol_errors**2 + squared_jitter
            )
        else:
            misspec_metric = np.asarray([0])

        return -1 * order_value / weight, misspec_metric
    return -1 * order_value / weight
