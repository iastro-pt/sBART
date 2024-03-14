import numpy as np


def first_numerical_derivative(wavelengths, flux, uncertainties):
    """
    Computing the numerical first derivative using the np.gradient prescription:
    https://numpy.org/doc/stable/reference/generated/numpy.gradient.html

    Parameters
    ----------
    wavelengths
    flux
    uncertainties

    Returns
    -------

    """
    values = []
    output_errors = []
    size = len(wavelengths)

    for index in range(size):
        if index == 0:
            step = wavelengths[index + 1] - wavelengths[index]
            val = (flux[index + 1] - flux[index]) / step
            prop_err = (uncertainties[index + 1] ** 2 + uncertainties[index] ** 2) / step**2

        elif index == size - 1:
            step = wavelengths[index] - wavelengths[index - 1]
            val = (flux[index] - flux[index - 1]) / step
            prop_err = (uncertainties[index - 1] ** 2 + uncertainties[index] ** 2) / step**2

        else:
            H_front = wavelengths[index + 1] - wavelengths[index]  # h_d
            H_back = wavelengths[index] - wavelengths[index - 1]  # h_s

            norm_const = H_front * H_back * (H_front + H_back)

            val = (
                H_back**2 * flux[index + 1]
                + (H_front**2 - H_back**2) * flux[index]
                - H_front**2 * flux[index - 1]
            )
            val = val / norm_const

            prop_err = (
                (H_back**2 * uncertainties[index + 1]) ** 2
                + ((H_front**2 - H_back**2) * (uncertainties[index])) ** 2
                + (H_front**2 * uncertainties[index - 1]) ** 2
            )

            # squared so that we can get the correct value after the sqrt at the end!
            prop_err /= norm_const**2

        values.append(val)
        output_errors.append(np.sqrt(prop_err))

    return values, output_errors


def compute_non_uni_step_first_derivative(wavelengths, flux, uncertainties):
    """
    Compute the numerical first derivative for a grid of non-constant step sizes

    Derivative and uncertainty are computed with the framework of
    https://www.tandfonline.com/doi/pdf/10.3402/tellusa.v22i1.10155?needAccess=true
    """
    deriva = []
    error_deriva = []
    steps = np.diff(wavelengths)

    for index in range(1, len(flux) - 1):
        step_ratio = steps[index] / steps[index - 1]
        deriva.append(
            (flux[index + 1] - flux[index - 1] * step_ratio**2 - (1 - step_ratio**2) * flux[index])
            / (steps[index] * (1 + step_ratio))
        )
        error_deriva.append(
            np.sqrt(
                uncertainties[index + 1] ** 2
                + (uncertainties[index - 1] * step_ratio**2) ** 2
                + (uncertainties[index] - uncertainties[index] * step_ratio**2) ** 2
            )
            / (steps[index] * (1 + step_ratio))
        )

    deriva = np.asarray(deriva)
    return deriva, error_deriva
