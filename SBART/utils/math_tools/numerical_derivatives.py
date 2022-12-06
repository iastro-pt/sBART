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
            prop_err = (uncertainties[index + 1] ** 2 + uncertainties[index] ** 2) / step ** 2

        elif index == size - 1:
            step = wavelengths[index] - wavelengths[index - 1]
            val = (flux[index] - flux[index - 1]) / step
            prop_err = (uncertainties[index - 1] ** 2 + uncertainties[index] ** 2) / step ** 2

        else:
            H_front = wavelengths[index + 1] - wavelengths[index]  # h_d
            H_back = wavelengths[index] - wavelengths[index - 1]  # h_s

            norm_const = H_front * H_back * (H_front + H_back)

            val = (
                    H_back ** 2 * flux[index + 1]
                    + (H_front ** 2 - H_back ** 2) * flux[index]
                    - H_front ** 2 * flux[index - 1]
            )
            val = val / norm_const

            prop_err = (H_back ** 2 * uncertainties[index + 1]) ** 2 \
                       + ((H_front ** 2 - H_back ** 2) * (uncertainties[index])) ** 2 \
                       + (H_front ** 2 * uncertainties[index - 1]) ** 2

            # squared so that we can get the correct value after the sqrt at the end!
            prop_err /= norm_const**2

        values.append(val)
        output_errors.append(np.sqrt(prop_err))

    return values, output_errors
