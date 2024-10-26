from curses.ascii import SP

import numpy as np
import scipy.optimize as sc
from loguru import logger
from matplotlib import pyplot as plt

from SBART.utils.math_tools.numerical_derivatives import (
    compute_non_uni_step_first_derivative,
)
from SBART.utils.shift_spectra import SPEED_OF_LIGHT


def compute_DLW(spec_wave, spec_flux, spec_variance, temp_flux, temp_variance):
    # Compute second derivation of the template

    derivative, errors = compute_non_uni_step_first_derivative(spec_wave, temp_flux, np.sqrt(temp_variance))
    deriv_1 = derivative
    # plt.plot(spec_wave[1:-1], deriv_1)

    derivative, errors = compute_non_uni_step_first_derivative(spec_wave, derivative, errors)
    derivative = np.asarray(derivative)
    errors = np.asarray(errors)

    spec_wave = spec_wave[2:-2]
    spec_flux = spec_flux[2:-2]
    temp_flux = temp_flux[2:-2]
    spec_variance = spec_variance[2:-2]
    temp_variance = temp_variance[2:-2]

    # derivative *= spec_wave**2
    weights = np.divide(1, errors**2 + spec_variance)

    squared_derivative = derivative**2
    squared_derivative_errors = errors**2
    residuals = spec_flux - temp_flux

    # plt.plot(spec_wave, residuals, color="black")
    # plt.plot(spec_wave, derivative)

    squared_residuals = residuals**2
    squared_weights = weights**2
    A = np.dot(weights * derivative, residuals)
    B = np.dot(weights, squared_derivative)

    res_variance = temp_variance + spec_variance
    sigma_A = np.sum(
        squared_weights
        * squared_derivative
        * squared_residuals
        * (squared_derivative_errors / squared_derivative + res_variance / squared_residuals)
    )
    sigma_B = np.sum(squared_weights * (2 * derivative * errors) ** 2)

    dlw = A / B
    cov_dlw = dlw**2 * (sigma_A / A**2 + sigma_B / B**2)

    if cov_dlw < 0:
        logger.warning("Covariance of the DLW value is negative. Changing it for nan")
        err_dlw = np.nan
    else:
        err_dlw = np.sqrt(cov_dlw)

    def scale_fit(_, val):
        return val * derivative

    # print("og", dlw)
    # dlw = sc.curve_fit(
    #     scale_fit, xdata=spec_wave, ydata=residuals, sigma=1 / weights, p0=1
    # )[0]
    # plt.scatter(spec_wave, scale_fit(None, dlw))
    # plt.show()

    # print("new", dlw)
    return dlw, err_dlw


if __name__ == "__main__":
    print("oasidoas")

    def normal_f(x, mu, sigma):
        """Gaussiana clássica"""

        dentro = (x - mu) / sigma
        princip = np.exp(-0.5 * dentro**2)

        return np.asarray((princip / (sigma * (2 * np.pi) ** 0.5)))

    spec = np.linspace(6000, 6500, 2000)
    print(spec.size)

    sigma = 10
    d_sigma = 0.1
    amp = 100
    offset = 10000
    fig, axis = plt.subplots()
    for d_sigma in np.linspace(-0.5, 0.5, 50):
        gauss = offset - amp * normal_f(spec, np.mean(spec), sigma)
        gauss_mod = offset - amp * normal_f(spec, np.mean(spec), sigma + d_sigma)
        dlw, err = compute_DLW(
            spec,
            spec_flux=gauss,
            spec_variance=(0.05 * spec) ** 2,
            temp_flux=gauss_mod,
            temp_variance=(0.01 * gauss_mod) ** 2,
        )

        injected = sigma * d_sigma
        print("injected", injected, "\n--//--")
        axis.errorbar(
            100 * d_sigma / sigma,
            100 * (dlw - injected) / injected,
            err,
            color="black",
            marker="x",
            ls="",
        )
    axis.set_ylabel("100*(DLW - injected)/injected")
    axis.set_xlabel(r"100*$\Delta \sigma / \sigma$")

    plt.show()
