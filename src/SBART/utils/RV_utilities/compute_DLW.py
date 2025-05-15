import numpy as np
from loguru import logger
from matplotlib import pyplot as plt

from SBART.utils.math_tools.numerical_derivatives import (
    compute_finite_differences_spectral_derivatives,
    compute_non_uni_step_first_derivative,
)
from SBART.utils.RV_utilities.create_spectral_blocks import build_blocks
from SBART.utils.shift_spectra import SPEED_OF_LIGHT


def compute_DLW(
    spec_wave,
    spec_flux,
    spec_variance,
    temp_flux,
    temp_variance,
    spectra_binary_mask,
):
    blocks = build_blocks(spectra_binary_mask)

    A = 0
    B = 0
    sigma_A = 0
    sigma_B = 0
    for block in blocks:
        # Iterate over blocks of "consecutive" points (i.e., no gaps in the data)

        # TODO: properly compute uncertainty
        first, derivative = compute_finite_differences_spectral_derivatives(spec_wave[block], temp_flux[block])

        # Skip over the first/last two points to avoid numerical instability
        derivative = derivative[2:-2]
        derivative_errors = spec_variance[block][2:-2]

        # derivative, errors = compute_non_uni_step_first_derivative(spec_wave, temp_flux, np.sqrt(temp_variance))
        # deriv_1 = derivative
        # # plt.plot(spec_wave[1:-1], deriv_1)

        # derivative, errors = compute_non_uni_step_first_derivative(spec_wave, derivative, errors)
        # derivative = np.asarray(derivative)
        # derivative_errors = np.asarray(errors)

        spec_wave = spec_wave[block][2:-2]
        spec_flux = spec_flux[block][2:-2]
        temp_flux = temp_flux[block][2:-2]
        spec_variance = spec_variance[block][2:-2]
        temp_variance = temp_variance[block][2:-2]

        weights = np.divide(1, spec_variance)

        squared_derivative = derivative**2
        squared_derivative_errors = derivative_errors**2
        residuals = spec_flux - temp_flux

        # plt.plot(spec_wave, residuals, color="black")
        # plt.plot(spec_wave, derivative)

        squared_residuals = residuals**2
        squared_weights = weights**2

        A += np.dot(weights * derivative, residuals)
        B += np.dot(weights, squared_derivative)

        res_variance = temp_variance + spec_variance
        sigma_A += np.sum(
            squared_weights
            * squared_derivative
            * squared_residuals
            * (squared_derivative_errors / squared_derivative + res_variance / squared_residuals),
        )
        sigma_B += np.sum(squared_weights * (2 * derivative * derivative_errors) ** 2)

    dlw = -1 * A / B
    cov_dlw = dlw**2 * (sigma_A / A**2 + sigma_B / B**2)

    if cov_dlw < 0:
        logger.warning("Covariance of the DLW value is negative. Changing it for nan")
        err_dlw = np.nan
    else:
        err_dlw = np.sqrt(cov_dlw)

    # print("og", dlw)
    # dlw = sc.curve_fit(
    #     scale_fit, xdata=spec_wave, ydata=residuals, sigma=1 / weights, p0=1
    # )[0]
    # plt.scatter(spec_wave, scale_fit(None, dlw))
    # plt.show()

    # print("new", dlw)
    return dlw, err_dlw


if __name__ == "__main__":

    def normal_f(x, mu, sigma):
        """Gaussiana clássica"""
        dentro = (x - mu) / sigma
        princip = np.exp(-0.5 * dentro**2)

        return np.asarray(princip / (sigma * (2 * np.pi) ** 0.5))

    spec = np.linspace(6000, 6020, 2000)
    center = np.median(spec)

    print(spec.size)
    sigma = 6 * center / SPEED_OF_LIGHT  # FWHM of 7km/s
    max_jump = 10 / 1000 * center / SPEED_OF_LIGHT  # Delta FWHM of 0.1 km/s
    amp = 1
    offset = 10

    fig, axis = plt.subplots()
    for d_sigma in np.linspace(-max_jump, max_jump, 100):
        gauss = offset - amp * normal_f(spec, center, sigma)
        gauss_mod = offset - amp * normal_f(spec, center, sigma + d_sigma)

        dlw, err = compute_DLW(
            spec,
            spec_flux=gauss / np.median(gauss),
            spec_variance=(np.ones_like(spec) * offset) ** 0.5,
            temp_flux=gauss_mod / np.median(gauss_mod),
            temp_variance=(np.ones_like(spec) * offset / 2) ** 0.5,
            spectra_binary_mask=np.where(np.ones_like(spec, dtype=bool)),
        )

        expected = sigma * d_sigma
        axis.errorbar(
            d_sigma,
            dlw / d_sigma,
            err,
            color="black",
            marker="x",
            ls="",
        )

    axis.set_ylabel("100*(DLW - injected)/injected")
    axis.set_xlabel(r"100*$\Delta \sigma / \sigma$")

    plt.show()
