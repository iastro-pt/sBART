import numpy as np
from loguru import logger

from SBART.utils.math_tools.numerical_derivatives import first_numerical_derivative
import scipy.optimize as sc


def compute_DLW(spec_wave, spec_flux, spec_variance, temp_flux, temp_variance):
    # Compute second derivation of the template
    derivative, errors = first_numerical_derivative(spec_wave, temp_flux, np.sqrt(temp_variance))
    derivative, errors = first_numerical_derivative(spec_wave, derivative, errors)
    weights = 1 / (errors ** 2 + spec_variance)

    squared_derivative = derivative ** 2
    squared_derivative_errors = errors ** 2
    residuals = spec_flux - temp_flux
    squared_residuals = residuals ** 2
    squared_weights = weights ** 2
    A = np.sum(weights * derivative * residuals)
    B = np.sum(weights * squared_derivative)

    res_variance = temp_variance + spec_variance
    sigma_A = np.sum(squared_weights * squared_derivative * squared_residuals * (squared_derivative_errors/squared_derivative + res_variance/squared_residuals))
    sigma_B = np.sum(4*squared_weights*squared_derivative_errors/squared_derivative)

    dlw = A / B
    cov_dlw = dlw**2 * (sigma_A / A**2 + sigma_B / B**2)

    # dlw, cov_dlw = sc.curve_fit(lambda x, param: np.multiply(param, derivative), spec_wave, spec_flux - temp_flux, p0=1,
    #                             sigma=spec_variance + temp_variance
    #                             )
    if cov_dlw < 0:
        logger.warning("Covariance of the DLW value is negative. Changing it for nan")
        err_dlw = np.nan
    else:
        err_dlw = np.sqrt(cov_dlw)
    return dlw, err_dlw



