import numpy as np
from loguru import logger

from SBART.utils.math_tools.numerical_derivatives import first_numerical_derivative
import scipy.optimize as sc


def compute_DLW(spec_wave, spec_flux, spec_variance, temp_flux, temp_variance):
    # Compute second derivation of the template
    derivative, errors = first_numerical_derivative(spec_wave, temp_flux, np.sqrt(temp_variance))
    derivative, errors = first_numerical_derivative(spec_wave, derivative, errors)

    dlw, cov_dlw = sc.curve_fit(lambda x, param: np.multiply(param, derivative), spec_wave, spec_flux - temp_flux, p0=1,
                                sigma=spec_variance + temp_variance
                                )
    if cov_dlw < 0:
        logger.warning("Covariance of the DLW value is negative. Changing it for nan")
        err_dlw = np.nan
    else:
        err_dlw = np.sqrt(cov_dlw)
    return dlw, err_dlw



