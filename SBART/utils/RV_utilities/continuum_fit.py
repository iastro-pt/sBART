from functools import partial

import numpy as np
from loguru import logger
from scipy import optimize
from SBART.utils import custom_exceptions


def polynomial_continuum_model(x, model_coeffs):
    try:
        return np.poly1d(model_coeffs)(x)
    except:
        logger.warning("Problem in coeefs: {}".format(model_coeffs))


def stretch_continuum_nodel(x, model_coeffs, template):
    """
    Implements the model specified by Xavier Dumusque:
        A*Template + B*lambda + C

    Parameters
    =============
    x:
        Wavelength solution in which the model will be evaluated
    model_coeffs:
        Coefficients of the model, where the leftmost entry is A and all other entries are a polynomial
    template:
        The data that will be "scaled up"
    """
    return np.poly1d(model_coeffs[1:])(x) + template * model_coeffs[0]


def fit_continuum_level(
        spectra_wave,
        spectra,
        template,
        interpolate_wave_indexes,
        fit_degree: int,
        continuum_type: str,
):
    min_val = np.min(np.abs(template))
    offset = 1e6 if min_val < 1e-3 else 1
    spectral_ratio = (spectra * offset) / (template * offset)

    p0 = fit_degree * [0] + [1]

    xx_data = spectra_wave[interpolate_wave_indexes]
    yy_data = spectral_ratio

    if continuum_type == "paper":
        coeffs, coefs_cov = optimize.curve_fit(
            polynomial_continuum_model, xx_data, yy_data, p0=p0
        )
        continuum_model = polynomial_continuum_model
    elif continuum_type == "stretch":
        Matrice1 = np.array([template, xx_data, np.ones(len(xx_data))]).transpose()
        res1 = optimize.lsq_linear(Matrice1, spectra, verbose=0)
        coeffs, coefs_cov = res1.x, [0 for _ in res1.x]
        continuum_model = partial(stretch_continuum_nodel, template=template)
    else:
        raise custom_exceptions.InternalError("Passed wrong keyword {} for the continuum type!", continuum_type)
    return coeffs, coefs_cov, spectral_ratio, continuum_model
