from functools import partial

import numpy as np
from loguru import logger
from scipy import optimize
from SBART.utils import custom_exceptions


def polynomial_continuum_model(x, *model_coeffs):
    try:
        return np.poly1d(model_coeffs)(x)
    except:
        logger.warning("Problem in coeefs: {}".format(model_coeffs))


def stretch_continuum_nodel(x, template, *model_coeffs):
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

    p0 = fit_degree * [0] + [1]

    xx_data = spectra_wave[interpolate_wave_indexes]

    if continuum_type == "paper":
        spectral_ratio = (spectra * offset) / (template * offset)
        yy_data = spectral_ratio
        coeffs, coefs_cov = optimize.curve_fit(polynomial_continuum_model, xx_data, yy_data, p0=p0)
        continuum_model = lambda x, coeffs: polynomial_continuum_model(x, *coeffs)
    elif continuum_type == "stretch":
        Matrice1 = np.array([template, xx_data, np.ones(len(xx_data))]).transpose()
        spectral_ratio = 0
        res1 = optimize.lsq_linear(Matrice1, spectra, verbose=0)
        coeffs, coefs_cov = res1.x, [0 for _ in res1.x]
        continuum_model = lambda x, coeffs: stretch_continuum_nodel(x, template, *coeffs)
    else:
        raise custom_exceptions.InternalError(
            "Passed wrong keyword {} for the continuum type!", continuum_type
        )
    return coeffs, coefs_cov, spectral_ratio, continuum_model


def match_continuum_levels(
    spectra_wave,
    spectra,
    template,
    interpolate_wave_indexes,
    fit_degree: int,
    continuum_type: str,
    template_uncertainties=None,
):
    """
    Match the continuum level of the template to that of the provided observation.
    If the template uncertainties are passed, it will also update them, accounting for the
    normalization step!

    Parameters
    ----------
    spectra_wave
    spectra
    template
    interpolate_wave_indexes
    fit_degree
    continuum_type
    template_uncertainties

    Returns
    -------

    """
    coeffs, cov, spec_ratio, cont_model = fit_continuum_level(
        spectra_wave,
        spectra=spectra,
        template=template,
        interpolate_wave_indexes=interpolate_wave_indexes,
        fit_degree=fit_degree,
        continuum_type=continuum_type,
    )

    continuum_model = cont_model(spectra_wave[interpolate_wave_indexes], coeffs)

    if continuum_type == "paper":
        normalized_template = continuum_model * template

    elif continuum_type == "stretch":
        normalized_template = continuum_model

    if template_uncertainties is not None and continuum_type == "paper":
        normalized_uncertainties = continuum_model * template_uncertainties
        return normalized_template, normalized_uncertainties, coeffs, spec_ratio
    return normalized_template, coeffs, spec_ratio
