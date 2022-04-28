import numpy as np
from loguru import logger
from scipy import optimize


def chosen_trend(x, *poly_coeffs):
    try:
        return np.poly1d(poly_coeffs)(x)
    except:
        logger.warning("Problem in coeefs: {}".format(poly_coeffs))


def fit_continuum_level(spectra_wave, spectra, template, interpolate_wave_indexes, fit_degree: int):

    min_val = np.min(np.abs(template))
    offset = 1e6 if min_val < 1e-3 else 1
    spectral_ratio = (spectra * offset) / (template * offset)

    p0 = fit_degree * [0] + [1]

    xx_data = spectra_wave[interpolate_wave_indexes]
    yy_data = spectral_ratio

    coefs, coefs_cov = optimize.curve_fit(chosen_trend, xx_data, yy_data, p0=p0)

    return coefs, coefs_cov, spectral_ratio, chosen_trend
