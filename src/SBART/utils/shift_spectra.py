"""Functions to shift spectra.

Implements utility functions to correct for stellar RV and BERV
"""

import numpy as np

SPEED_OF_LIGHT = 299792.458


def apply_RVshift(wave: np.ndarray, stellar_RV: float) -> np.ndarray:
    """Apply RV shift to spectra.

    Args:
        wave (np.ndarray): Wavelength array
        stellar_RV (float): RV shift to apply, in km/s

    Returns:
        np.ndarray: New wavelength array, after RV shift

    """
    return np.multiply(wave, (1 + stellar_RV / SPEED_OF_LIGHT))


def remove_RVshift(wave: np.ndarray, stellar_RV: float) -> np.ndarray:
    """Remove RV shift from wavelength vector.

    Args:
        wave (np.ndarray): Wavelength vector
        stellar_RV (float): stellar RV, in km/s

    Returns:
        np.ndarray: New wavelength array

    """
    return wave / (1 + stellar_RV / SPEED_OF_LIGHT)


def apply_approximated_BERV_correction(wave: np.ndarray, BERV: float) -> np.ndarray:
    """Apply the approximated BERV correction that is implemented in ESO DRS 3.2.

    Args:
        wave (np.ndarray): wavelength array
        BERV (float): BERV value, with the 1.55 * 10**-8 approximation

    Returns:
        np.ndarray: BERV-corrected wavelength array

    """
    # Addint the new term that we found in the ESPRESSO DRS. Source of this is unclear.....
    return wave * (1 + 1.55e-8) * (1 + BERV / SPEED_OF_LIGHT)


def remove_approximated_BERV_correction(wave: np.ndarray, BERV: float) -> np.ndarray:
    """Remove the approximated BERV correction implemented in ESO DRS 3.2.

    Args:
        wave (np.ndarray): wavelength array
        BERV (float): BERV value, with the 1.55 * 10**-8 approximation

    Returns:
        np.ndarray: BERV-corrected wavelength array

    """
    return wave / ((1 + 1.55e-8) * (1 + BERV / SPEED_OF_LIGHT))


def apply_BERV_correction(wave: np.ndarray, BERV: float) -> np.ndarray:
    """Apply BERV correction to data.

    Args:
        wave (np.ndarray): wavelength array
        BERV (float): BERV value, with the 1.55 * 10**-8 approximation

    Returns:
        np.ndarray: BERV-corrected wavelength array

    """
    return wave * (1 + BERV / SPEED_OF_LIGHT)


def remove_BERV_correction(wave: np.ndarray, BERV: float) -> np.ndarray:
    """Remove BERV correction to data.

    Args:
        wave (np.ndarray): wavelength array
        BERV (float): BERV value, with the 1.55 * 10**-8 approximation

    Returns:
        np.ndarray: BERV-corrected wavelength array

    """
    return wave / (1 + BERV / SPEED_OF_LIGHT)
