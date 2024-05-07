import numpy as np
from scipy.interpolate import CubicSpline

from SBART.utils.math_tools.Cubic_spline import CustomCubicSpline

SPEED_OF_LIGHT = 299792.458


def apply_RVshift(wave: np.ndarray, stellar_RV: float) -> np.ndarray:
    """

    Parameters
    ----------
    wave : np.ndarray
        [description]
    stellar_RV : float
        [description]
    BERV : float, optional
        [description], by default 0

    Returns
    -------
    np.ndarray
        [description]
    """
    return np.multiply(wave, (1 + stellar_RV / SPEED_OF_LIGHT))


def remove_RVshift(wave: np.ndarray, stellar_RV: float, BERV: float = 0) -> np.ndarray:
    """

    Parameters
    ----------
    wave : np.ndarray
        [description]
    stellar_RV : float
        [description]
    BERV : float, optional
        [description], by default 0

    Returns
    -------
    np.ndarray
        [description]
    """
    return wave / (1 + stellar_RV / SPEED_OF_LIGHT)


def apply_BERV_correction(wave, BERV):
    return wave * (1 + BERV / SPEED_OF_LIGHT)


def remove_BERV_correction(wave, BERV):
    return wave / (1 + BERV / SPEED_OF_LIGHT)


def shift_to_spectrograph_frame(wave: np.ndarray, stellar_RV: float, BERV: float) -> np.ndarray:
    """Remove the contributions of the RV from the star and remove the BERV correction

    Parameters
    ----------
    wave : np.ndarray
        [description]
    stellar_RV : float
        [description]
    BERV : float
        [description]

    Returns
    -------
    np.ndarray
        [description]
    """
    return wave * (SPEED_OF_LIGHT + stellar_RV) / (SPEED_OF_LIGHT + BERV)


def interpolate_data(
    original_lambda,
    original_spectrum,
    original_errors,
    new_lambda,
    lower_limit,
    upper_limit,
    propagate_interpol_errors,
    interpol_cores=4,
    indexes=None,
):
    """Interpolate the input data for a new wavelength solutionx

    Parameters
    ----------
    original_lambda : np.ndarray
        Input wavelengths
    original_spectrum : np.ndarray
        Input data
    original_errors : np.ndarray
        INput errors
    new_lambda : np.ndarray
        Wavelenghts in which we are interpolating
    lower_limit : np.float64
        above which new_lambda wavelength we can interpolate
    upper_limit : np.float64
        Below which new_lambda wavelenght we can interpolate
    propagate_interpol_errors : str
        Method of error propagation. Can be either "propagation" for the analytical propagation, "interpolation" to interpolate input errors or "none" to
        avoid all error propagation (i.e. return zeros)
    interpol_cores : int, optional
        Number of cores used in the analitical error propagation, by default 4
    indexes : np.ndarray, optional
        Previously computed indexes of wavelengths to interpolate, by default None

    Returns
    -------
    new_values: np.ndarray
        INterpolated values
    new_errors: np.ndarray
        Interpolated errors
    valid_indexes: np.ndarray
        Indexes, of new_lambda, in which we interpolated

    Raises
    ------
    Exception
        [description]
    """
    if indexes is None:
        valid_indexes = np.where(
            np.logical_and(new_lambda >= lower_limit, new_lambda <= upper_limit)
        )
    else:
        valid_indexes = indexes

    interpol_locs = new_lambda[valid_indexes]

    if propagate_interpol_errors == "propagation":
        # Custom Cubic spline routine!
        CSplineInterpolator = CustomCubicSpline(
            original_lambda,
            original_spectrum,
            original_errors,
            n_threads=interpol_cores,
        )
        new_data, new_errors = CSplineInterpolator.interpolate(interpol_locs)

    elif propagate_interpol_errors in ["interpolation", "none"]:
        CSplineInterpolator = CubicSpline(original_lambda, original_spectrum)
        new_data = CSplineInterpolator(new_lambda[valid_indexes])

        if propagate_interpol_errors == "none":
            new_errors = np.zeros(new_data.shape)
        else:
            CSplineInterpolator = CubicSpline(original_lambda, original_errors)
            new_errors = CSplineInterpolator(new_lambda[valid_indexes])
    else:
        raise Exception(f"{propagate_interpol_errors=} is not a valid choice!")

    return new_data, new_errors, valid_indexes


def apply_wavelength_shift(wave, direction, stellar_RV, BERV):
    """
    We are subtracting the stellar RV for convenience. Instead of working with the redshit we are
    working with blueshit to avoid adding minus everytime we call this function
    """
    c = 299792.458  # km/s
    if direction == "rest":
        beta = (c + BERV) / (c - stellar_RV)
    elif direction == "star":
        beta = (c - stellar_RV) / (c + BERV)
    return beta * wave
