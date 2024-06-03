import numpy as np


def roll_array(array, shift_amount):
    """

    Shift the array "array" by "shift_amount". The values that roll to the other edge of the array are replaced by
    numpy.nan.

    Used for the telluric template alignment

    Parameters
    ----------
    array : [type]
        Array to be shifted
    shift_amount : int
        Number of indexes to shift

    Returns
    -------
    Shifted template
    """
    if shift_amount == 0:
        return array
    shifted_array = np.roll(array, shift_amount)

    if shift_amount < 0:
        shifted_array[shift_amount:] = np.nan
    else:
        shifted_array[: shift_amount + 1] = np.nan

    return shifted_array
