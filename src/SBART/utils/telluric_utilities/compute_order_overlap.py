from typing import Iterable

import numpy as np


def compute_wavelength_order_overlap(
    wavelength_array: np.ndarray, region_of_interest: Iterable[int]
):
    """
    Find the orders in which there is a wavelength overlap
    Parameters
    ----------
    wavelength_array:
        wavelengths in which we want to search for overlaps
    region_of_interest:
        List with start and end of the wavelength region of interest


    Returns
    -------
    orders: Set of the orders in which there is an overlap
    """

    indexes = np.where(
        np.logical_and(
            wavelength_array >= region_of_interest[0], wavelength_array <= region_of_interest[1]
        )
    )

    return set(indexes[0])


if __name__ == "__main__":
    waves = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9], [5, 6, 7, 8, 9, 10, 11, 12, 13]])
    inte = [100, 800]
    print(compute_wavelength_order_overlap(waves, inte))

    from astropy.io import fits

    with fits.open(
        "/home/amiguel/seminar/data_files/ESPRESSO/TauCet/data/r.ESPRE.2021-10-03T06:13:37.519_S2D_BLAZE_A.fits"
    ) as hdu:
        wave = hdu[4].data
    print(compute_wavelength_order_overlap(wave, [7178, 7313]))

    import timeit

    print(
        "Time per iteration: ",
        timeit.timeit(
            "compute_wavelength_order_overlap(wave, [7178, 7313])", number=10, globals=globals()
        ),
    )
