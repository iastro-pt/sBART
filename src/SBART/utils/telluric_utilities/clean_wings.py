import numpy as np

from SBART.utils.RV_utilities.create_spectral_blocks import build_blocks


def clean_wings_order(
    current_order_wavelengths, telluric_bin_temp, template_base_berv, BERV_window
):
    """Around each telluric feature, remove a window that spans the entire BERV motion over the year.

    Parameters
    ----------
    current_order_wavelengths : np.ndarray
        Wavelengths associated with the template
    telluric_bin_temp : np.ndarray
        Telluric (binary) template
    template_base_berv : float
        BERV value from the template
    BERV_window : tuple/list
        Minimum and Maximum BERV

    Returns
    -------
    np.ndarray
        Binary template with the increased span
    """

    telluric_positions = np.where(telluric_bin_temp == 1)

    blocks = build_blocks(telluric_positions)
    # set 5 points to each side as a telluric point
    c = 299792.458  # km/s

    for BERV_block in blocks:
        start, end = BERV_block[0], BERV_block[-1]

        # removes the previous BERV correction and calculates the edges based on the MAXBERV window!
        # The template was created for the
        lowest_wavelength = (
            current_order_wavelengths[start] * (c - BERV_window) / (c + template_base_berv)
        )
        highest_wavelength = (
            current_order_wavelengths[end] * (c + BERV_window) / (c + template_base_berv)
        )
        telluric_bin_temp[
            np.where(
                np.logical_and(
                    current_order_wavelengths >= lowest_wavelength,
                    current_order_wavelengths <= highest_wavelength,
                )
            )
        ] = 1

    return telluric_bin_temp
