import numpy as np

from SBART.utils.shift_spectra import apply_RVshift
from .create_spectral_blocks import build_blocks


def find_wavelength_limits(
    current_order_mask,
    spectra_order_waves,
    template_order_mask,
    template_order_wavelengths,
    lower_limit,
    upper_limit,
    min_block_size,
    order=None,
):
    """
    Find the actual wavelength limits
    """
    current_order_wavelength = spectra_order_waves

    blocks = []  # storing the first and last wavelength value of each block, to cut edges afterwards

    template_telluric_free_blocks = build_blocks(np.where(template_order_mask))
    for block in template_telluric_free_blocks:  # account for gaps in the template
        if (
            len(block) < min_block_size
        ):  # random number for now, only want to compare "large" portions of the template
            continue
        blocks.append(
            (
                template_order_wavelengths[block[0]],
                template_order_wavelengths[block[-1]],
            )
        )
    valid_wave_positions = np.zeros(current_order_wavelength.shape, dtype=bool)
    for wavelengths_block in blocks:
        # calculates the common wavelengths, for all RV shifts
        # first value: highest initial wavelenngth
        # last_value: smallest final wavelength
        first_value = apply_RVshift(wavelengths_block[0], upper_limit)
        last_value = apply_RVshift(wavelengths_block[-1], lower_limit)

        wavelengths_limits = np.where(
            np.logical_and(spectra_order_waves >= first_value, spectra_order_waves <= last_value)
        )

        if len(wavelengths_limits[0]) < min_block_size:
            continue
        valid_wave_positions[wavelengths_limits] = True

    # Find common "valid" points to the two masks!
    wavelengths_limits = np.logical_and(valid_wave_positions, current_order_mask)

    # Find blocks of min_block_size to compute RVs
    blocks = build_blocks(np.where(wavelengths_limits))
    for block in blocks:
        if len(block) < min_block_size:
            wavelengths_limits[block] = False
    return wavelengths_limits
