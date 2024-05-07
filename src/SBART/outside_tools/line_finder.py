from pathlib import Path
from typing import Optional, Iterable

import numpy as np
from matplotlib import pyplot as plt

from SBART.Base_Models import Frame
from SBART.utils import build_blocks
from SBART.utils.math_tools.numerical_derivatives import first_numerical_derivative


def find_lines_indexes(
    frame: Frame, skip_orders: Optional[Iterable] = None, include_invalid: bool = False
):
    """
    Find regions where the spectrum is in a line
    Parameters
    ----------
    frame

    Returns
    -------

    """
    is_line = np.zeros(frame.array_size, dtype=bool)
    make_plot = False
    if make_plot:
        plt.switch_backend("TkAgg")

    for order in range(frame.N_orders):
        if skip_orders is not None and order in skip_orders:
            continue

        if make_plot:
            fig, axis = plt.subplots(2, sharex=True)

        wave, flux, uncert, mask = frame.get_data_from_spectral_order(
            order, include_invalid=include_invalid
        )
        first_derivative, errors = first_numerical_derivative(wave, flux, uncert)
        first_derivative = np.asarray(first_derivative)
        derivative_regions = np.where(
            np.logical_or(
                np.subtract(first_derivative, 2 * np.asarray(errors)) >= 0,
                np.subtract(first_derivative, -2 * np.asarray(errors)) <= 0,
            )
        )

        blocks = build_blocks(derivative_regions)
        pixel_jumps = []
        for b_index in range(len(blocks) - 1):
            pixel_jumps.append(blocks[b_index + 1][0] - blocks[b_index][-1])
        marked_regions = []
        new_block = True
        for jump_index, jump in enumerate(
            pixel_jumps
        ):  # TODO: missing the last block if it is not merged!
            if new_block:
                start = blocks[jump_index][0]
                end = blocks[jump_index][-1]

            if jump < 10 and jump_index < len(pixel_jumps) - 1 and (end + 1 - start) >= 2:
                end = blocks[jump_index + 1][-1]
                new_block = False
            else:
                new_block = True
                if (end + 1 - start) < 10:
                    continue
                marked_regions.extend(range(start, end + 1))
        if marked_regions[-1] != blocks[-1][-1]:
            # print("here", blocks[-1], range(blocks[-1][0], blocks[-1][-1] + 1))
            # THis does not extend the last detection to match the previous one!!!
            marked_regions.extend(range(blocks[-1][0], blocks[-1][-1] + 1))

        is_line[order][marked_regions] = 1
        if make_plot:
            axis[0].scatter(
                wave[marked_regions], flux[marked_regions], color="red", ls="--", marker="d"
            )
            axis[0].plot(wave, flux)
            axis[1].plot(wave, first_derivative, marker="+")
            axis[1].scatter(
                wave[derivative_regions], first_derivative[derivative_regions], color="blue"
            )

            plt.show()

    return is_line


if __name__ == "__main__":
    path = Path("/home/amiguel/phd/spectra_collection/ESPRESSO/TauCeti")

    from SBART.Instruments import ESPRESSO

    frame = ESPRESSO(path / "r.ESPRE.2022-07-06T09:22:31.285_S2D_BLAZE_A.fits", user_configs={})

    find_lines_indexes(frame, list(range(0, 100)))
