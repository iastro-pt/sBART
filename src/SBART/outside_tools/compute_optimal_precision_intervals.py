from tabletexifier import Table

from SBART.Instruments import ESPRESSO
from SBART.utils.units import convert_data

try:
    from eniric.precision import rv_precision

    ENIRIC_AVAILABLE = True
except Exception:
    ENIRIC_AVAILABLE = False

import numpy as np

from SBART.data_objects import DataClass


def compute_interval_value(data, start, end):
    return 1 / np.sqrt(np.sum(1 / data[:, start : end + 1] ** 2, axis=1))


def generate_all_possible_combinations(available_orders, N_intervals=4, min_interval_size=10):
    """Generate all possible combinations of all elements

    Args:
        available_orders (List[int]): List of indexes that we want to group together
        N_intervals (int, optional): Number of intervals. Defaults to 4.
        min_interval_size (int, optional): Minimum number of points in the interval. Defaults to 10.

    Returns:
        _type_: _description_

    """
    combinations = []
    levels = []
    # The level represents the left-to-right index of the interval
    # |---0---|---1---|
    for lvl in range(N_intervals - 1):
        # Max index at which a level can end
        max_index_end = len(available_orders) - min_interval_size * (N_intervals - lvl - 2)
        levels.append(max_index_end)

    first_interval_end = list(range(min_interval_size, levels[0] + 1))
    print(f"Generated {levels=}, assuming {len(available_orders)=}, {min_interval_size=}, and {N_intervals=}")
    combinations = [[(0, i)] for i in first_interval_end]
    level_index = 1
    for level in range(N_intervals - 1):
        new_comb = []
        for item in combinations:
            if level == N_intervals - 2:
                # Last loop, go from last value up to end of interval
                new_comb.append([*item, [item[-1][1], levels[-1]]])
            else:
                for end in range(item[-1][1] + min_interval_size, levels[level_index], 1):
                    new_comb.append([*item, [item[-1][1], end]])
        combinations = new_comb
        level_index += 1
    return combinations


def optimize_precision(data, orders_to_avoid, N_intervals=3, min_interval_size=10):
    """Computed the order-bins to generate time-series with similar expected RV precision

    Brute-force algorithm to find the optimal cromatic bins to ensure consistent RV precision in each
    time-series.


    Args:
        data (DataClass): DataClass object
        orders_to_avoid (list[int]): Orders that will not be used for the RV extraction
        N_intervals (int, optional): Number of bins in the spectral orders. Defaults to 3.
        min_interval_size (int, optional): Minimum number of order in each bin. Defaults to 10.

    Raises:
        ImportError: _description_

    """
    if not ENIRIC_AVAILABLE:
        raise ImportError("eniric needs to be installed")

    array_size = data.get_instrument_information()["array_sizes"]["S2D"]

    for subInst in data.get_subInstruments_with_valid_frames():
        print("Generating bins for", subInst)

        frameIDs = data.get_frameIDs_from_subInst(subInst)
        Nob = len(frameIDs)
        if isinstance(orders_to_avoid, dict):
            rejected_orders = orders_to_avoid[subInst]
        else:
            rejected_orders = orders_to_avoid

        list_of_orders = [i for i in range(array_size[0]) if i not in rejected_orders]
        empty_storage = np.zeros((Nob, len(list_of_orders)))

        for id_index, ID in enumerate(frameIDs):
            for ordeR_index, order in enumerate(list_of_orders):
                wave, flux, _, mask = data.get_frame_OBS_order(frameID=ID, order=order)
                precision = rv_precision(
                    wavelength=wave[~mask],
                    flux=flux[~mask],
                )
                empty_storage[id_index, ordeR_index] = convert_data(precision, as_value=True)

        intervals = generate_all_possible_combinations(
            list_of_orders, N_intervals=N_intervals, min_interval_size=min_interval_size,
        )
        result = []
        for combination in intervals:
            precisions = []
            for interval in combination:
                precisions.append(np.mean(compute_interval_value(empty_storage, *interval)))

            metric = np.sum([np.sum(np.divide(i, precisions)) for i in precisions]) - N_intervals**2
            result.append(metric)

        sort_indexes = np.argsort(result)

        tab = Table(["Intervals", "metric", "RV precision in each bin [m/s]"])

        for index in sort_indexes[:6]:
            precisions = []
            for interval in intervals[index]:
                precisions.append(np.mean(compute_interval_value(empty_storage, *interval)))

            row = [intervals[index], result[index], precisions]
            tab.add_row(row)
        print(tab)


if __name__ == "__main__":
    from pathlib import Path

    combinations = []
    N_intervals = 3
    min_interval_size = 10
    available_orders = list(range(170))

    combinations = generate_all_possible_combinations(available_orders=available_orders)

    paths = Path("/home/amiguel/phd/spectra_collection/ESPRESSO/HD10700_2series1night").glob("*S2D_A*")
    d = DataClass(
        input_files=list(paths)[::10],
        instrument=ESPRESSO,
        storage_path="",
        instrument_options={"apply_FluxCorr": True},
    )
    optimize_precision(d, [1, 2], N_intervals=N_intervals)
