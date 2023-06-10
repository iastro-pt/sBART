import numpy as np


def find_close_lambdas(original_reference_position, used_lambdas):
    """
    Searches in which positions from original_reference_position we should insert used_lambdas values,
    to maintain order; In practice this searches for the closest point to each lambda

    If two values should be followed they have the same index, i.e. insert 4,5 in [2,3, 10,11] we would get [2,2]
    thus we have to add +1 to repeated values, to maintain ordering
    """

    if len(used_lambdas) == 0:
        return []
    points = np.searchsorted(original_reference_position, used_lambdas)
    final_indices = [points[0]]
    repeated = 0
    for ind in points[1:]:
        if ind == final_indices[-1]:
            repeated += 1
        else:
            repeated = 0
        final_indices.append(ind + repeated)
    return final_indices
