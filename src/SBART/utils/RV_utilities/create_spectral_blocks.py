import numpy as np


def build_blocks(indexes):
    """Evaluate the output of a np.where to find the continuous regions in it.
    This returns a numpy array, where each entry is a list of a 'set' of ones.

    E.g.
    >>> arr = np.array([0,0,1,1,1,1,1,1,0,1,0])
    >>> print(build_blocks(arr))
    [[2, 3, 4, 5, 6, 7], [9]]


    Parameters
    ----------
    indexes : [type]
        [description]

    Returns
    -------
    [type]cd
        [description]
    """
    diffs = np.diff(indexes)

    if len(indexes[0]) == 0:
        return []
    grouped_indexes = [[indexes[0][0]]]
    for index, dif in enumerate(diffs[0]):  # diffs are offseted by 1 index
        if dif == 1:
            grouped_indexes[-1].append(indexes[0][index + 1])
        else:
            grouped_indexes.append([indexes[0][index + 1]])

    return grouped_indexes


if __name__ == "__main__":
    import numpy as np

    arr = np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0])
    ind = np.where(arr != 0)
    print(build_blocks(ind))
    print(arr[ind])
