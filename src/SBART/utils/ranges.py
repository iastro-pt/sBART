import numpy as np


def ranges(a):
    if len(a) == 0:
        return a
    if isinstance(a, np.ndarray) and a.ndim == 0:
        return a

    a = np.sort(a)
    _none_1 = a[np.ediff1d(a, None, -1) != 1]
    _1_none = a[np.ediff1d(a, -1, None) != 1]
    starts = np.setdiff1d(_none_1, _1_none)
    ends = np.setdiff1d(_1_none, _none_1)
    uniques = np.intersect1d(_none_1, _1_none)
    R, nums = [], []
    for start, end in zip(starts, ends):
        R.append(f'{start}-{end}')
        nums.append(start)
    for u in uniques:
        R.append(str(u))
        nums.append(u)
    R = np.array(R)[np.argsort(nums)]
    return ', '.join(R)
