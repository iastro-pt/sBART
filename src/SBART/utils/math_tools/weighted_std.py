import numpy as np


def wstd(y, e, dim=(), ret_err=False):
    """
    Compute the (biased) weighted standard deviation of the y data

    Parameters
    """
    W_mean = np.average(y, weights=e)
    variance = np.average((y - W_mean) ** 2, weights=e)
    return np.sqrt(variance)
