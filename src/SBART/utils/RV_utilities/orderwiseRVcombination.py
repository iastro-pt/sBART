import numpy as np

from SBART.data_objects.RV_cube import RV_cube
from SBART.utils.math_tools.weighted_mean import weighted_mean


def orderwise_combination(RV_cube, variance_estimator) -> RV_cube:
    """
    Merge the information from the individual orders.

    Automatically convert the results to meter_second, as this is the internal "selected" units!

    """
    rvs, errors, _ = RV_cube.data

    problematic_orders = RV_cube.problematic_orders
    rvs[:, problematic_orders] = np.nan
    errors[:, problematic_orders] = np.nan

    squared_errors = errors**2
    final_rv, final_error = weighted_mean(rvs, squared_errors, variance_estimator)

    return final_rv, final_error
