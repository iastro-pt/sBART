import numpy as np
from astropy.units import Quantity

from SBART.utils.units import convert_data, meter_second


def json_ready_converter(value):
    """Converts a given value to a json-serializable format!
    TODO: also prepare SBART objects that have the to_json attribute!

    Parameters
    ----------
    value : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    if isinstance(value, set):
        value = list(value)
    elif isinstance(value, Quantity):
        value = value.to(meter_second).value
    elif isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and len(value) > 0:
        if isinstance(value[0], Quantity):
            # TODO: this will fail if we mix QUantity and unitless in the same list!
            # TODO: this will also fail on lists of lists!
            value = convert_data(value, new_units=meter_second, as_value=True)
    return value
