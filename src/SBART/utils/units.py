"""
Collection of utilities to handle units throughout the code base. Aside from the data conversion function, it also
provides three RV "measurements", which are *astropy* Quantities:

* meter_second
* kilometer_second
* centimeter_second


"""

from typing import Optional, Union

from astropy import units

from SBART.utils.types import RV_measurement, data_vector

meter_second = 1 * units.m / units.second

kilometer_second = 1 * units.km / units.second

centimeter_second = 1 * units.cm / units.second


def convert_data(
    data: Union[data_vector, RV_measurement],
    new_units: Optional[RV_measurement] = None,
    as_value: bool = False,
):
    """
    Accept input data that are astropy.units and converts it to a new unit and/or to a numerical value

    Parameters
    =============
    data :
        Input data
    new_units :
        Units to which each element will be converted to
    as_value :
        Switch the list entries to numerical values

    Returns
    ===========
    converted_list : list
        Data in the specified format
    """

    single_elem = False
    data_to_process = data

    if not isinstance(data, (list, tuple)):
        single_elem = True
        data_to_process = [data]

    if new_units is not None:
        converted_list = [i.to(new_units) for i in data_to_process]
    else:
        converted_list = data_to_process

    if as_value:
        converted_list = [i.value for i in converted_list]

    return converted_list if not single_elem else converted_list[0]


if __name__ == "__main__":
    test_list = [meter_second, centimeter_second]

    for A in [meter_second, test_list]:
        print(convert_data(A))
        print(convert_data(A, as_value=True))
        print(convert_data(A, new_units=kilometer_second))
        print(convert_data(A, new_units=kilometer_second, as_value=True))
