from copy import deepcopy
from typing import Any, Dict, NoReturn, Optional

import numpy as np
from loguru import logger

from SBART.utils.custom_exceptions import InvalidConfiguration


class Constraint:
    def __init__(self, const_text: str):
        self._constraint_list = [self.evaluate]
        self.constraint_text = const_text

    def __add__(self, other):
        new_const = Constraint(self.constraint_text)
        # ensure that we don't propagate changes to all existing constraints
        new_const._constraint_list = deepcopy(self._constraint_list)
        new_const._constraint_list.append(other.evaluate)
        new_const.constraint_text += " and " + other.constraint_text

        return new_const

    def __radd__(self, other):
        return self.__add__(other)

    def evaluate(self, param_name, value):
        ...

    def apply_to_value(self, param_name: str, value: Any) -> NoReturn:
        for evaluator in self._constraint_list:
            evaluator(param_name, value)

    def __str__(self):
        return self.constraint_text

    def __repr__(self):
        return self.constraint_text

    def __call__(self, value):
        for evaluator in self._constraint_list:
            print(evaluator)
            evaluator(value)


class ValueInInterval(Constraint):
    def __init__(self, interval, include_edges: bool = False):
        super().__init__(const_text=f"Value inside interval <{interval}>; Edges: {include_edges}")
        self._interval = interval
        self._include_edges = include_edges

    def evaluate(self, param_name: str, value: Any) -> NoReturn:
        good_value = False
        try:
            if self._include_edges:
                if self._interval[0] <= value <= self._interval[1]:
                    good_value = True
            else:
                if self._interval[0] < value < self._interval[1]:
                    good_value = True
        except TypeError:
            raise InvalidConfiguration(
                f"Config ({param_name}) value can't be compared with the the interval: {type(value)} vs {self._interval}"
            )

        if not good_value:
            raise InvalidConfiguration(
                f"Config ({param_name}) value not inside the interval: {value} vs {self._interval}"
            )


class ValueFromDtype(Constraint):
    def __init__(self, dtype_list):
        super().__init__(const_text=f"Value from dtype <{dtype_list}>")
        self.valid_dtypes = dtype_list

    def evaluate(self, param_name: str, value: Any) -> NoReturn:
        if not isinstance(value, self.valid_dtypes):
            raise InvalidConfiguration(
                f"Config ({param_name}) value ({value}) not from the valid dtypes: {type(value)} vs {self.valid_dtypes}"
            )


class ValueFromList(Constraint):
    def __init__(self, available_options):
        super().__init__(const_text=f"Value from list <{available_options}>")
        self.available_options = available_options

    def evaluate(self, param_name, value) -> NoReturn:
        bad_value = False
        if isinstance(value, (list, tuple)):
            for element in value:
                if element not in self.available_options:
                    bad_value = True
                    break
        else:
            if value not in self.available_options:
                bad_value = True

        if bad_value:
            raise InvalidConfiguration(
                f"Config ({param_name})  value not one of the valid ones: {value} vs {self.available_options}"
            )


Positive_Value_Constraint = ValueInInterval([0, np.inf], include_edges=True)
StringValue = ValueFromDtype((str,))
NumericValue = ValueFromDtype((int, float))
IntegerValue = ValueFromDtype((int,))
BooleanValue = ValueFromDtype((bool,))
ListValue = ValueFromDtype((list, tuple))


class UserParam:
    __slots__ = ("_valueConstraint", "_default_value", "_mandatory", "quiet")

    def __init__(
        self,
        default_value: Optional[Any] = None,
        constraint: Optional[Constraint] = None,
        mandatory: bool = False,
        quiet: bool = False,
    ):
        self._valueConstraint = constraint if constraint is not None else Constraint("")
        self._default_value = default_value
        self._mandatory = mandatory
        self.quiet = quiet

    def apply_constraints_to_value(self, param_name, value) -> NoReturn:
        self._valueConstraint.apply_to_value(param_name, value)

    @property
    def existing_constraints(self):
        return self._valueConstraint

    @property
    def is_mandatory(self) -> bool:
        return self._mandatory

    @property
    def quiet_output(self) -> bool:
        return self.quiet

    @property
    def default_value(self) -> Any:
        if self.is_mandatory:
            raise InvalidConfiguration("Trying to use default value of a mandatory parameter")

        self.apply_constraints_to_value("default_value", self._default_value)
        return self._default_value

    def __repr__(self):
        return f"Default Value: {self._default_value}; Mandatory Flag: {self._mandatory}; Constraints: {self._valueConstraint}\n"


class InternalParameters:
    __slots__ = ("_default_params", "_user_configs", "_name_of_parent")

    def __init__(self, name_of_parent, default_params: Dict[str, UserParam]):
        self._default_params = default_params
        self._user_configs = {}
        self._name_of_parent = name_of_parent

    def receive_user_inputs(self, user_configs: Optional[Dict[str, Any]] = None):
        logger.debug("Generating internal configs of {}", self._name_of_parent)

        for key, value in user_configs.items():
            try:
                parameter_def_information = self._default_params[key]
            except KeyError:
                logger.warning(
                    "{} received a configuration flag that is not recognized: {}",
                    self._name_of_parent,
                    key,
                )
                continue

            parameter_def_information.apply_constraints_to_value(key, value)
            self._user_configs[key] = value
            if not self._default_params[key].quiet_output:
                logger.debug("Configuration <{}> taking the value: {}", key, value)
            else:
                logger.debug("Configuration <{}> was updated")

        for key, default_param in self._default_params.items():
            if key not in self._user_configs:
                logger.debug(
                    "Configuration <{}> using the default value: {}",
                    key,
                    default_param.default_value,
                )
                try:
                    self._user_configs[key] = default_param.default_value
                except Exception as e:
                    logger.critical("Error in the generation of parameter: {}", key)
                    raise e

    def text_pretty_description(self, indent_level: int) -> str:
        file_offset = indent_level * "\t"
        to_write = ""

        to_write += f"\n{file_offset}Parameters in use:"
        for key, value in self._user_configs.items():
            to_write += f"\n{file_offset}\t{key} -> {value}"
        return to_write

    def __getitem__(self, item):
        try:
            return self._user_configs[item]
        except KeyError:
            msg = f"<{item}> is not a valid parameter of {self._name_of_parent}"
            logger.critical(msg)
            raise Exception(msg)

    def get_user_configs(self) -> Dict[str, Any]:
        return self._user_configs

    def items(self):
        return self._user_configs.items()


class DefaultValues:
    """
    Holds all of the user parameters that SBART has available for any given object.
    """

    def __init__(self, **kwargs):
        self.default_mapping = kwargs

    def update(self, item: str, new_value: Any):
        """
        Update the default value of a stored parameter, if it exists. Otherwise, raises an Exception

        Parameters
        ----------
        item
        new_value

        Returns
        -------

        """
        if item not in self.default_mapping:
            raise Exception
        self.default_mapping[item] = new_value

    def __add__(self, other):
        new_default_mapping = {**self.default_mapping, **other.default_mapping}
        return DefaultValues(**new_default_mapping)

    def __radd__(self, other):
        return self.__add__(other)

    def __getitem__(self, item):
        return self.default_mapping[item]

    def __str__(self):
        return self.__repr__()

    def __repr__(self) -> str:
        representation = "User_ params:\n\n"

        for key, value in self.default_mapping.items():
            representation += f"|{key} : {value} \n"

        return representation

    ### Map the inside dict properties to the outside!
    def items(self):
        return self.default_mapping.items()

    def keys(self):
        return self.default_mapping.keys()

    def values(self):
        self.default_mapping.values()

    def __iter__(self):
        raise TypeError("no!")


if __name__ == "__main__":
    print("yeuryuer")

    x = DefaultValues(a="asdsad") + DefaultValues(v=893712)
    print(x.keys())
    const = ValueInInterval([0, np.inf], include_edges=True) + ValueFromDtype((int,))
    const.apply_to_value("a", -1)
