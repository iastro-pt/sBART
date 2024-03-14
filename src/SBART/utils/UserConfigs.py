from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, NoReturn, Optional

import numpy as np
from loguru import logger

from SBART.utils.custom_exceptions import InvalidConfiguration, InternalError


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


class IterableMustHave(Constraint):
    def __init__(self, available_options, mode: str = "all"):
        super().__init__(const_text=f"Must have value from list <{available_options}>")
        self.available_options = available_options
        self.mode = mode

        if mode not in ["all", "either"]:
            raise InternalError("Using the wrong mode")

    def evaluate(self, param_name, value) -> NoReturn:
        if not isinstance(value, (list, tuple)):
            raise InvalidConfiguration("Constraint needs a list or tuple")

        evaluation = [i in value for i in self.available_options]

        good_value = False

        if self.mode == "all":
            good_value = all(evaluation)
        elif self.mode == "either":
            good_value = any(evaluation)

        if not good_value:
            raise InvalidConfiguration(
                f"Config ({param_name}) value {value} does not have {self.mode} of {self.available_options}"
            )


Positive_Value_Constraint = ValueInInterval([0, np.inf], include_edges=True)
StringValue = ValueFromDtype((str,))
PathValue = ValueFromDtype((str, Path))
NumericValue = ValueFromDtype((int, float))
IntegerValue = ValueFromDtype((int,))
BooleanValue = ValueFromDtype((bool,))
ListValue = ValueFromDtype((list, tuple))


class UserParam:
    __slots__ = (
        "_valueConstraint",
        "_default_value",
        "_mandatory",
        "quiet",
        "description",
    )

    def __init__(
        self,
        default_value: Optional[Any] = None,
        constraint: Optional[Constraint] = None,
        mandatory: bool = False,
        quiet: bool = False,
        description: Optional[str] = None,
    ):
        self._valueConstraint = constraint if constraint is not None else Constraint("")
        self._default_value = default_value
        self._mandatory = mandatory
        self.quiet = quiet
        self.description = description

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
        return f" Mandatory Flag: {self._mandatory}\nDefault Value: {self._default_value}\n Constraints: {self._valueConstraint}\n"

    def get_terminal_output(self, indent_level: int = 1) -> str:
        """Generate terminal-formatted text from this UserParam

        Args:
            indent_level (int, optional): How many tabs to add at the start of each line. Defaults to 1.

        Returns:
            str: Formatted message with the Description, mandatory status, default value and constraints
        """
        offset = indent_level * "\t"
        message = ""
        for name, value in [
            ("Description", self.description),
            ("Mandatory", self._mandatory),
            ("Default value", self._default_value),
            ("Constraints", self._valueConstraint),
        ]:
            message += offset + f"{name}:: {value}\n"
        return message


class InternalParameters:
    __slots__ = ("_default_params", "_user_configs", "_name_of_parent", "no_logs")

    def __init__(
        self,
        name_of_parent,
        default_params: Dict[str, UserParam],
        no_logs: bool = False,
    ):
        self._default_params = default_params
        self._user_configs = {}
        self._name_of_parent = name_of_parent
        self.no_logs = no_logs

    def update_configs_with_values(self, user_configs):
        for key, value in user_configs.items():
            try:
                parameter_def_information = self._default_params[key]
            except KeyError:
                if not self.no_logs:
                    # The only object that will have this enabled are the Frames
                    # And we shall call one of the Frames with the User-Param logs enabled!
                    logger.warning(
                        "{} received a configuration flag that is not recognized: {}",
                        self._name_of_parent,
                        key,
                    )
                continue

            try:
                parameter_def_information.apply_constraints_to_value(key, value)
            except InvalidConfiguration as exc:
                logger.critical("User-given parameter {} does not meet the constraints", key)
                raise InternalError from exc

            self._user_configs[key] = value

            if not self.no_logs:
                if not self._default_params[key].quiet_output:
                    logger.debug("Configuration <{}> taking the value: {}", key, value)
                else:
                    logger.debug("Configuration <{}> was updated")

    def receive_user_inputs(self, user_configs: Optional[Dict[str, Any]] = None):
        if not self.no_logs:
            logger.debug("Generating internal configs of {}", self._name_of_parent)

        self.update_configs_with_values(user_configs)

        if not self.no_logs:
            logger.info("Checking for any parameter that will take default value")
        for key, default_param in self._default_params.items():
            if key not in self._user_configs:
                if default_param.is_mandatory:
                    raise InvalidConfiguration(f"SBART parameter <{key}> is mandatory.")

                if not self.no_logs:
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

    def __setitem__(self, key, value):
        logger.warning(f"Internal configs are being updated in real time ({key=})")
        try:
            parameter_def_information = self._default_params[key]
        except KeyError:
            if not self.no_logs:
                # The only object that will have this enabled are the Frames
                # And we shall call one of the Frames with the User-Param logs enabled!
                logger.warning(
                    "{} received a configuration flag that is not recognized: {}",
                    self._name_of_parent,
                    key,
                )
        try:
            parameter_def_information.apply_constraints_to_value(key, value)
        except InvalidConfiguration as exc:
            logger.critical("User-given parameter {} does not meet the constraints", key)
            raise InternalError from exc
        self._user_configs[key] = value

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
        representation = f"Configurations:\n\n"

        for key, value in self.default_mapping.items():
            representation += f"Name:: {key}\n{value.get_terminal_output()} \n"

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
