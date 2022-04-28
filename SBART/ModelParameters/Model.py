import json
from pathlib import Path

from typing import Dict, List, NoReturn, Tuple, Union

from loguru import logger

from SBART.ModelParameters.Parameter import ModelComponent
from SBART.utils import custom_exceptions
from SBART.utils.status_codes import Flag
from SBART.utils.types import RV_measurement
from SBART.utils.units import convert_data


class Model:
    def __init__(self, params_of_model: List[ModelComponent]):
        logger.info(
            "Generating model with following available parameters: {}".format(params_of_model)
        )
        self.components = params_of_model
        self._results_flags = {}
        self.has_results_stored = False

    def generate_priors(self, DataClassProxy):
        logger.info("Generating priors of the model' parameters")
        for component in self.components:
            component.generate_priors(DataClassProxy)

    def generate_prior_from_frameID(self, frameID):
        for component in self.components:
            component.generate_prior_from_frameID(frameID)

    def update_params_initial_guesses(self, frameID, guesses: Dict):
        for param_name, value in guesses.items():
            component = self.get_component_by_name(param_name)
            component.manual_set_frameID_info(frameID, init_guess=value, bound=None)

    def add_extra_param(self, new_param):
        self.components.append(new_param)

    def generate_optimizer_inputs(self, frameID, rv_units) -> Tuple[List, List]:
        """
        Generate the information needed for the optimization:
            - list with the initial guesses
            - list with the bounds of each parameter (for bounded optimization). The format is:
                [[start, end], [start, None], [None, end], ...]
            Note: None is used to flag "open" edge, i.e. no bound on that direction

        Parameters
        ----------
        frameID
        rv_units
            If astropy unit, the Quantity is converted to this unit. TODO: rename param

        Returns
        -------

        """
        init_guesses = []
        bounds = []
        for component in self.get_enabled_components():
            init_guess, bound = component.get_optimizer_input(frameID)

            if isinstance(init_guess, RV_measurement):  # convert astropy.units to "unitless" values
                init_guess = convert_data(init_guess, new_units=rv_units, as_value=True)
                bound = convert_data(bound, new_units=rv_units, as_value=True)

            if isinstance(init_guess, (list, tuple)):
                # The chromatic trend has multiple parameters associated with it!
                # data["initial_guess"] and data["prior"] will be lists with the information
                # of each param!
                init_guesses.extend(init_guess)
                bounds.extend(bound)
            else:
                init_guesses.append(init_guess)
                bounds.append(bound)

        return init_guesses, bounds

    def check_if_enabled(self, param_name: str) -> bool:
        for component in self.components:
            if component.param_name == param_name:
                return component.is_enabled
        raise custom_exceptions.InvalidConfiguration("Parameter {} doesn't exist", param_name)

    def has_valid_identifier_results(self, identifier):
        if identifier in self._results_flags:
            return self._results_flags[identifier].is_good_flag
        return False

    def enable_param(self, param_name):
        self._change_param_settings(param_name, True, mode="enable")

    def disable_param(self, param_name):
        self._change_param_settings(param_name, False, mode="enable")

    def lock_param(self, param_name):
        self._change_param_settings(param_name, True, mode="lock")

    def unlock_param(self, param_name):
        self._change_param_settings(param_name, False, mode="lock")

    def store_frameID_results(
        self, frameID: int, result_vector: List[float], result_flag
    ) -> NoReturn:
        for comp_index, component in enumerate(self.get_enabled_components()):
            component.store_fit_results(
                frameID=frameID,
                final_value=result_vector[comp_index],
            )
        self._results_flags[frameID] = result_flag

        if not self.has_results_stored:
            self.has_results_stored = True

    def _change_param_settings(self, param_name: str, set_to_True: bool, mode: str) -> NoReturn:
        found = False
        for component in self.components:
            if component.param_name == param_name:
                found = True
                if set_to_True:
                    if mode == "enable":
                        component.enable_param()
                    elif mode == "lock":
                        component.lock_param()
                else:
                    if mode == "enable":
                        component.disable_param()
                    elif mode == "lock":
                        component.unlock_param()

        if not found:
            raise custom_exceptions.InvalidConfiguration("Param {} does not exist", param_name)

    def enable_full_model(self):
        for comp in self.components:
            comp.enable_param()

    def disable_full_model(self):
        for comp in self.components:
            comp.disable_param()

    def enable_param_group(self, param_type):
        for comp in self.get_components_of_type(param_type, only_enabled=False):
            comp.enable_param()

    def disable_param_group(self, param_type):
        for comp in self.get_components_of_type(param_type, only_enabled=False):
            comp.disable_param()

    ###
    # Access the parameters
    ###

    def get_enabled_params(self) -> List[str]:
        return [comp.param_name for comp in self.get_enabled_components()]

    def get_disabled_params(self) -> List[str]:
        return [comp.param_name for comp in self.get_disabled_components()]

    def get_enabled_components(self):
        return [comp for comp in self.components if comp.is_enabled]

    def get_disabled_components(self):
        return [comp for comp in self.components if not comp.is_enabled]

    def get_fit_results_from_frameID(self, frameID, allow_disabled=False):
        if not self._results_flags[frameID].is_good_flag:
            raise custom_exceptions.NoConvergenceError()

        return [
            comp.get_results_from_frameID(frameID, allow_disabled)
            for comp in self.get_components(include_disabled=allow_disabled)
        ]

    def get_initial_guess_of_component(
        self, param_name: str, frameID: int, allow_disabled: bool = False
    ):
        return self.get_component_by_name(param_name).get_initial_guess(
            frameID=frameID, allow_disabled=allow_disabled
        )

    def get_component_by_name(self, param_name):
        for comp in self.components:
            if comp.param_name == param_name:
                return comp
        raise custom_exceptions.InternalError(f"Parameter ({param_name}) does not exist")

    def get_components_of_type(self, desired_type, only_enabled=True):
        return [
            comp
            for comp in self.get_components(include_disabled=not only_enabled)
            if comp.is_parameter_type(desired_type)
        ]

    def get_names_of_type(self, desired_type, only_enabled=True):
        return [comp.param_name for comp in self.get_components_of_type(desired_type, only_enabled)]

    def get_components(self, include_disabled=False):
        return [comp for comp in self.components if (comp.is_enabled or include_disabled)]

    def get_component_names(self, include_disabled):
        return [comp.param_name for comp in self.get_components(include_disabled)]

    def __str__(self):
        out = "Model:"
        for comp in self.components:
            out += "\n\t{} - Enabled: {}".format(comp.param_name, comp.is_enabled)
        return out

    #####
    #   Disk operations
    ####

    def save_to_json_file(self, storage_path):

        full_json = {"components": {}, "result_flags": None}

        for component in self.components:
            full_json["components"] = {**full_json["components"], **component.json_ready()}

        full_json["result_flags"] = {
            int(d_ID): flag.to_json() for d_ID, flag in self._results_flags.items()
        }
        print(full_json)
        with open(storage_path, mode="w") as handle:
            json.dump(full_json, handle, indent=4)

    @classmethod
    def load_from_json(
        cls,
        storage_path: Union[Path, str],
        component_to_use: ModelComponent = ModelComponent,
    ):
        with open(storage_path) as handle:
            data = json.load(handle)

        model_params = []
        for key, json_info in data["components"].items():
            model_params.append(component_to_use.load_from_json(json_info))

        loaded_model = Model(params_of_model=model_params)
        loaded_model._results_flags = {
            int(d_ID): Flag.create_from_json(info) for d_ID, info in data["result_flags"].items()
        }

        return loaded_model


if __name__ == "__main__":
    from SBART.ModelParameters import ModelComponent

    components = [
        ModelComponent(name="a", bounds=[0, 1], initial_guess=2),
        ModelComponent(name="B", bounds=[0, 1], initial_guess=2),
    ]
    model = Model(components)
