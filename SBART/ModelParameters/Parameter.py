from typing import Any, Dict, List, NoReturn, Tuple, Union

import numpy as np
from loguru import logger

from SBART.Base_Models.BASE import BASE
from SBART.Quality_Control import ensure_value_in_window
from SBART.utils import custom_exceptions
from SBART.utils.types import RV_measurement
from SBART.utils.UserConfigs import DefaultValues, UserParam, ValueFromList


class ModelComponent(BASE):
    _name = "ModelParameter"

    _default_params = BASE._default_params + DefaultValues(
        # How to ensure that we always use the same wavelength regions:
        #     GLOBAL - across all valid frames
        #     SUB-INSTRUMENT - inside the sub-Instrument
        #     OBSERVATION - only inside an observation
        GENERATION_MODE=UserParam(
            default_value="GLOBAL",
            constraint=ValueFromList(("GLOBAL", "SUB-INSTRUMENT", "OBSERVATION")),
            mandatory=False,
        )
    )

    def __init__(
        self,
        name,
        initial_guess=None,
        bounds=None,
        user_configs=None,
        default_enabled=False,
        param_type="general",
    ):
        """
        Define a parameter in our RV model. Contains information regarding the initial guess and the bounds that the
        parameter can take
        Parameters
        ----------
        name: str
            parameter name
        initial_guess: Any
            default guess for the parameters. Unless a children overrides the generate_priors function, this will be always used for all frameIDs
        bounds: List[Any, ANy]
            Bounds for the minimization. Pass a None to get an open interval
        user_configs:
            fairly certain that it does nothing
        param_type: string
            Describe an overall "theme" for the parameter. The MOdels then allow to search all parameters that "theme"

        name
        """
        super().__init__(user_configs=user_configs)
        self.param_name = name
        self._default_limits = bounds
        self._default_init_guess = initial_guess

        self.limits_for_frameID = {}
        self.guess_for_frameID = {}

        self.frameID_information = {}
        self.frameID_results = {}

        self._enabled = default_enabled
        self._locked = False

        self.parameter_type = param_type

    ###
    # Generate needed info
    ###

    def generate_priors(self, DataClassProxy):
        """
        Generate the initial guess and parameter bounds. Currently, only supports using the same initial guess
        and bounds for ALL subInstruments. For a more fine-tuned control, inherit from this class and overload this function

        Parameters
        ----------
        DataClassProxy
        bounds
        initial_guess

        Returns
        -------

        """
        for frameID in DataClassProxy.get_valid_frameIDS():
            self.generate_prior_from_frameID(frameID)

    def generate_prior_from_frameID(self, frameID):
        self._update_frameID_info(frameID, self._default_init_guess, self._default_limits)

    def _update_frameID_info(self, frameID, init_guess, bound, bypass_QC=False):

        if self.is_locked:
            logger.debug("Can't update the values of a locked parameter")
            return

        if not self.is_enabled:
            # If we log in here, we will get one line for each parameter (for each loaded OBS)
            # logger.warning("Trying to update a disabled prior. Doing nothing")
            return

        if not bypass_QC:
            window = bound.copy()
            if bound[0] is None:
                window[0] = -np.inf
            if bound[1] is None:
                window[1] = np.inf

            ensure_value_in_window(tentative_value=init_guess, desired_interval=window)

        if frameID not in self.frameID_information:
            self.frameID_information[frameID] = {
                "initial_guess": init_guess,
                "bounds": bound,
                "generated_prior": True,  # do we need this?
            }

            self.frameID_results[frameID] = np.nan
        else:
            if init_guess is not None:
                self.frameID_information[frameID]["initial_guess"] = init_guess

            if bound is not None:
                self.frameID_information[frameID]["bounds"] = bound

    def manual_set_frameID_info(self, frameID, init_guess, bound):
        self._update_frameID_info(frameID, init_guess, bound, bypass_QC=True)

    def store_fit_results(self, frameID, final_value):
        self.frameID_results[frameID] = final_value

    ###
    # Get parameter information
    ###

    def get_bounds(self, frameID: int):
        self.ensure_enabled_param(frameID, allow_disabled=False)
        return self.frameID_information[frameID]["bounds"]

    def get_initial_guess(self, frameID, allow_disabled=False) -> Union[float, RV_measurement]:
        self.ensure_enabled_param(frameID, allow_disabled)
        return self.frameID_information[frameID]["initial_guess"]

    def get_optimizer_input(self, frameID):
        self.ensure_enabled_param(frameID, allow_disabled=False)
        return self.get_initial_guess(frameID), self.get_bounds(frameID)

    def get_results_from_frameID(self, frameID: int, allow_disabled=False):
        self.ensure_enabled_param(frameID, allow_disabled=allow_disabled)
        return self.frameID_results[frameID]

    ###
    # Control "status" of parameter
    ###
    def ensure_enabled_param(self, frameID, allow_disabled):
        bad_param = False
        try:
            if not self.frameID_information[frameID]["generated_prior"]:
                bad_param = True

        except IndexError:
            bad_param = True

        if bad_param:
            msg = f"Attempting to use parameter ({self.param_name}) that did not generate the prior information"
            logger.critical(msg)
            raise custom_exceptions.InvalidConfiguration(msg)

        if not self.is_enabled and not allow_disabled:
            msg = f"Attempting to get information from disabled parameter: {self.param_name}"
            logger.critical(msg)
            raise custom_exceptions.InternalError(msg)

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def is_locked(self) -> bool:
        return self._locked

    def is_parameter_type(self, type_to_check):
        return self.parameter_type == type_to_check

    def lock_param(self) -> NoReturn:
        """
        Lock the parameter space of a given parameter. When doing so, checks:
            - if the lower bound of the param space is smaller than the upper bound
            - If the initial guess is inside the parameter space
        If any of the conditions is not met: raises SBART.utils.custom_exceptions.InvalidConfiguration

        Parameters
        ----------
        param: str
            Name of the parameter

        Returns
        -------

        """
        logger.info("Locking parameter of the model - {}", self.param_name)

        if not self.is_enabled:
            logger.critical("Trying to lock disabled parameter")
            raise Exception

        if self.is_locked:
            logger.warning("Trying to re-lock locked parameter")
            return
        self._change_lock_status(True)

    def unlock_param(self) -> NoReturn:
        self._change_lock_status(False)

    def enable_param(self) -> NoReturn:
        if self.is_enabled:
            logger.warning(
                "Attempting to enable param that is already enabled: {}", self.param_name
            )
            return
        self._change_enabled_status(True)

    def disable_param(self) -> NoReturn:
        if not self.is_enabled:
            logger.warning(
                "Attempting to disable param that is already disabled: {}", self.param_name
            )
            return
        if self.is_locked:
            logger.warning("Attempting to disable locked parameter")

        self._change_enabled_status(False)

    def _change_enabled_status(self, new_status: bool) -> NoReturn:
        self._enabled = new_status

    def _change_lock_status(self, new_status: bool) -> NoReturn:
        self._locked = new_status

    def string_representation(self, indent_level) -> str:
        msg = "\n"

        string_offset = indent_level * "\t"
        msg += f"\n{string_offset}Parameter - {self.param_name}:"
        msg += f"\n{string_offset}\tEnabled : {self.is_enabled};"
        msg += f"\n{string_offset}\tLocked : {self.is_locked}"
        msg += f"\n{string_offset}\tGeneration mode: {self._generation_mode}"

        return msg

    def __str__(self) -> str:
        return "Parameter::{}".format(self.param_name)

    def __repr__(self):
        return self.__str__()

    def json_ready(self) -> Dict[str, Any]:
        """
        "TODO: stop ignoring the user parameters!!!!"


        NOTE: this will NOT work for Models that have astropy.Quantity inside....
        Returns
        -------

        """
        base_json = super().json_ready()
        class_json = {
            self.param_name: {
                "name": self.param_name,
                "_enabled": self._enabled,
                "_locked": self._locked,
                "default_guess": self._default_init_guess,
                "default_bounds": self._default_limits,
                "parameter_type": self.parameter_type,
                "frameID_information": self.frameID_information,
                "frameID_results": self.frameID_results,
            }
        }

        return {**base_json, **class_json}

    @classmethod
    def load_from_json(cls, json_info):
        "TODO: stop ignoring the user parameters!!!!"
        comp = ModelComponent(
            name=json_info["name"],
            initial_guess=json_info["default_guess"],
            bounds=json_info["default_bounds"],
            user_configs={},
            default_enabled=True,
            param_type=json_info["parameter_type"],
        )

        comp.frameID_information = json_info["frameID_information"]
        comp.frameID_results = json_info["frameID_results"]

        if json_info["_locked"]:
            comp.lock_param()

        if not json_info["_enabled"]:
            comp.disable_param()

        return comp


class RV_component(ModelComponent):
    _name = ModelComponent._name + "::RV_component"

    def __init__(self, RVwindow, RV_keyword, user_configs):
        self.RVwindow = RVwindow
        self.RV_key = RV_keyword
        super().__init__(name="RV", user_configs=user_configs, default_enabled=True)

    def generate_priors(self, DataClassProxy):

        RV_default_window = self.RVwindow
        logger.debug("Generating RV priors")

        if self._internal_configs["GENERATION_MODE"] == "OBSERVATION":
            for frameID in DataClassProxy.get_valid_frameIDS():
                obs_rv = DataClassProxy.get_KW_from_frameID(self.RV_key, frameID)
                RVLowerBound = obs_rv - RV_default_window[0]
                RVUpperBound = obs_rv + RV_default_window[1]
                self._update_frameID_info(frameID, obs_rv, [RVLowerBound, RVUpperBound])

        elif self._internal_configs["GENERATION_MODE"] == "GLOBAL":
            subInstruments = DataClassProxy.get_subInstruments_with_valid_frames()
            available_previous_rvs = DataClassProxy.collect_KW_observations(
                KW=self.RV_key, subInstruments=subInstruments
            )

            RVLowerBound = min(available_previous_rvs) - RV_default_window[0]
            RVUpperBound = max(available_previous_rvs) + RV_default_window[1]

            for frameID in DataClassProxy.get_valid_frameIDS():
                obs_rv = DataClassProxy.get_KW_from_frameID("previousRV", frameID)
                self._update_frameID_info(frameID, obs_rv, [RVLowerBound, RVUpperBound])

        elif self._internal_configs["GENERATION_MODE"] == "SUB-INSTRUMENT":
            for subInst in DataClassProxy.get_subInstruments_with_valid_frames():
                available_previous_rvs = DataClassProxy.collect_KW_observations(
                    KW=self.RV_key, subInstruments=[subInst]
                )

                RVLowerBound = min(available_previous_rvs) - RV_default_window[0]
                RVUpperBound = max(available_previous_rvs) + RV_default_window[1]

                for frameID in DataClassProxy.get_frameIDs_from_subInst(subInst):
                    obs_rv = DataClassProxy.get_KW_from_frameID(self.RV_key, frameID)
                    self._update_frameID_info(frameID, obs_rv, [RVLowerBound, RVUpperBound])

    def string_representation(self, indent_level) -> str:
        string_offset = indent_level * "\t"
        return (
            super().string_representation(indent_level)
            + f"\n{string_offset}\tRV window:{self.RVwindow}"
        )

    def disable_param(self) -> NoReturn:
        msg = "Attempting to disable the RV parameter"
        logger.critical(msg)
        raise custom_exceptions.InvalidConfiguration(msg)
