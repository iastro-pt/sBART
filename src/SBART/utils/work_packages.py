from typing import Any, Dict, Iterable, NoReturn

import numpy as np

from SBART.utils.json_ready_converter import json_ready_converter
from SBART.utils.status_codes import ORDER_SKIP, Flag


class Package:
    __slots__ = ("params", "_locked_params", "extra_keys")

    def __init__(self, extra_keys=None, data_only_pkg=False):
        if data_only_pkg:
            self.params = {}
        else:
            self.params = {"shutdown": False}

        self._locked_params = []
        if extra_keys is None:
            self.extra_keys = []
        else:
            self.extra_keys = list(extra_keys)

    def items(self):
        return self.params.items()

    def to_list(self):
        return list(self.params.values())

    def lock_parameter(self, param):
        self._locked_params.append(param)

    def lock_parameters(self, parameters):
        for param in parameters:
            self.lock_parameter(param)

    def update_params(self, update_dict):
        for key, val in update_dict.items():
            self.update_param(key, val)

    def update_param(self, key, value):
        self.params[key] = value

    def __getitem__(self, key):
        return self.params[key]

    def __setitem__(self, key, val):
        if key in self._locked_params:
            raise Exception("Trying to change a locked entry")

        if key not in self.params and key not in self.extra_keys:
            self.extra_keys.append(key)
        self.params[key] = val

    def ingest_data(self, other):
        """Copy the data from the extra keys of the 'other' Package  to this one

        Parameters
        ----------
        other : Package
            Package with the data that will be copied over
        """
        for key in other.extra_keys:
            if key in self.params:
                Warning(
                    "Trying to override key {} during package ingestion. The two packages share the same extra key, skipping over this key".format(
                        key
                    )
                )
                continue
            # print(self.params)
            # print(other.params)
            self.params[key] = other[key]
            self.extra_keys.append(key)

    def delete_keys(self, keys_to_delete: Iterable[str]) -> NoReturn:
        for key in keys_to_delete:
            del self.params[key]

    def json_ready(self) -> Dict[str, Any]:
        out_pkg = {}

        for key, val in self.items():
            if key == "shutdown":
                continue
            if key == "status":
                out_pkg[key] = val.to_json()
            else:
                val = json_ready_converter(val)
                out_pkg[key] = val
        return out_pkg

    @classmethod
    def create_from_json(cls, json_info):
        new_pkg = Package(list(json_info.keys()))
        for key, val in json_info.items():
            if key == "status":
                new_pkg[key] = Flag.create_from_json(val)
            else:
                new_pkg[key] = val

        return new_pkg


class ShutdownPackage(Package):
    def __init__(self):
        super().__init__()
        self.params["shutdown"] = True


class BayesianPackage(Package):
    def __init__(self):
        super().__init__()
        keys = [
            "epoch",
            "order",
            "telluric_info",
            "jitter",
            "tentative_rv",
            "extra_skip",
            "weigthed",
            "rv_prior" "central_wavelength",
            "poly_params",
            "compute_FluxModel_misspecification",
        ]


class WorkerInput(Package):
    def __init__(self, extra_keys=None):
        super().__init__(extra_keys)

        self.params = {
            **self.params,
            **{
                "frameID": np.nan,
                "order": np.nan,
                "target_function": None,
                "tentative_RV": None,
                "subInst": None,
                "RVprior_params": [],
                "target_specific_configs": {},
            },
        }

        for key in self.extra_keys:
            if key in self.params:
                print(
                    f"Key {key} is among the default ones. Can't be overwritten in class instantiation"
                )

            self.params[key] = None


class WorkerOutput(Package):
    def __init__(self, extra_keys=None):
        super().__init__(extra_keys)

        self.params = {"epochID": np.nan, "order": np.nan, "status": ORDER_SKIP}

        for key in self.extra_keys:
            if key in self.params:
                print(
                    f"Key {key} is among the default ones. Can't be overwritten in class instantiation"
                )

            self.params[key] = None


class SamplerOutput(Package):
    def __init__(self):
        super().__init__()
        self.params = {
            "opt_status": None,
            "opt_message": None,
            "RV": 0,
            "RV_uncertainty": 0,
            "jitter": 0,
            "jitter_uncertainty": 0,
            "detailed_RV_likelihood": [],
            "detailed_jitter_likelihood": [],
        }
        self.extra_keys = [f"detailed_{i}_likelihood" for i in ["RV", "jitter"]]
