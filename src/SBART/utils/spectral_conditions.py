"""
.. _SBART-OBS-REJECTION:

==============================================
Selecting Observations based on conditions
==============================================

When using SBART we can reject observations in two different ways:

- globally - they are effectivelly discarded for all SBART operations
- temporarily -Only disable them for the creation of the Stellar Templates

Even though the two rejection types have completely different effects, we always use the same format, as defined in :py:class:`~SBART.utils.spectral_conditions`.


Based on Header values
================================

- Allows to define bounds for the "valid" values that a given header value can take.
- The bounds argument takes a list of list, with each entry defining a "region" that the value can take.
- If we want to place no upper/lower limits, we can place a None

.. code-block:: python

    # Only use observations that can have AIRMASS smaller than 1.5 or larger than 1.6
    valid_AIRM_condition = KEYWORD_condition(KW = "airmass",
                                          bounds=[[None, 1.5], [1.6, None]]
                                         )

    # Only select observations that have a previous RV error (the one from the CCF) smaller than 50 meter_second
    # Note1: the conditions for 'RV related' values (previousRV, drift, BERV) must be in meter_second OR kilometer_second
    # Note2: the units will be converted later on, so either "unit" can be chosen

    valid_error_condition = KEYWORD_condition(KW="previousRV_ERR",
                                              bounds = [0, 50*meter_second])


Rejection based on filenames
================================

Provide a list of bad filenames that will be rejected.

**Note:** Filename only, do NOT pass a path


.. code-block:: python

    bad_filename_condition = FNAME_condition(filename_list=["filename.fits"])

Rejecting based on the "subInstrument"
========================================

# Reject all observations from ESPRESSO21

.. code-block:: python

    bad_subInst_condition = SubInstrument_condition("ESPRESSO21")

Rejecting based on Warning Flags when loading the Frames
=============================================================

Reject observation if a given warning flag was set when loading the spectra

.. code-block:: python

    bad_subInst_condition = WarningFlag_Notset("HIERARCH ESO QC SCIRED DRIFT CHECK")

Combining conditions
================================

To combine multiple conditions, sum them together.
The observations will be rejected if **any** Condition decides to reject it

.. code-block:: python

    full_conditions = valid_AIRM_condition + valid_error_condition + bad_filename_condition + bad_subInst_condition


Applying the conditions
================================

For a :py:class:`~SBART.data_objects.DataClass.DataClass` object, we can use its :obj:`~SBART.data_objects.DataClass.DataClass.reject_observations` method

.. code-block:: python

    data.reject_observations(full_conditions)

"""

from pathlib import Path
from typing import Any, List, Tuple, overload

import numpy as np
from loguru import logger

from SBART.utils import custom_exceptions
from SBART.utils.custom_exceptions import InvalidConfiguration
from SBART.utils.status_codes import USER_BLOCKED, VALID, Flag, KW_WARNING
from typing_extensions import override


class ConditionModel:
    """
    Defines the Base Condition Class, from which we can generate new conditions. The conditions represent a boolean check that
    is applied to a given spectra, which is then rejected or kept, depending on the result from the check.

    The ConditionModel is implemented in such a way that summing two conditions together creates a new, independent,
    condition that combines them all. This new, merged, conditions works as a sequential application of all conditions.

    """

    def __init__(self):
        self._condition_list = [self.select_spectra]
        self._cond_information = [self.cond_info]

    def __add__(self, other):
        self._condition_list.append(other.select_spectra)
        self._cond_information.append(other.cond_info)
        return self

    def __radd__(self, other):
        return self.__add__(other)

    def evaluate(self, frame) -> Tuple[bool, List[Flag]]:
        """
        Apply all boolean checks to a given frame, returning a boolean value and a list of Flags with the results.

        Parameters
        ----------
        frame:
            frame to be validated against the conditions

        Returns
        -------
        valid_OBS:
            Boolean result of the comparison
        flags:
            List of flags, one for each condition that was applied
        """
        valid_OBS = True

        flags = []
        for condition in self._condition_list:
            output_flag = condition(frame)
            if output_flag != VALID:
                flags.append(output_flag)
                valid_OBS = False

        return valid_OBS, flags

    def select_spectra(self, frame) -> Flag:
        # must return a given flag and message
        return [], ""

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"{self._cond_information}"

    @property
    def set_conditions(self):
        return self._cond_information

    @property
    def cond_info(self):
        return "Base spectral condition"

    def write_to_disk(self, file) -> None:
        file.write("Constraints on observations:\n")
        for condition in self._cond_information:
            file.write("\t" + "".join(condition) + "\n")


class KEYWORD_condition(ConditionModel):
    """
    Limit the KW to be inside the defined interval (edges included)
    Parameters
    ===============
    """

    def __init__(self, KW: str, bounds: List[Any], include_edges: bool = True):
        self.KW = KW
        self.bounds = bounds
        super().__init__()

        if len(bounds) == 0:
            raise InvalidConfiguration("No bounds were provided")

        self._include_edges = include_edges

        if isinstance(bounds[0], (list, tuple)):
            bounds_to_check = self.bounds
            self._multiple_bounds = True
        else:
            bounds_to_check = [self.bounds]
            self._multiple_bounds = False

        self._standardize_bounds()
        for bound_entry in bounds_to_check:
            if bound_entry[0] > bound_entry[1]:
                raise InvalidConfiguration("The lower bound must be larger than the upper one")

    def _standardize_bounds(self):
        new_bounds = []
        for bound in self._bounds_to_check:
            new_entry = bound
            if bound[0] is None:
                new_entry[0] = -np.inf
            if bound[1] is None:
                new_entry[1] = np.inf
            new_bounds.append(new_entry)

        if not self._multiple_bounds:
            self.bounds = new_bounds[0]
        else:
            self.bounds = new_bounds

    def select_spectra(self, frame):
        keep = False
        KW_val = frame.get_KW_value(self.KW)
        for bound_elem in self._bounds_to_check:
            if not np.isfinite(KW_val):
                keep = True
                logger.warning(
                    "Frame has a NaN value for the KW: {}. Not applying the spectral condition",
                    self.KW,
                )
            if self._include_edges:
                if bound_elem[0] <= KW_val <= bound_elem[1]:
                    keep = True
            else:
                if bound_elem[0] < KW_val < bound_elem[1]:
                    keep = True

            if keep:
                break

        message = "KW {} inside the boundary {}".format(self.KW, self.bounds)
        if not keep:
            message = "KW {} outside the boundary {} : {}".format(self.KW, self.bounds, KW_val)

        flag = VALID if keep else USER_BLOCKED(message)

        return flag

    @property
    def _bounds_to_check(self) -> List[list]:
        if self._multiple_bounds:
            bounds_to_check = self.bounds
        else:
            bounds_to_check = [self.bounds]
        return bounds_to_check

    @property
    def cond_info(self):
        return "KW {} inside window: {}".format(self.KW, self.bounds)


class SubInstrument_condition(ConditionModel):
    """
    Flag the observations that are from the defined subInstrument

    Parameters
    ============
    """

    def __init__(self, subInst: str):
        self._bad_subInst = subInst
        super().__init__()

    def select_spectra(self, frame) -> Flag:
        if frame.is_SubInstrument(self._bad_subInst):
            message = "Removed subInstrument: {}".format(self._bad_subInst)
            flag = USER_BLOCKED(message)
        else:
            flag = VALID
        return flag

    @property
    def cond_info(self) -> str:
        return "Removed subInstrument {}".format(self._bad_subInst)


class Empty_condition(ConditionModel):
    """
    Place no Condition

    Parameters
    ============
    """

    def __init__(self):
        super().__init__()

    def select_spectra(self, frame) -> Flag:
        return VALID

    @property
    def cond_info(self) -> str:
        return "No conditions"


class WarningFlag_Notset(ConditionModel):
    """
    Reject the observation if the given warning flag is True
    """

    def __init__(self, flag_name: str, full_flag=False):
        self.flag_name = flag_name
        self.full_flag = full_flag
        super().__init__()

    @override
    def select_spectra(self, frame) -> Flag:
        if self.full_flag:
            msg = self.flag_name
        else:
            msg = f"QC flag {self.flag_name} meets the bad value"
        KW_flag = KW_WARNING(msg)
        if frame.status.check_if_warning_exists(KW_flag):
            message = f"Frame has the KW warning flag {self.flag_name} active"
            flag = USER_BLOCKED(message)
        else:
            flag = VALID

        return flag

    @property
    def cond_info(self) -> str:
        return "Warning KW flag {} was raised".format(self.flag_name)


class FNAME_condition(ConditionModel):
    """
    Flag the observations that have filename inside the filename_list list

    Parameters
    ============
    filename_list: list
        List of files to either be outright used or list of paths to .txt files from which we can select
        observations
    only_keep_filenames: bool
        Only keep the **filenames** that were selected. Default: False
    load_from_file: bool
        If True (default False), use the filename_list parameter to provide a list of files that will be
        selected
    """

    def __init__(
        self, filename_list: list, only_keep_filenames=False, load_from_file: bool = False
    ):
        self._load_from_file = load_from_file
        if self._load_from_file:
            logger.info(f"Loading files to 'condition' from a disk file: {filename_list}")
            files_to_reject = []
            for entry in filename_list:
                if not isinstance(entry, Path):
                    entry = Path(entry)
                if not entry.name.endswith("txt"):
                    raise custom_exceptions.InvalidConfiguration("File to load must be txt")
                with open(entry) as file:
                    files_to_reject.extend(file.readlines())

            filename_list = list(map(lambda x: x.replace("\n", ""), files_to_reject))

        self._filename_list = filename_list
        self._only_keep_filenames = only_keep_filenames
        super().__init__()

    @override
    def select_spectra(self, frame) -> Flag:
        if not self._only_keep_filenames:
            if frame.fname in self._filename_list:
                message = "Filename rejected"
                flag = USER_BLOCKED(message)
            else:
                flag = VALID
        else:
            if frame.fname in self._filename_list:
                flag = VALID
            else:
                message = "Filename rejected"
                flag = USER_BLOCKED(message)
        return flag

    @property
    def cond_info(self) -> str:
        return "Filename list {} - only keep: {}".format(
            self._filename_list, self._only_keep_filenames
        )


class SNR_condition(ConditionModel):
    """
    Reject observations based on the order-wise SNR. Compares the SNR
    of the valid orders against the minimum SNR that is provided to this
    object.

    Mostly useful for the creation of the stellar template
    """

    def __init__(self, minimum_SNR: float):
        self.minimum_SNR = minimum_SNR
        super().__init__()

        if minimum_SNR <= 0:
            raise InvalidConfiguration(f"SNR must be >= 0 ({minimum_SNR})")

    @override
    def select_spectra(self, frame):
        KW = np.asarray(frame.get_KW_value("orderwise_snrs"))
        valid_orders = list(frame.valid_orders())
        message = f"Did not pass SNR cutoff: {self.minimum_SNR:}"
        flag = VALID if np.any(KW[valid_orders] < self.minimum_SNR) else USER_BLOCKED(message)
        return flag

    @property
    def cond_info(self):
        return f"Minimum order-wise SNR of: {self.minimum_SNR}"
