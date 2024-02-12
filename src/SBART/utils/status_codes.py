import ujson as json
from typing import Any, Dict, List, NoReturn, Optional, Set, Tuple, Union

import numpy as np
from loguru import logger


class Flag:
    """
    Used to represent a "state" of operation. The majority of them represents failures and/or warnings.
    """

    __slots__ = (
        "name",
        "code",
        "description",
        "extra_info",
        "is_fatal",
        "is_warning",
        "is_good_flag",
    )

    def __init__(
        self,
        name,
        value,
        description: Optional[str] = None,
        fatal_flag: bool = True,
        is_warning: bool = False,
        is_good_flag: bool = False,
        extra_info: Optional[str] = None,
    ):
        self.name = name
        self.code = value
        self.description = description
        self.extra_info = "" if extra_info is None else extra_info
        self.is_fatal = fatal_flag
        self.is_warning = is_warning
        self.is_good_flag = is_good_flag

    def __eq__(self, flag_2):
        if not isinstance(flag_2, Flag):
            return False
        return (
            (self.name == flag_2.name)
            and (self.code == flag_2.code)
            and (self.extra_info == flag_2.extra_info)
        )

    def add_extra_info(self, extra_info: str) -> NoReturn:
        self.extra_info = extra_info

    def __str__(self):
        return f"{self.name}" + min(1, len(self.extra_info)) * f": {self.extra_info}"

    def __repr__(self):
        return (
            f"Flag(name = {self.name}, "
            f"\n\tvalue = {self.code},"
            f"\n\tdescription= {self.description},"
            f"\n\tfatal_flag= {self.is_fatal},"
            f"\n\tis_warning = {self.is_warning},"
            f"\n\tis_good_flag = {self.is_good_flag}"
            f"\n\textra_info={self.extra_info}"
        )

    def __hash__(self):
        # Allow the Flag class to be hashable
        return hash(tuple((self.name, self.code, self.extra_info)))

    def __call__(self, message: str):
        new_flag = Flag(
            name=self.name,
            value=self.code,
            description=self.description,
            fatal_flag=self.is_fatal,
            is_warning=self.is_warning,
            is_good_flag=self.is_good_flag,
        )
        new_flag.add_extra_info(message)
        return new_flag

    def to_json(self) -> Dict[str, Any]:
        """
        Returns
        -------
        Dict[str, Any]
            Flag converted to a json entry, for disk storage purposes
        """
        return dict(
            name=self.name,
            value=self.code,
            description=self.description,
            fatal_flag=self.is_fatal,
            is_warning=self.is_warning,
            is_good_flag=self.is_good_flag,
            extra_info=self.extra_info,
        )

    @classmethod
    def create_from_json(cls, json_info: Dict[str, Any]):
        """
        Create a new Flag object from a json representation

        Parameters
        ----------
        json_info
            Json representation of the flag

        Returns
        -------
        flag:
            The new flag
        """
        return Flag(**json_info)


class Status:
    def __init__(self, assume_valid: bool = True):
        self._stored_flags = set()
        self._assume_valid = assume_valid
        if assume_valid:
            self.store_flag(VALID)

        self._warnings = set()

    def store_flag(self, new_flag: Flag) -> NoReturn:
        self._stored_flags.add(new_flag)

    def has_flag(self, flag) -> bool:
        return flag in self._stored_flags

    def check_if_warning_exists(self, flag):
        return flag in self._warnings

    def delete_flag(self, flag):
        try:
            self._stored_flags.remove(flag)
        except KeyError:
            logger.warning(f"Trying to remove flag that doesn't exist (flag)")

    ###
    #   Adding new flags
    ###

    def store_warning(self, warning_flag: Flag) -> NoReturn:
        if not warning_flag.is_warning:
            raise RuntimeError("Trying to store an error as a warning")

        self._warnings.add(warning_flag)

    def __add__(self, other: Flag):
        self.store_flag(other)
        return self

    def __radd__(self, other):
        return self.__add__(other)

    def __eq__(self, other):
        if other == SUCCESS:
            return self.is_valid
        raise NotImplementedError("Can only compare a status with SUCCESS")

    def reset(self) -> NoReturn:
        self._stored_flags = set()
        if self._assume_valid:
            self.store_flag(VALID)

    ###
    #   Status properties
    ###

    @property
    def all_flags(self) -> Set[Flag]:
        return self._stored_flags

    @property
    def all_warnings(self) -> Set[Flag]:
        return self._warnings

    @property
    def number_warnings(self) -> int:
        return len(self._warnings)

    @property
    def has_warnings(self) -> bool:
        return len(self._warnings) != 0

    @property
    def is_valid(self) -> bool:
        valid = True

        for flag in self._stored_flags:
            if flag.is_fatal:
                valid = False
                break
        if len(self._stored_flags) == 0:
            valid = False
            logger.warning("Status has NO stored flags (i.e. no SUCCESS flag found)")
        return valid

    ###
    #   String representation of Status
    ###
    def get_rejection_reasons(self):
        rejection = ""
        for flag in self._stored_flags:
            if not flag.is_good_flag:
                rejection += flag.name + " "
        return rejection

    def description(self, indent_level: int = 0) -> Tuple[List[str], Dict]:
        """
        string to directly place on a txt file
        Returns
        -------

        """
        skip_reasons = {"Warnings": {}, "Rejections": {}}

        indent_character = "\t"
        base_indent = indent_level * indent_character

        message = [base_indent + f"Current Status - valid = {self.is_valid}"]

        message.append(f"\n" + base_indent + indent_character + "Rejection Flags:")
        if not self.is_valid:
            for flag in self._stored_flags:
                if not flag.is_good_flag:
                    message.append(
                        "\n"
                        + base_indent
                        + 2 * indent_character
                        + f"{flag.name} : {flag.extra_info}"
                    )

                    skip_reasons["Rejections"][flag.name] = flag.description
        else:
            message.append(f"\n" + base_indent + 2 * indent_character + "No Rejection")

        if self.has_warnings:
            message.append(f"\n" + base_indent + indent_character + "Warning Flags:")
            for flag in self._warnings:
                message.append(
                    "\n" + base_indent + 2 * indent_character + f"{flag.name} : {flag.extra_info}"
                )

                skip_reasons["Warnings"][flag.name] = flag.description

        return message, skip_reasons

    def __str__(self) -> str:
        return f"Flags = {[i.name for i in self._stored_flags]}; valid = {self.is_valid}"

    def __repr__(self) -> str:
        return str(self)

    def to_json(self):
        out = {}
        out["flags"] = [i.to_json() for i in self._stored_flags]
        out["warnings"] = [i.to_json() for i in self._warnings]
        return out


class OrderStatus:
    def __init__(self, N_orders: int, frameIDs: Optional[List[int]] = None):
        N_epochs = len(frameIDs) if frameIDs is not None else frameIDs

        # we might have invalid frames -> frameIDs are not (necessarily) continuous
        self._stored_frameIDs = frameIDs if frameIDs is not None else [None]
        if N_epochs is not None:
            self._internal_mode = "matrix"
        else:
            self._internal_mode = "line"
        self._OrderStatus: np.ndarray[Status]
        self._generate_status_array(N_orders, N_epochs)

        # TODO: do we want to init to VALID ??
        self._OrderStatus += VALID

    def mimic_status(self, frameID: int, other_status) -> NoReturn:
        """
        WARNING: this does not copy the warnings!
        Parameters
        ----------
        frameID
        other_status

        Returns
        -------

        """
        order_wise_stats = other_status.get_status_from_order(all_orders=True)
        if not other_status.from_frame:
            raise RuntimeError("We can only mimic the status of a single frame at a time!")

        for order, order_status in enumerate(order_wise_stats):
            for flag in order_status.all_flags:
                self.add_flag_to_order(frameID=frameID, order=order, order_flag=flag)

    def add_flag_to_order(
        self,
        order: Union[List[int], int],
        order_flag: Flag,
        frameID: Optional[int] = None,
        all_frames: bool = False,
    ) -> NoReturn:
        if self._internal_mode == "line":
            self._OrderStatus[0, order] += order_flag
        elif self._internal_mode == "matrix":
            if all_frames:
                self._OrderStatus[:, order] = self._OrderStatus[:, order] + order_flag
            else:
                frame_index = self._stored_frameIDs.index(frameID)
                self._OrderStatus[frame_index, order] = (
                    self._OrderStatus[frame_index, order] + order_flag
                )

    def worst_rejection_flag_from_frameID(
        self, frameID: Optional[int] = None, ignore_flags=()
    ) -> Tuple[str, int]:
        flag_count = {}
        for order, status in enumerate(
            self.get_status_from_order(frameID=frameID, all_orders=True)
        ):
            for flag in status.all_flags:
                if flag in ignore_flags or flag.is_good_flag:
                    continue
                if flag not in flag_count:
                    flag_count[flag] = 0
                flag_count[flag] += 1

        max_count = 0
        max_flag = VALID
        for flag, counts in flag_count.items():
            if counts > max_count:
                max_count = counts
                max_flag = flag

        return max_flag.name, max_count

    def get_status_from_order(
        self, order: Optional[int] = None, frameID: Optional[int] = None, all_orders: bool = False
    ):
        """
        Return the status from a given set of orders for one frame

        Parameters
        ----------
        order: Optional[int]
            If None, return all orders when the all_orders flag is set to True
        frameID: Optional[int]
            If None, return all orders when the all_orders flag is set to True. Optional when this is used inside the
            from instances of Frame obects
        all_orders

        Returns
        -------

        """
        if self._internal_mode == "line":
            if all_orders:
                return self._OrderStatus[0]
            return self._OrderStatus[0, order]

        elif self._internal_mode == "matrix":
            if frameID is None:
                raise RuntimeError("When we have multiple observations we need a frameID")

            epoch = self._stored_frameIDs.index(frameID)
            if all_orders:
                return self._OrderStatus[epoch, :]
            else:
                return self._OrderStatus[epoch, order]

    @property
    def from_frame(self) -> bool:
        return self._internal_mode == "line"

    @property
    def bad_orders(self) -> Set[int]:
        if self._internal_mode == "matrix":
            raise RuntimeError(
                "bad_orders is only defined at the Frame level. Use the common_bad_orders property"
            )
        bad_orders = set()
        for order, order_stat in enumerate(self._OrderStatus[0]):
            if not order_stat.is_valid:
                bad_orders.add(order)
        return bad_orders

    @property
    def common_bad_orders(self):
        """
        Find the common set of spectral orders that is rejected in all epochs
        Returns
        -------

        """
        return np.unique(np.where(self._OrderStatus != SUCCESS)[1])

    def as_boolean(self):
        return np.where(self._OrderStatus == SUCCESS, True, False)

    def _generate_status_array(self, N_orders, N_epochs):
        if self._internal_mode == "line":
            self._OrderStatus = np.empty((1, N_orders), dtype=Status)
        elif self._internal_mode == "matrix":
            self._OrderStatus = np.empty((N_epochs, N_orders), dtype=Status)
        else:
            raise RuntimeError("Internal mode not recognized")
        if N_epochs is None:
            N_epochs = 1

        for epoch in range(N_epochs):
            for order in range(N_orders):
                self._OrderStatus[epoch][order] = Status()

    ###
    #   Status string representation
    ###

    def description(
        self,
        indent_level: int = 0,
        frameID: Optional[int] = None,
        include_header: bool = True,
        include_footer: bool = True,
    ) -> Tuple[List[str], Dict]:
        skip_reasons = {"Warnings": {}, "Rejections": {}}

        indent_character = "\t"
        base_indent = indent_level * indent_character
        message = []

        if frameID is not None:
            ID_to_process = [frameID]
        else:
            ID_to_process = self._stored_frameIDs

        for frameID in ID_to_process:
            fatal_flag_dict = {}
            warning_flag_dict = {}

            if include_header:
                message.append(f"\n{base_indent}FrameID:{frameID}")

            for order_number, status in enumerate(
                self.get_status_from_order(frameID=frameID, all_orders=True)
            ):
                for flag in status.all_flags:
                    if not flag.is_good_flag:
                        if flag.name not in fatal_flag_dict:
                            fatal_flag_dict[flag.name] = []
                        fatal_flag_dict[flag.name].append(order_number)
                        skip_reasons["Rejections"][flag.name] = flag.description

                for flag in status.all_warnings:
                    if flag.name not in warning_flag_dict:
                        warning_flag_dict[flag.name] = []
                    warning_flag_dict[flag.name].append(order_number)
                    skip_reasons["Warnings"][flag.name] = flag.description

            message.append(
                "\n"
                + base_indent
                + indent_character
                + "Order Rejections (Worst - {} -> N = {}):".format(
                    *self.worst_rejection_flag_from_frameID(frameID)
                )
            )

            for key, orders in fatal_flag_dict.items():
                message.append(
                    "\n"
                    + base_indent
                    + 2 * indent_character
                    + f"{key} (N = {len(orders)}): {orders}"
                )

            if len(warning_flag_dict) != 0:
                message.append("\n" + base_indent + indent_character + "Order Warnings:")
                for key, orders in warning_flag_dict.items():
                    message.append("\n" + base_indent + 2 * indent_character + f"{key}: {orders}")
            message.append("\n")

        if include_footer:
            message.append("\n\n" + base_indent + "Rejection reasons:")
            for key, descr in skip_reasons["Rejections"].items():
                message.append("\n" + base_indent + indent_character + f"{key}: {descr}")

            if len(skip_reasons["Warnings"]) != 0:
                message.append("\n" + base_indent + "Warnings:")
                for key, descr in skip_reasons["Warnings"].items():
                    message.append("\n" + base_indent + indent_character + f"{key}: {descr}")
        return message, skip_reasons

    def __str__(self):
        return str(self._OrderStatus)

    def store_as_json(self, storage_path):
        """
        Directly stores to a single file all information inside this class
        Parameters
        ----------
        storage_path

        Returns
        -------

        """
        with open(storage_path, mode="w") as file:
            json.dump(self.to_json(), file, indent=4)

    def to_json(self):
        out = {
            "general_confs": {
                "frameIDs": self._stored_frameIDs if self._internal_mode == "matrix" else None,
                "N_orders": self._OrderStatus.shape[1],
            }
        }

        for epoch in range(self._OrderStatus.shape[0]):
            epoch_key = f"frameID::{self._stored_frameIDs[epoch]}"
            out[epoch_key] = {}
            for order in range(self._OrderStatus.shape[1]):
                out[epoch_key][order] = self._OrderStatus[epoch, order].to_json()
        return out

    @classmethod
    def load_from_json(cls, storage_path):
        with open(storage_path) as file:
            json_info = json.load(file)

        N_orders = json_info["general_confs"]["N_orders"]
        frameIDs = json_info["general_confs"]["frameIDs"]
        new_stats = OrderStatus(N_orders, frameIDs)

        for frameID_string, values in json_info.items():
            if frameID_string in ["general_confs"]:
                continue

            frameID = frameID_string.split("::")[1]
            for order in range(N_orders):
                for flag_dict in values[str(order)]["flags"]:
                    loaded_frameID = None if frameIDs is None else int(frameID)
                    new_stats.add_flag_to_order(
                        order_flag=Flag.create_from_json(flag_dict),
                        order=order,
                        frameID=loaded_frameID,
                    )
                for flag_dict in values[str(order)]["warnings"]:
                    new_stats.add_flag_to_order(
                        order_flag=Flag.create_from_json(flag_dict),
                        order=order,
                        frameID=int(frameID),
                    )
        return new_stats


###########################################################
#
# General codes
#
###########################################################

INTERNAL_ERROR = Flag("INTERNAL_ERROR", "D")
SUCCESS = Flag("SUCCESS", 0, fatal_flag=False, is_good_flag=True)
DISK_LOADED_DATA = Flag("LOADED", 0, fatal_flag=False, is_good_flag=True)
SHUTDOWN = Flag("SHUTDOWN", "S")

MANDATORY_KW_FLAG = Flag("Mandatory KW", "MKW")
###########################################################
#
# Codes for the Frames
#
###########################################################

VALID = Flag("VALID", value="V", fatal_flag=False, is_good_flag=True)
WARNING = Flag("WARNING", value="W", fatal_flag=False, is_good_flag=False, is_warning=True)

SIGMA_CLIP_REJECTION = Flag("SIGMA CLIP", value="SC")
USER_BLOCKED = Flag("USER_BLOCKED", value="U")
FATAL_KW = Flag("FATAL_KW", value="F")
KW_WARNING = Flag("KW_WARNING", value="KW_W", is_warning=True)

MISSING_FILE = Flag("MISS_FILE", value="M")
NO_VALID_ORDERS = Flag("NO_VALID_ORDERS", value="NO")

MISSING_EXTERNAL_DATA = Flag("MISS_EXTERNAL_LOAD", value="S")
MISSING_SHAQ_RVS = Flag("MISS_SHAQ_RV", value="S")
LOADING_EXTERNAL_DATA = Flag("LOADING_EXTERNAL", value="LS", fatal_flag=False)

CREATING_MODEL = Flag("CREATING MODEL", value="CM", fatal_flag=False)
FAILED_MODEL_CREATION = Flag("FAILED MODEL CREATION", value="FCM", fatal_flag=True)
###########################################################
#
# Codes for the RV routines to use
#
###########################################################

WORKER_OFF = Flag("WORKER_OFF", "O")
IDLE_WORKER = Flag("SUCCESS", "I")
ACTIVE_WORKER = Flag("SUCCESS", "A")

# Positive codes for problems with the orders
LOW_SNR = Flag("LOW_SNR", 5, "SNR under the user-set threshold")
MASSIVE_RV_PRIOR = Flag(
    "MASSIVE_RV_PRIOR", 4, "Too little spectra left after accountinf for RV window"
)
BAD_TEMPLATE = Flag("BAD_TEMPLATE", 3, "Could not create stellar template for given order")
HIGH_CONTAMINATION = Flag(
    "HIGH_CONTAMINATION", 2, "Too many points removed due to masks + tellurics"
)
ORDER_SKIP = Flag("ORDER_SKIP", 1, "Order was skipped")

# negative values for errors in the RV
WORKER_ERROR = Flag("WORKER_ERROR", -1)
CONVERGENCE_FAIL = Flag("CONVERGENCE_FAIL", -2)
MAX_ITER = Flag("MAX ITERATIONS", -3)

###########################################################
#
# Codes for removal of points from the masks  -> uint16 is the max size for the status codes !!!!!!!!!
#
###########################################################

QUAL_DATA = Flag("QUAL_DATA", 1, " Qual data different than zero")  # qual data different than zero
ERROR_THRESHOLD = Flag(
    "ERROR_THRESHOLD", 2, "Error over specified threshold"
)  # error threshold over the selected threshold
INTERPOLATION = Flag(
    "INTERPOLATION", 4, "Removed due to interpolation"
)  # removed due to interpolation constraints
TELLURIC = Flag("TELLURIC", 8, "Telluric feature")  # classified as telluric feature,
MISSING_DATA = Flag(
    "MISSING_DATA", 16, "Missing spectral data in the pixel"
)  # data is missing in the given points,
SPECTRAL_MISMATCH = Flag(
    "SPECTRAL_MISMATCH", 32, "Removed due to outlier routine"
)  # mismatch between the template and the spectra
SATURATION = Flag(
    "SATURATION", 64, "Saturated Pixel"
)  # Saturation of the detector; Only used by HARPS
NAN_DATA = Flag("NaN_Pixel", 128, "Nan Value")
ACTIVITY_LINE = Flag("ACTIVITY_INDICATOR", 256)  # this spectral regions belongs to a marked line

NON_COMMON_WAVELENGTH = Flag(
    "NON_COMMON_WAVELENGTH", 512
)  # this spectral regions belongs to a marked line
MULTIPLE_REASONS = Flag("MULTIPLE", 100)  # flagged by more than one reason

if __name__ == "__main__":
    x = MULTIPLE_REASONS("ewoquiqweoui")
    print(x)
    y = OrderStatus(4, [1, 2, 3, 4, 5, 6])

    y.add_flag_to_order(1, SUCCESS, all_frames=True)
    y.add_flag_to_order(1, LOW_SNR, all_frames=True)
    y.add_flag_to_order(order=1, order_flag=ORDER_SKIP, frameID=1)
    y.add_flag_to_order(order=2, order_flag=ORDER_SKIP, frameID=1)

    print("---*-")
    print(
        "".join(
            y.description(indent_level=1, frameID=1, include_header=False, include_footer=False)[0]
        )
    )

    # print("".join(y.get_status_from_order(1, 1).description(indent_level=1)))

    x = OrderStatus(N_orders=4)
    print("---")
    print(y.get_status_from_order(3, frameID=1, all_orders=True))
    print(y.worst_rejection_flag_from_frameID(1))
    print("---")
    input()
    print(x.bad_orders)

    x.add_flag_to_order(3, SPECTRAL_MISMATCH)
    y.mimic_status(1, x)

    print(y.get_status_from_order(3, frameID=1))

    print("finit")

    h = ORDER_SKIP("TESTE")
    print(h)

    print(ORDER_SKIP)

    x = KW_WARNING("teste")

    print(x, "--", x.extra_info)
