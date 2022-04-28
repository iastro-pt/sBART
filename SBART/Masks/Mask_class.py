from multiprocessing import shared_memory

import numpy as np
from loguru import logger

from SBART.utils.concurrent_tools.create_shared_arr import create_shared_array
from SBART.utils.status_codes import MULTIPLE_REASONS


class Mask:
    def __init__(self, initial_mask, mask_type="normal"):

        self.mask_type = mask_type
        self._internal_mask = initial_mask
        self._outdated_cache = True

        self._current_types = {MULTIPLE_REASONS.name: MULTIPLE_REASONS.code}

        self._cached_mask = self.get_custom_mask([])
        self.shm = {}

    def override_entire_mask(self, new_mask):
        self._internal_mask = new_mask

    def get_submask(self, include):
        """
        Only returns one of the masked reasons
        """

        exclude = list(self._current_types.keys())
        try:
            exclude.remove(include.name)
        except ValueError:
            pass

        return self.get_custom_mask(exclude)

    def get_custom_mask(self, exclude=()):
        """
        Retrives a binary mask where the reasons in "exclude" are not acocunted for
        The masked points are set as True

        PARAMETERS
        ===============
        exclude: list
            List of objects from Flag type or names of flags. If a given flag (or name) does not exist, a warning is self.logger.infoed and
            the process continues
        """

        if self.mask_type == "binary":
            if len(exclude) != 0:
                self.logger.critical(f"Binary mask does not allow to remove flags. {exclude=}")
            return self._internal_mask

        if not exclude and not self._outdated_cache:
            return self._cached_mask

        non_interest_points = [0]  # the mask is not sert with a  value of zero
        for exclusion in exclude:
            try:
                flag_name = exclusion if isinstance(exclusion, str) else exclusion.name

                if (
                    flag_name == MULTIPLE_REASONS.name
                ):  # always block the points that are marked by multiple reasons
                    continue

                non_interest_points.append(self._current_types[flag_name])
            except KeyError as e:
                self.logger.warning(f"Key {e} does not exist in mask")

        custom_mask = np.zeros(self._internal_mask.shape, dtype=np.bool)

        custom_mask[np.where(np.isin(self._internal_mask, non_interest_points) == 0)] = True
        custom_mask.flags.writeable = False

        if len(exclude) == 0:
            self._cached_mask = custom_mask
            self._outdated_cache = False
        return custom_mask

    def add_to_mask(self, epoch, new_masked, mask_type):
        """
        Adds the True points from new_masked to the existing mask

        Parameters
        ----------
        epoch : [type]
            [description]
        new_masked : [type]
            [description]
        """

        self._outdated_cache = True
        points_to_add = np.where(new_masked == True)

        new_masked_region = self._internal_mask[epoch][points_to_add]
        indexes = np.where(new_masked_region != 0)
        new_masked_region[indexes] = MULTIPLE_REASONS.code
        new_masked_region[np.where(new_masked_region == 0)] = mask_type.code

        if mask_type.name not in self._current_types:
            self._current_types[mask_type.name] = mask_type.code

        self._internal_mask[epoch][points_to_add] = new_masked_region

    def add_indexes_to_mask(self, indexes, mask_type):
        self._outdated_cache = True

        if mask_type.name not in self._current_types:
            self._current_types[mask_type.name] = mask_type.code

        self._internal_mask[indexes] = np.where(
            self._internal_mask[indexes] == 0, mask_type.code, MULTIPLE_REASONS.code
        )[:]

    def add_indexes_to_mask_order(self, order, indexes, mask_type):
        self._outdated_cache = True

        if mask_type.name not in self._current_types:
            self._current_types[mask_type.name] = mask_type.code

        self._internal_mask[order][indexes] = np.where(
            self._internal_mask[order][indexes] == 0,
            mask_type.code,
            MULTIPLE_REASONS.code,
        )[:]

    def clean_mask(self, type_to_clean):
        interest_points = []
        for to_clean in type_to_clean:
            interest_points.append(self._current_types[to_clean])
        self._internal_mask[np.where(np.isin(self._internal_mask, interest_points) == 0)] = 0

    def compute_statistics(self, detailed=False):

        if self.mask_type == "binary":
            sum_mask = np.sum(self.get_custom_mask([]), axis=1)
            string = "\tMinimum masked points (order {}): {}\n\tMaximum masked points (order {}): {}\n\tmedian (across all orders) masked points: {}"
            argmin = np.argmin(sum_mask)
            argmax = np.argmax(sum_mask)

            logger.info(
                "Statistics from the template creation: \n"
                + string.format(
                    argmin,
                    np.min(sum_mask),
                    argmax,
                    np.max(sum_mask),
                    np.median(sum_mask),
                )
            )

        else:
            detailed = True
            logger.info("Computing statistics for the entire dataset")
            global_points = []

            full_mask = self._internal_mask

            for index, sum_mask in enumerate(full_mask):

                if detailed:
                    logger.info("\n\tEpoch  # {}".format(index))

                    logger.info(
                        "\n\t\t Reason \t Min (non-0) points  (order) \t Max points (order) \t Median (across orders)\n"
                    )
                    logger.info("\t\t" + "-" * 100)

                    all_ords = {key: [] for key in self._current_types}

                    for order in sum_mask:
                        different_vals, counts = np.unique(order, return_counts=True)
                        for key, key_val in self._current_types.items():
                            all_ords[key].append(np.nan)
                        for ind, value in enumerate(different_vals):
                            for key, key_val in self._current_types.items():
                                if value == key_val:
                                    all_ords[key][-1] = counts[ind]
                                    break

                    global_points.append(0)
                    for key in self._current_types:

                        caused_by_reason = all_ords[key]
                        string = "\t\t{}\t\t(order {}): {}\t\t(order {}): {} \t {}"

                        try:
                            argmin = np.nanargmin(caused_by_reason)
                            min_val = caused_by_reason[argmin]
                        except ValueError:
                            argmin = np.nan
                            min_val = np.nan

                        try:
                            argmax = np.nanargmax(caused_by_reason)
                            max_val = caused_by_reason[argmax]
                        except ValueError:
                            argmax = np.nan
                            max_val = np.nan

                        logger.info(
                            string.format(
                                key,
                                argmin,
                                min_val,
                                argmax,
                                max_val,
                                np.median(caused_by_reason),
                            )
                        )
                        global_points[-1] += np.sum(caused_by_reason)
                else:
                    number_masked = len(np.where(sum_mask != 0)[0])
                    global_points.append(number_masked)

            logger.info("\n\tGlobal Summary:\n")
            logger.info("\tMinimum epoch \t\t Maximum epoch \t \t Median points per epoch\n")
            string = "\t(epoch {}): {}\t (epoch {}): {} \t {}\n"
            argmin = np.argmin(global_points)
            argmax = np.argmax(global_points)

            logger.info(
                string.format(
                    argmin,
                    global_points[argmin],
                    argmax,
                    global_points[argmax],
                    np.median(global_points),
                )
            )

    @property
    def masked_count(self):
        return len(np.where(self._internal_mask != 0)[0])

    def close_buffers(self):
        for mem_block in self.shm.values():
            mem_block[0].close()
            mem_block[0].unlink()

        self.shm = {}
