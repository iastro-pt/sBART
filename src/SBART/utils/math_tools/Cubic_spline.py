import time

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from scipy.misc import derivative

from SBART.utils import custom_exceptions

try:
    from SBART.utils.cython_codes.cubic_interpolation import (
        partial_derivative,
        second_derivative,
        tridiagonal_inverter,
    )

    CYTHON_UNAVAILABLE = False
except ImportError:
    logger.critical(
        "Cython interface is not found, please make sure that the installation went smoothly"
    )
    CYTHON_UNAVAILABLE = True

# np.seterr(all='raise')


class CustomCubicSpline:
    def __init__(
        self,
        old_wavelengths,
        original_data,
        original_errors,
        ignore_covariances=True,
        n_threads=4,
    ):
        if CYTHON_UNAVAILABLE:
            raise custom_exceptions.InternalError("Cython interface is not installed")
        self.old_wavelengths = old_wavelengths
        self.original_data = original_data
        self.original_errors = original_errors
        self.data_size = original_data.size

        self.ignore_covariances = ignore_covariances

        if ignore_covariances:
            self.cov_matrix = original_errors**2
        else:
            self.cov_matrix = (
                np.zeros((original_data.size, original_data.size)) + original_errors**2
            )

        self._cached_h = False
        self._inv_h = []
        self._partials = {}
        self.n_threads = n_threads
        self.inv_delta_wave = 1 / np.diff(old_wavelengths)

        # using memory layout C as the "second index" is the one that changes fastest
        self.cached_ones = np.ones(
            (self.data_size - 2, self.data_size - 2), dtype=np.float64, order="C"
        )

    def compute_h_matrix(self):
        """
        Create the H matrix
        """

        if self._cached_h:
            return self._inv_h

        x = self.old_wavelengths

        diag_c = (np.roll(x, -2) - x)[0:-2] / 3
        # we remove the first element as we are interested in x_3 - x_2
        # the last one is removed for the same reason
        # the one before the last is removed as the second diagonal only goes up to N-1
        diag_r = (np.roll(x, -1) - x)[1:-2] / 6
        self._cached_h = True

        # avoid creating a new numpy array at each iteration!
        output = self.cached_ones
        output[:] = 1

        tridiagonal_inverter(diag_c, diag_r, output, n_threads=self.n_threads)
        # print("CYTHON vs Numpy EQUAL: ", np.allclose(np.linalg.inv(h), output))
        self._inv_h = output
        return self._inv_h

    def compute_partial(self, index_i):
        try:
            return self._partials[index_i]
        except KeyError:
            pass

        if index_i in (0, self.data_size - 1):
            # both edges have y'' equal to zero
            partials = np.zeros(self.data_size)
            self._partials[index_i] = partials
            return partials

        partials = np.empty(self.data_size)

        # in the expressions the array starts at one, instead of zero !!
        inv_h_matrix = self.compute_h_matrix()

        # the two edges have a value of zero!
        partials[0] = 0
        partials[-1] = 0

        partial_derivative(inv_h_matrix, self.inv_delta_wave, index_i, partials, self.n_threads)

        # avoid underflows
        partials[np.where(np.abs(partials) < 1e-100)] = 0
        self._partials[index_i] = partials
        return partials

    def _U_entry(self, mode, first_0, second_0):
        """
        mode == '22':
            calculate u(y''_first_0, y''_second_0)
        if mode == '20':
            calculate u(y''_first_0, y_second_0)

        """
        partial_first = self.compute_partial(first_0)
        if mode == "22":  # calculate entry for cov between two 2nd derivatives
            partial_second = self.compute_partial(second_0)
            if self.ignore_covariances:
                first_part = np.multiply(partial_first, self.cov_matrix)
            else:
                first_part = np.matmul(partial_first, self.cov_matrix)
            return np.matmul(first_part, np.transpose(partial_second))
        elif mode == "20":  # entry for 2nd derivative and data
            second = np.zeros(partial_first.shape)
            second[second_0] = 1

            if self.ignore_covariances:
                first_part = np.multiply(partial_first, self.cov_matrix)
                # print(np.min(first_part), np.max(first_part), partial_first.dtype, self.cov_matrix.dtype, first_part.dtype)

            else:
                first_part = np.matmul(partial_first, self.cov_matrix)
            return np.matmul(first_part, second)

    def build_U_matrix(self, index_i_0, index_j_0):
        """
        Create the "U" matrix for the covariance estimation; Only computes the relevant quadrant
        """

        cov_mat = self.cov_matrix

        i = index_i_0
        j = index_j_0

        if self.ignore_covariances:
            cov_i_i = cov_mat[i]
            cov_i_j1 = 0
            cov_i1_j = 0
            cov_j_j = cov_mat[i + 1]
        else:
            cov_i_i = cov_mat[i, j]
            cov_i_j1 = cov_mat[i, j + 1]
            cov_i1_j = cov_mat[i + 1, j]
            cov_j_j = cov_mat[i + 1, j + 1]

        # we are only evaluating the diagonal, i == j
        u_03 = self._U_entry("20", j + 1, i)
        u_01 = self._U_entry("20", j, i)
        u_13 = self._U_entry("20", j + 1, i + 1)

        u_23 = self._U_entry("22", i, j + 1)

        U = np.array(
            [
                [cov_i_i, cov_i_j1, u_01, u_03],
                [cov_i1_j, cov_j_j, u_03, u_13],
                [u_01, u_03, self._U_entry("22", i, j), u_23],
                [u_03, u_13, u_23, self._U_entry("22", i + 1, j + 1)],
            ]
        )

        return U

    def compute_second_derivative(self):
        old_wavelengths = self.old_wavelengths
        original_data = self.original_data
        inv_h_matrix = self.compute_h_matrix()

        second_deriv_vals = np.zeros(original_data.shape, dtype=np.float64)

        second_derivative(
            np.diff(old_wavelengths),
            np.diff(original_data),
            inv_h_matrix,
            second_deriv_vals,
            self.n_threads,
        )

        return second_deriv_vals

    def interpolate(self, new_wavelengths):
        """
        Interpolate the data to the new wavelengths
        """
        old_wavelengths = self.old_wavelengths
        original_data = self.original_data

        number_points = old_wavelengths.size
        new_data = np.empty(new_wavelengths.size)
        new_errors = np.empty(new_wavelengths.size)

        start_index = 0

        index_0 = 0
        index_1 = 0
        inv_h = self.compute_h_matrix()
        self.second_derivatives = self.compute_second_derivative()

        for new_index, new_wave_position in enumerate(new_wavelengths):
            found_exact_match = False

            for index in range(start_index, number_points):
                if old_wavelengths[index] == new_wave_position:
                    interpolated_value = old_wavelengths[index]
                    propagated_error = self.original_errors[index]
                    found_exact_match = True
                    break
                try:
                    if (
                        old_wavelengths[index] < new_wave_position
                        and old_wavelengths[index + 1] >= new_wave_position
                    ):
                        index_0 = index
                        index_1 = index + 1
                        # wavelengths are sorted; can start searching after the i
                        start_index = index_0
                        break
                except Exception as e:
                    print(new_wave_position, old_wavelengths[-1])
                    raise e

            if not found_exact_match:
                x_0 = old_wavelengths[index_0]
                x_1 = old_wavelengths[index_1]
                delta_x = x_1 - x_0
                A = (x_1 - new_wave_position) / delta_x
                B = (new_wave_position - x_0) / delta_x
                C = (1 / 6) * (A**3 - A) * (delta_x) ** 2
                D = (1 / 6) * (B**3 - B) * (delta_x) ** 2

                interpolated_value = (
                    A * original_data[index_0]
                    + B * original_data[index_1]
                    + C * self.second_derivatives[index_0]
                    + D * self.second_derivatives[index_1]
                )

                first = [A, B, C, D]

                U = self.build_U_matrix(index_0, index_0)
                second = np.transpose(first)
                propagated_error = np.matmul(np.matmul(first, U), second)

            new_data[new_index] = interpolated_value
            new_errors[new_index] = abs(propagated_error)

        return new_data, np.sqrt(new_errors)
