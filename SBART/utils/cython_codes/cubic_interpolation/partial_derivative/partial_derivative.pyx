# To compile:  #warning "Using deprecated NumPy API, disable it by #defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION"
# python setup.py build_ext --inplace --force

# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np
cimport cython
from cython.parallel import prange
import numpy as np
ctypedef np.float64_t DTYPE_t

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def partial_derivative(const DTYPE_t[:,:] inv_h_mat, const DTYPE_t[:] inv_delta_wave, const int index_to_calc, DTYPE_t[:] output, const int n_threads):
    """
    COmputes partial derivatives for the error propagation
    Parameters
    ============

    inv_delta_wave : np.ndarray
        Precomputed values of 1/(x[i+1] - x[i])
    n_threads : int
        Number of threads to use 
    """

    cdef Py_ssize_t x_max = output.shape[0]

    cdef DTYPE_t k_entry = 0
    
    cdef Py_ssize_t mat_ind = 0 
    cdef Py_ssize_t k = 0

    cdef Py_ssize_t index_i = index_to_calc
    for k in prange(1, x_max-1, nogil=True, num_threads=n_threads):
        k_entry = 0 
        mat_ind = k-1
        if 1 <= k <= x_max -2 : 
            # entry that goes from j = [2, N -1]  (in python: [1, N-2]) 
            # we have to discard the last value
            k_entry = k_entry -  inv_h_mat[index_i - 1, mat_ind] * (inv_delta_wave[k] + inv_delta_wave[k-1])
        if k <= x_max - 3 : 
            # entry that goes from j = [1, N -2]  (in python: [0, N-3]) 
            # we have to discard the last two values
            k_entry = k_entry + inv_h_mat[index_i - 1, mat_ind+1] * inv_delta_wave[k]
        if k >= 2:  # entry from the Sum that starts at j == 3  (in oython, j == 2)
            # we have to discard the first two values
            k_entry = k_entry + inv_h_mat[index_i - 1, mat_ind-1] * inv_delta_wave[k-1]
        output[k] = k_entry 
    

