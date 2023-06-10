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
def second_derivative(const DTYPE_t[:] delta_wave, const DTYPE_t[:] delta_data, DTYPE_t[:,:] inv_h_matrix, DTYPE_t[:] output, const int n_threads):
    """
    Calculate the second derivative of the input data using natural boundary conditions
    Parameters
    ============
    diag_c : np.ndarray
        The central diagonal of the matrix 
    diag_offset : np.ndarray
        The non-central diagonal
    output : np.ndarray
        Array to which the inverted matrix will be written to 
    n_threads : int
        Number of threads to use 
    """
    cdef Py_ssize_t x_max = output.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] coefs = np.empty(x_max-2)

    cdef Py_ssize_t row = 0 
    cdef Py_ssize_t column = 0
    cdef Py_ssize_t int_i = 0

    cdef DTYPE_t entry = 0
    
    cdef Py_ssize_t j = 0
    cdef Py_ssize_t j_1 = 0


    for j in range(1, x_max-1):
        j_1 = j - 1 
        coefs[j_1] = delta_data[j]/delta_wave[j] - delta_data[j_1]/delta_wave[j_1]

    # natural boundary conditions
    output[0] = 0 
    output[x_max-1] = 0
    for row in prange(1, x_max-1, nogil=True, num_threads=n_threads):
        entry = 0
        int_i = row -1 
        for column in range(0, x_max-2):
            #print(row, column)
            entry = entry + coefs[column] *  inv_h_matrix[int_i,column]
        output[row] = entry