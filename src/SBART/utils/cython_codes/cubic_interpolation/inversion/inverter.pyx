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
def diag_inverter(const DTYPE_t[:] diag_c, const DTYPE_t[:] diag_offset, DTYPE_t[:,:] output, const int n_threads):
    """
    Inverts a symmetric tridiagonal matrix

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
    cdef Py_ssize_t x_max = diag_c.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] C_VALUES = np.empty(x_max)
    cdef np.ndarray[DTYPE_t, ndim=1] inv_C_VALUES = np.empty((x_max))    

    cdef Py_ssize_t row = 0 
    cdef Py_ssize_t column = 0
    cdef DTYPE_t temp = 0
    
    cdef Py_ssize_t i = 0
    cdef DTYPE_t val = 1
    cdef DTYPE_t previous = 1

    C_VALUES[0] = diag_c[0] 
    inv_C_VALUES[0] = 1/C_VALUES[0]

    for i in range(1, x_max):
        C_VALUES[i] =  diag_c[i] - diag_offset[i-1]**2/C_VALUES[i-1]
        inv_C_VALUES[i] = 1/C_VALUES[i]

    for row in prange(x_max, nogil=True, num_threads=n_threads):
    #for row in range(x_max):
        previous = 1
        for column in range(row+1, x_max): 
            # non diagonal terms, without accounting for w_ii
            val = previous * diag_offset[column-1]*inv_C_VALUES[column-1] * (-1)
            previous = val
            output[row, column] = val
            output[column, row] = val

        output[row, row] = inv_C_VALUES[row]
        if row != x_max - 1: # compute main diagonal values!
            val = 0
            for i in range(row+1, x_max):
                val = val + (inv_C_VALUES[i]) * output[row, i]**2
            #output[row, row] = output[row, row] + np.dot(1/C_VALUES[row+1:], np.square(output[row, row+1:]))
            output[row, row] = output[row, row] + val

    for row in prange(x_max, nogil=True, num_threads=n_threads):
        # add the diagonal value to the non-diagonal values
        # this is done this way to avoid problems when re-using output[i,j] to store 
        # data until it is needed further ahead
        for column in range(row+1, x_max):
            output[row, column] =  output[row, column] * output[column,column]
            output[column, row] = output[column, row] * output[column,column]