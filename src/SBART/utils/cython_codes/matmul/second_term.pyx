# To compile:  #warning "Using deprecated NumPy API, disable it by #defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION"
# python setup.py build_ext --inplace --force

# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np
cimport cython
from cython.parallel import prange


ctypedef np.float64_t DTYPE_t

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def matrix_dot(const DTYPE_t[:,:] H, const DTYPE_t[:] k, const DTYPE_t[::1] data, DTYPE_t a_00, DTYPE_t a_01, DTYPE_t a_11):
    
    cdef Py_ssize_t x_max = H.shape[1]
    
    cdef Py_ssize_t i = 0 
    cdef Py_ssize_t j = 0

    cdef DTYPE_t value = 0

   

    for j in prange(x_max, nogil=True, num_threads=4):
    #for j in range(x_max):

        for i in range(j,x_max):
            # it is faster to do everything and follow "column order" than it is
            # to jump around and set result_view[i,j] and result_view[j,i]
            
            value += (H[0,j]*(a_00*H[0,i]+H[1,i]*a_01) + H[1,j]*(H[0,i]*a_01 + H[1,i]*a_11))/(k[i]*k[j]) * data[j] * data[i] * ( 1 + min(1,i-j))
    
    return 0.5*value  