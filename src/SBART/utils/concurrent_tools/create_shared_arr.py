from multiprocessing import shared_memory

import numpy as np


def create_shared_array(data):
    """
    Creates a numpy array using a shared memory block as the buffer.

    """

    array_size = data.shape
    buffer_info = [
        shared_memory.SharedMemory(create=True, size=data.nbytes),
        array_size,
        data.dtype,
    ]
    shared_data = np.ndarray(array_size, dtype=data.dtype, buffer=buffer_info[0].buf)
    shared_data[:] = data[:]
    return buffer_info, shared_data
