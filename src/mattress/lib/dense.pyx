cimport common
from dense cimport initialize_dense_matrix

cimport numpy as np
import numpy as np

cdef initialize_dense_matrix_exported(int nr, int nc, np.ndarray arr, bool byrow):
    if arr.dtype == np.int32:
        cdef np.int32[:] view = arr
        return TatamiNumericPointer(initialize_dense_matrix(nr, nc, &view[0], byrow), arr)
    else:
        print("AARGH")
