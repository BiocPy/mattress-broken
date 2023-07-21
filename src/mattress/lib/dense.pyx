cimport common
cimport dense
from libcpp cimport bool
cimport numpy as np
import numpy as np

cdef initialize_dense_matrix_exported(int nr, int nc, np.ndarray arr, bool byrow):
    cdef common.TatamiNumericPointer output;

    # Defining everything here because otherwise Cython complains about scoping.
    cdef np.float64_t[:] view_f64
    cdef np.float32_t[:] view_f32
    cdef np.int64_t[:] view_i64
    cdef np.int32_t[:] view_i32
    cdef np.int16_t[:] view_i16
    cdef np.int8_t[:] view_i8
    cdef np.uint64_t[:] view_u64
    cdef np.uint32_t[:] view_u32
    cdef np.uint16_t[:] view_u16
    cdef np.uint8_t[:] view_u8

    if arr.dtype == np.float64:
        view_f64 = arr
        output = common.TatamiNumericPointer(dense.initialize_dense_matrix(nr, nc, &(view_f64[0]), byrow), arr)

    return output

