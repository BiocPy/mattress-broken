from libcpp cimport bool
from libc.stdint cimport (uintptr_t, int64_t, uint64_t, int32_t, uint32_t, int16_t, uint16_t, int8_t, uint8_t)

cdef extern from "cpp/dense.cpp":
    uintptr_t initialize_dense_matrix(int, int, float*, bool)
    uintptr_t initialize_dense_matrix(int, int, double*, bool)
    uintptr_t initialize_dense_matrix(int, int, int64_t*, bool)
    uintptr_t initialize_dense_matrix(int, int, uint64_t*, bool)
    uintptr_t initialize_dense_matrix(int, int, int32_t*, bool)
    uintptr_t initialize_dense_matrix(int, int, uint32_t*, bool)
    uintptr_t initialize_dense_matrix(int, int, int16_t*, bool)
    uintptr_t initialize_dense_matrix(int, int, uint16_t*, bool)
    uintptr_t initialize_dense_matrix(int, int, int8_t*, bool)
    uintptr_t initialize_dense_matrix(int, int, uint8_t*, bool)
