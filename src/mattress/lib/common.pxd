from libc.stdint cimport uintptr_t
from libcpp cimport bool

cdef class TatamiNumericPointer:
    cdef uintptr_t ptr
    cdef object obj # held to prevent garbage collection.

cdef extern from "cpp/common.cpp":
    int extract_nrow(uintptr_t)
    int extract_ncol(uintptr_t)
    bool extract_sparse(uintptr_t)
    void extract_row(uintptr_t, int, double*);
    void extract_column(uintptr_t, int, double*);
