cimport common
from libc.stdint cimport uintptr_t
from cython.operator cimport dereference
cimport numpy as np
import numpy as np

cdef class TatamiNumericPointer:
    def __init__(self, uintptr_t ptr, obj):
        self.ptr = ptr
        self.obj = obj

    def nrow(self):
        return common.extract_nrow(self.ptr);

    def ncol(self):
        return common.extract_ncol(self.ptr);

    def sparse(self):
        return common.extract_sparse(self.ptr);

    def row(self, r):
        cdef int NC = common.extract_ncol(self.ptr);
        cdef np.ndarray[double, ndim=1] myarr = np.empty(NC, dtype=np.float64)
        common.extract_row(self.ptr, r, &myarr[0]);
        return myarr

    def column(self, c):
        cdef int NR = common.extract_nrow(self.ptr);
        cdef np.ndarray[double, ndim=1] myarr = np.empty(NR, dtype=np.float64)
        common.extract_column(self.ptr, c, &myarr[0]);
        return myarr
