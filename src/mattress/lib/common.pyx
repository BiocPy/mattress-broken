cimport common
from cython.operator cimport dereference
cimport numpy as np

cdef class TatamiNumericPointer:
    def __init__(self, ptr, obj):
        self.ptr = ptr
        self.obj = obj

    def nrow(self):
        return dereference(self.ptr).nrow();

    def ncol(self):
        return dereference(self.ptr).ncol();

    def sparse(self):
        return dereference(self.ptr).sparse();

    def row(self, r):
        cdef int NR = dereference(ptr).nrow();
        cdef np.ndarray[np.double, ndim=1] myarr = np.empty(NR, dtype=np.int32)
        dereference(dereference(self.ptr).dense_row()).fetch_copy(r, &myarr[0]);
        return myarr

    def column(self, r):
        cdef int NR = dereference(ptr).nrow();
        cdef np.ndarray[np.double, ndim=1] myarr = np.empty(NR, dtype=np.int32)
        dereference(dereference(self.ptr).dense_row()).fetch_copy(r, &myarr[0]);
        return myarra
