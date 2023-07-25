from bindings cimport (
    Mattress,
    extract_nrow,
    extract_ncol,
    extract_sparse,
    extract_row,
    extract_column,
    free_mat,
    initialize_dense_matrix
)
from libcpp cimport bool
import numpy as np
cimport numpy as np

cdef class TatamiNumericPointer:
    cdef Mattress* ptr
    cdef object obj # held to prevent garbage collection.

    def __dealloc__(self):
        free_mat(self.ptr)

    def nrow(self):
        return extract_nrow(self.ptr);

    def ncol(self):
        return extract_ncol(self.ptr);

    def sparse(self):
        return extract_sparse(self.ptr);

    def row(self, r):
        cdef int NC = extract_ncol(self.ptr);
        cdef np.ndarray[double, ndim=1] myarr = np.empty(NC, dtype=np.float64)
        extract_row(self.ptr, r, &myarr[0]);
        return myarr

    def column(self, c):
        cdef int NR = extract_nrow(self.ptr);
        cdef np.ndarray[double, ndim=1] myarr = np.empty(NR, dtype=np.float64)
        extract_column(self.ptr, c, &myarr[0]);
        return myarr

    @staticmethod
    def from_dense(int nr, int nc, np.ndarray arr, bool byrow):
        cdef TatamiNumericPointer output = TatamiNumericPointer.__new__(TatamiNumericPointer)

        # Defining everything here because otherwise Cython complains about scoping.
        cdef np.float64_t[:,:] view_f64
        cdef np.float32_t[:,:] view_f32
        cdef np.int64_t[:,:] view_i64
        cdef np.int32_t[:,:] view_i32
        cdef np.int16_t[:,:] view_i16
        cdef np.int8_t[:,:] view_i8
        cdef np.uint64_t[:,:] view_u64
        cdef np.uint32_t[:,:] view_u32
        cdef np.uint16_t[:,:] view_u16
        cdef np.uint8_t[:,:] view_u8

        if arr.dtype == np.float64:
            view_f64 = arr
            output.ptr = initialize_dense_matrix(nr, nc, &(view_f64[0,0]), byrow)
        elif arr.dtype == np.float32:
            view_f32 = arr
            output.ptr = initialize_dense_matrix(nr, nc, &(view_f32[0,0]), byrow)
        elif arr.dtype == np.int64:
            view_i64 = arr
            output.ptr = initialize_dense_matrix(nr, nc, &(view_i64[0,0]), byrow)
        elif arr.dtype == np.int32:
            view_i32 = arr
            output.ptr = initialize_dense_matrix(nr, nc, &(view_i32[0,0]), byrow)
        elif arr.dtype == np.int16:
            view_i16 = arr
            output.ptr = initialize_dense_matrix(nr, nc, &(view_i16[0,0]), byrow)
        elif arr.dtype == np.int8:
            view_i8 = arr
            output.ptr = initialize_dense_matrix(nr, nc, &(view_i8[0,0]), byrow)
        elif arr.dtype == np.uint64:
            view_u64 = arr
            output.ptr = initialize_dense_matrix(nr, nc, &(view_u64[0,0]), byrow)
        elif arr.dtype == np.uint32:
            view_u32 = arr
            output.ptr = initialize_dense_matrix(nr, nc, &(view_u32[0,0]), byrow)
        elif arr.dtype == np.uint16:
            view_u16 = arr
            output.ptr = initialize_dense_matrix(nr, nc, &(view_u16[0,0]), byrow)
        elif arr.dtype == np.uint8:
            view_u8 = arr
            output.ptr = initialize_dense_matrix(nr, nc, &(view_u8[0,0]), byrow)
        else:
            raise TypeError("unsupported numpy type for dense matrix initialization")

        output.obj = arr
        return output
