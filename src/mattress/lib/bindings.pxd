from libcpp cimport bool

cdef extern from "cpp/Mattress.h":
    ctypedef struct Mattress:
        pass

cdef extern from "cpp/common.cpp":
    int extract_nrow(Mattress*)
    int extract_ncol(Mattress*)
    bool extract_sparse(Mattress*)
    void extract_row(Mattress*, int, double*);
    void extract_column(Mattress*, int, double*);
    void free_mat(Mattress*);

cdef extern from "cpp/dense.cpp":
    Mattress* initialize_dense_matrix[T](int, int, T*, bool)
