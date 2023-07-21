from libcpp.memory cimport shared_ptr

cdef extern from "tatami/tatami.hpp":
    cdef cppclass TatamiNumericMatrix:
        pass

cdef class TatamiNumericPointer:
    cdef shared_ptr[TatamiNumericMatrix] ptr
    cdef object obj # held to prevent garbage collection.
