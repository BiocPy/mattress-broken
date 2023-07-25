from functools import singledispatch
from typing import Any

import numpy as np

from .utils import map_order_to_bool

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"

import ctypes
import os
lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "core.cpython-311-darwin.so"))
lib.py_initialize_dense_matrix_double.restype = ctypes.c_void_p

@singledispatch
def tatamize(x: Any, order: str = "C"):
    """Converts python matrix representations to tatami.

    Args:
        x (Any): Any matrix-like object.
        order (str): dense matrix representation, ‘C’, ‘F’,
            row-major (C-style) or column-major (Fortran-style) order.

    Raises:
        NotImplementedError: if x is not supported

    Returns:
        TatamiNumericPointer: a pointer to tatami object.
    """
    raise NotImplementedError(
        f"tatamize is not supported for objects of class: {type(x)}"
    )

class TatamiNumericPointer:
    def __init__(self, ptr):
        self.ptr = ptr

    def __del__(self):
        lib.py_free_mat(ctypes.c_void_p(self.ptr))

    def nrow(self):
        return lib.py_extract_nrow(ctypes.c_void_p(self.ptr))

    def ncol(self):
        return lib.py_extract_ncol(ctypes.c_void_p(self.ptr))

    def sparse(self):
        return lib.py_extract_sparse(ctypes.c_void_p(self.ptr))

    def row(self, r):
        output = np.ndarray((self.ncol(),), dtype="float64")
        lib.py_extract_row(ctypes.c_void_p(self.ptr), r, ctypes.c_void_p(output.ctypes.data))
        return output

    def column(self, c):
        output = np.ndarray((self.nrow(),), dtype="float64")
        lib.py_extract_column(ctypes.c_void_p(self.ptr), c, ctypes.c_void_p(output.ctypes.data))
        return output

@tatamize.register
def _tatamize_numpy(x: np.ndarray, order: str = "C"): 
    order_to_bool = map_order_to_bool(order=order)
    print(x.ctypes.data)
    return TatamiNumericPointer(lib.py_initialize_dense_matrix_double(x.shape[0], x.shape[1], ctypes.c_void_p(x.ctypes.data), ctypes.c_char(order_to_bool)))
