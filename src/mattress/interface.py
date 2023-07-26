from functools import singledispatch
from typing import Any

import numpy as np

from .utils import map_order_to_bool

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"

import ctypes as ct

# TODO: surely there's a better way than whatever this is.
import os


def load_dll():
    dirname = os.path.dirname(os.path.abspath(__file__))
    contents = os.listdir(dirname)
    for x in contents:
        if x.startswith("core") and not x.endswith("py"):
            return ct.CDLL(os.path.join(dirname, x))


lib = load_dll()


def includes():
    dirname = os.path.dirname(os.path.abspath(__file__))
    return [
        os.path.join(dirname, "extern", "tatami", "include"),
        os.path.join(dirname, "include"),
    ]


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


lib.py_free_mat.argtypes = [ct.c_void_p]
lib.py_extract_nrow.restype = ct.c_int
lib.py_extract_nrow.argtypes = [ct.c_void_p]
lib.py_extract_ncol.restype = ct.c_int
lib.py_extract_ncol.argtypes = [ct.c_void_p]
lib.py_extract_sparse.restype = ct.c_int
lib.py_extract_sparse.argtypes = [ct.c_void_p]
lib.py_extract_row.argtypes = [ct.c_void_p, ct.c_int, ct.c_void_p]
lib.py_extract_column.argtypes = [ct.c_void_p, ct.c_int, ct.c_void_p]


class TatamiNumericPointer:
    def __init__(self, ptr):
        self.ptr = ptr

    def __del__(self):
        lib.py_free_mat(self.ptr)

    def nrow(self):
        return lib.py_extract_nrow(self.ptr)

    def ncol(self):
        return lib.py_extract_ncol(self.ptr)

    def sparse(self):
        return lib.py_extract_sparse(self.ptr) > 0

    def row(self, r):
        output = np.ndarray((self.ncol(),), dtype="float64")
        lib.py_extract_row(self.ptr, r, output.ctypes.data)
        return output

    def column(self, c):
        output = np.ndarray((self.nrow(),), dtype="float64")
        lib.py_extract_column(self.ptr, c, output.ctypes.data)
        return output


lib.py_initialize_dense_matrix.restype = ct.c_void_p
lib.py_initialize_dense_matrix.argtypes = [
    ct.c_int,
    ct.c_int,
    ct.c_char_p,
    ct.c_void_p,
    ct.c_char,
]


@tatamize.register
def _tatamize_numpy(x: np.ndarray, order: str = "C"):
    order_to_bool = map_order_to_bool(order=order)
    dtype = str(x.dtype).encode("utf-8")
    return TatamiNumericPointer(
        lib.py_initialize_dense_matrix(
            x.shape[0], x.shape[1], dtype, x.ctypes.data, order_to_bool
        )
    )
