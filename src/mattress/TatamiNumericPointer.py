import ctypes as ct
from typing import Sequence

import numpy as np
import scipy.sparse as sp

from .cpphelpers import load_dll
from .types import NumberTypes

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


lib = load_dll()

lib.py_free_mat.argtypes = [ct.c_void_p]
lib.py_extract_nrow.restype = ct.c_int
lib.py_extract_nrow.argtypes = [ct.c_void_p]
lib.py_extract_ncol.restype = ct.c_int
lib.py_extract_ncol.argtypes = [ct.c_void_p]
lib.py_extract_sparse.restype = ct.c_int
lib.py_extract_sparse.argtypes = [ct.c_void_p]
lib.py_extract_row.argtypes = [ct.c_void_p, ct.c_int, ct.c_void_p]
lib.py_extract_column.argtypes = [ct.c_void_p, ct.c_int, ct.c_void_p]

lib.py_initialize_dense_matrix.restype = ct.c_void_p
lib.py_initialize_dense_matrix.argtypes = [
    ct.c_int,
    ct.c_int,
    ct.c_char_p,
    ct.c_void_p,
    ct.c_char,
]

lib.py_initialize_compressed_sparse_matrix.restype = ct.c_void_p
lib.py_initialize_compressed_sparse_matrix.argtypes = [
    ct.c_int,
    ct.c_int,
    ct.c_uint64,
    ct.c_char_p,
    ct.c_void_p,
    ct.c_char_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_uint8,
]

class TatamiNumericPointer:
    """Initialize a Tatami Numeric Ponter object."""

    def __init__(self, ptr: "lib.Mattress", obj: "Any"):
        """Initialize the class.

        Args:
            ptr (lib.Mattress): pointer to a tatami instance.
            obj (Any): arbitrary Python object that is referenced by the tatami instance.
                       This is stored here to avoid garbage collection.
        """
        self.ptr = ptr
        self.obj = obj

    def __del__(self):
        lib.py_free_mat(self.ptr)

    def nrow(self) -> int:
        """Get number of rows.

        Returns:
            int: number of rows.
        """
        return lib.py_extract_nrow(self.ptr)

    def ncol(self) -> int:
        """Get number of columns.

        Returns:
            int: number of columns.
        """
        return lib.py_extract_ncol(self.ptr)

    def sparse(self) -> bool:
        """Is the matrix sparse?

        Returns:
            bool: True if matrix is sparse.
        """
        return lib.py_extract_sparse(self.ptr) > 0

    def row(self, r: int) -> Sequence[NumberTypes]:
        """Access a row from the tatami matrix.

        Args:
            r (int): row to access.

        Returns:
            Sequence[NumberTypes]: row from the matrix.
        """
        output = np.ndarray((self.ncol(),), dtype="float64")
        lib.py_extract_row(self.ptr, r, output.ctypes.data)
        return output

    def column(self, c: int) -> Sequence[NumberTypes]:
        """Access a column from the tatami matrix.

        Args:
            c (int): column to access.

        Returns:
            Sequence[NumberTypes]: column from the matrix.
        """
        output = np.ndarray((self.nrow(),), dtype="float64")
        lib.py_extract_column(self.ptr, c, output.ctypes.data)
        return output

    @classmethod
    def from_dense_matrix(cls, x: np.ndarray) -> "TatamiNumericPointer":
    """Initialize class from a dense matrix.

        Args:
            x (np.ndarray): input numpy array with 2 dimensions.

        Returns:
            TatamiNumericPointer: instance of the class.
        """

        if len(x.shape) != 2:
            raise ValueError("'x' should be a 2-dimensional array")

        dtype = str(x.dtype).encode("utf-8")
        return cls(
            ptr = lib.py_initialize_dense_matrix(
                x.shape[0], x.shape[1], dtype, x.ctypes.data, True
            ),
            obj = x
        )

    @classmethod
    def from_csc_array(cls, x: sp.csc_array) -> "TatamiNumericPointer":
        """Initialize class from a compressed sparse column matrix.

        Args:
            x (scipy.sparse.csc_array): input sparse matrix.

        Returns:
            TatamiNumericPointer: instance of the class.
        """

        tmp = x.indptr.astype(np.uint64)

        return cls(
            ptr = lib.py_initialize_compressed_sparse_matrix(
                x.shape[0], 
                x.shape[1], 
                len(x.data),
                str(x.data.dtype).encode("UTF-8"),
                x.data.ctypes.data,
                str(x.indices.dtype).encode("UTF-8"),
                x.indices.ctypes.data,
                tmp.ctypes.data,
                False
            ),
            obj = [tmp, x]
        )

    @classmethod
    def from_csr_array(cls, x: sp.csr_array) -> "TatamiNumericPointer":
        """Initialize class from a compressed sparse row matrix.

        Args:
            x (scipy.sparse.csc_array): input sparse matrix.

        Returns:
            TatamiNumericPointer: instance of the class.
        """

        tmp = x.indptr.astype(np.uint64)
        return cls(
            ptr = lib.py_initialize_compressed_sparse_matrix(
                x.shape[0], 
                x.shape[1], 
                len(x.data),
                str(x.data.dtype).encode("UTF-8"),
                x.data.ctypes.data,
                str(x.indices.dtype).encode("UTF-8"),
                x.indices.ctypes.data,
                tmp.ctypes.data,
                True 
            ),
            obj = [tmp, x]
        )
