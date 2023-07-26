from typing import Sequence

import numpy as np

from .cpphelpers import lib
from .types import NumberTypes

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


class TatamiNumericPointer:
    """Initialize a Tatami Numeric Ponter object."""

    def __init__(self, ptr: "lib.Mattress"):
        """Initialize the class.

        Args:
            ptr (lib.Mattress): pointer to the tatami object.
        """
        self.ptr = ptr

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
    def from_dense_matrix(
        cls, x: np.ndarray, dtype: str, order: bool
    ) -> "TatamiNumericPointer":
        """Initialize class from a dense matrix.

        Args:
            x (np.ndarray): input numpy matrix.
            dtype (str): dtype of the values.
            order (bool): True if order is 'C' else False.

        Returns:
            TatamiNumericPointer: instance of the class.
        """
        return cls(
            lib.py_initialize_dense_matrix(
                x.shape[0], x.shape[1], dtype, x.ctypes.data, order
            )
        )
