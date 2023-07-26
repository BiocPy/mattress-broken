import ctypes as ct

# TODO: surely there's a better way than whatever this is.
import os

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


def includes() -> list[str]:
    """Provides access to C++ headers (including tatami) for downstream packages.

    Returns:
        list[str]: list of paths to the header files.
    """
    dirname = os.path.dirname(os.path.abspath(__file__))
    return [
        os.path.join(dirname, "extern", "tatami", "include"),
        os.path.join(dirname, "include"),
    ]


def load_dll() -> ct.CDLL:
    """load the shared library.

    usually starts with core.<platform>.<so or dll>.

    Returns:
        ct.CDLL: shared object.
    """
    dirname = os.path.dirname(os.path.abspath(__file__))
    contents = os.listdir(dirname)
    for x in contents:
        if x.startswith("core") and not x.endswith("py"):
            return ct.CDLL(os.path.join(dirname, x))
