from mattress import tatamize
from scipy.sparse import rand

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


def test_sparse_csr():
    m = rand(3, 4, density=0.25, format="csr", random_state=42)
    ptr = tatamize(m)
    assert all(ptr.row(0) == m[0, :])
    assert all(ptr.column(1) == m[:, 1])


def test_sparse_csc():
    m = rand(3, 4, density=0.25, format="csc", random_state=42)
    ptr = tatamize(m)
    assert all(ptr.row(0) == m[0, :])
    assert all(ptr.column(1) == m[:, 1])
