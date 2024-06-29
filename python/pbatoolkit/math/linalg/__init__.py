from ..._pbat.math import linalg as _linalg
import numpy as np
import scipy as sp
from enum import Enum


class Ordering(Enum):
    Natural = 0
    AMD = 1
    COLAMD = 2


class SolverBackend(Enum):
    Eigen = 0
    SuiteSparse = 1
    IntelMKL = 2


def ldlt(A, ordering: Ordering = Ordering.AMD, solver: SolverBackend = SolverBackend.Eigen):
    """Returns an instance to an LDLT (Bunch-Kaufman) factorization suitable for matrices of 
    the same type as A. The instance returned by this function should be used to factorize A,
    i.e. this function does not actually compute the factorization.

    Args:
        A: Input sparse matrix to decompose, either scipy.sparse.csc_matrix or scipy.sparse.csr_matrix
        ordering (Ordering, optional): The fill-reducing ordering algorithm. Defaults to Ordering.AMD.
        solver (SolverBackend, optional): The LDLT implementation to use. Defaults to SolverBackend.Eigen.

    Raises:
        TypeError: A must be either scipy.sparse.csc_matrix or scipy.sparse.csr_matrix
        ValueError: SuiteSparse backend is currently not supported.
        ValueError: Intel MKL backend is currently not supported.

    Returns:
        An uninitialized LDLT factorization of A
    """
    mtype = None
    if isinstance(A, sp.sparse.csc_matrix):
        mtype = "Csc"
    if isinstance(A, sp.sparse.csr_matrix):
        mtype = "Csr"
    if mtype is None:
        raise TypeError(
            "Argument A should be either of scipy.sparse.csc_matrix or scipy.sparse.csr_matrix")

    if solver == SolverBackend.Eigen:
        class_ = getattr(_linalg, f"SimplicialLdlt_{mtype}_{ordering.name}")
        return class_()
    if solver == SolverBackend.SuiteSparse:
        raise ValueError("SuiteSparse backend not yet supported")
    if solver == SolverBackend.IntelMKL:
        raise ValueError("IntelMKL backend not yet supported")
