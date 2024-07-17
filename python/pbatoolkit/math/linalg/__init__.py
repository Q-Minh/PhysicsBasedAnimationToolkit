from ..._pbat.math import linalg as _linalg
import numpy as np
import scipy as sp
from enum import Enum

if hasattr(_linalg, "Cholmod"):
    Cholmod = _linalg.Cholmod


class Ordering(Enum):
    Natural = 0
    AMD = 1
    COLAMD = 2


class SolverBackend(Enum):
    Eigen = 0
    SuiteSparse = 1
    IntelMKL = 2


def lu(A, solver: SolverBackend = SolverBackend.IntelMKL):
    """Returns an instance to an LU factorization suitable for matrices of 
    the same type as A. The instance returned by this function should be used to factorize A,
    i.e. this function does not actually compute the factorization. We currently only support 
    Pardiso's LU factorization, if pbatoolkit was built with MKL support. Otherwise, use scipy's 
    LU factorization (i.e. SuperLU). We may support umfpack and Eigen's supernodal LU factorization
    in the future.

    Args:
        A: Input sparse matrix to decompose, either scipy.sparse.csc_matrix or scipy.sparse.csr_matrix
        solver (SolverBackend, optional): The LLT implementation to use. Defaults to SolverBackend.Eigen.

    Raises:
        TypeError: A must be either scipy.sparse.csc_matrix or scipy.sparse.csr_matrix
        ValueError: Eigen LU (supernodal LU) not supported.
        ValueError: SuiteSparse LU (umfpack, KLU) not supported.
        ValueError: pbatoolkit was not built with MKL support.

    Returns:
        An uninitialized LU factorization of A
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
        raise ValueError("Eigen LU (supernodal LU) not supported")
    if solver == SolverBackend.SuiteSparse:
        raise ValueError("SuiteSparse LU (umfpack, KLU) not supported")
    if solver == SolverBackend.IntelMKL:
        class_ = getattr(_linalg, f"PardisoLU_{mtype}")
        if class_ is None:
            raise ValueError("pbatoolkit was not built with MKL support")
        return class_()


def chol(A, solver: SolverBackend = SolverBackend.Eigen):
    """Returns an instance to an LLT (Cholesky) factorization suitable for matrices of 
    the same type as A. The instance returned by this function should be used to factorize A,
    i.e. this function does not actually compute the factorization. If the SolverBackend is 
    Eigen, we return an LDLT factorization.

    Args:
        A: Input sparse matrix to decompose, either scipy.sparse.csc_matrix or scipy.sparse.csr_matrix.
        solver (SolverBackend, optional): The LLT implementation to use. Defaults to SolverBackend.Eigen.

    Raises:
        TypeError: A must be either scipy.sparse.csc_matrix or scipy.sparse.csr_matrix
        ValueError: pbatoolkit was not built with SuiteSparse support.
        ValueError: pbatoolkit was not built with MKL support.

    Returns:
        An uninitialized LLT factorization of A
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
        return ldlt(A, solver=SolverBackend.Eigen)
    if solver == SolverBackend.SuiteSparse:
        class_ = getattr(_linalg, f"Cholmod")
        if class_ is None:
            raise ValueError(
                "pbatoolkit was not built with SuiteSparse support")
        return class_()
    if solver == SolverBackend.IntelMKL:
        class_ = getattr(_linalg, f"PardisoLLT_{mtype}")
        if class_ is None:
            raise ValueError("pbatoolkit was not built with MKL support")
        return class_()


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
        ValueError: SuiteSparse's LDLT is the same as Eigen's use SolverBackend.Eigen
        ValueError: pbatoolkit was not built with MKL support.

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
        raise ValueError(
            "Cholmod's LDLT factorization is simplicial just like Eigen's. Use SolverBackend.Eigen instead")
    if solver == SolverBackend.IntelMKL:
        class_ = getattr(_linalg, f"PardisoLDLT_{mtype}")
        if class_ is None:
            raise ValueError("pbatoolkit was not built with MKL support")
        return class_()
