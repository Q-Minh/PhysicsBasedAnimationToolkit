#ifndef PBAT_MATH_LINALG_EIGENSOLVER_H
#define PBAT_MATH_LINALG_EIGENSOLVER_H

#include <Eigen/SparseCore>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/MatOp/SparseSymShiftSolve.h>
#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/SymEigsSolver.h>
#include <exception>

namespace pbat::math::linalg {

/**
 * @brief Compute the nearest eigenvalues and corresponding eigenvectors of a sparse symmetric
 * matrix
 *
 * @tparam TScalar Type of the matrix elements
 * @tparam StorageOrder Eigen::ColMajor or Eigen::RowMajor
 * @tparam TIndex Type of the matrix indices
 * @tparam UpLo Eigen::Lower, Eigen::Upper, or Eigen::Lower | Eigen::Upper
 * @param A The input sparse symmetric matrix
 * @param shift The shift value to find eigenvalues near to
 * @param nEigenvalues Number of eigenvalues to compute
 * @param nMaxIters Maximum number of iterations
 * @param tol Tolerance for convergence
 * @param eigvals Output vector of computed eigenvalues
 * @param eigvecs Output matrix of computed eigenvectors
 */
template <class TScalar, int StorageOrder, class TIndex, int UpLo = Eigen::Lower | Eigen::Upper>
void NearestEigenvaluesTo(
    Eigen::SparseMatrix<TScalar, StorageOrder, TIndex> const& A,
    TScalar shift,
    Eigen::Index nEigenvalues,
    Eigen::Index nMaxIters,
    TScalar tol,
    Eigen::Vector<TScalar, Eigen::Dynamic>& eigvals,
    Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic>& eigvecs)
{
    Spectra::SparseSymShiftSolve<TScalar, UpLo, StorageOrder, TIndex> op(A);
    Spectra::SymEigsShiftSolver<decltype(op)> eigs{op, nEigenvalues, 2 * nEigenvalues, shift};
    eigs.init();
    eigs.compute(Spectra::SortRule::LargestMagn, nMaxIters, tol);
    if (eigs.info() != Spectra::CompInfo::Successful)
    {
        throw std::runtime_error("Eigenvalue computation did not converge");
    }
    eigvals = eigs.eigenvalues();
    eigvecs = eigs.eigenvectors();
}

/**
 * @brief Compute the largest eigenvalues and corresponding eigenvectors of a sparse symmetric
 * matrix
 *
 * @tparam TScalar Type of the matrix elements
 * @tparam StorageOrder Eigen::ColMajor or Eigen::RowMajor
 * @tparam TIndex Type of the matrix indices
 * @tparam UpLo Eigen::Lower, Eigen::Upper, or Eigen::Lower | Eigen::Upper
 * @param A The input sparse symmetric matrix
 * @param nEigenvalues Number of eigenvalues to compute
 * @param nMaxIters Maximum number of iterations
 * @param tol Tolerance for convergence
 * @param eigvals Output vector of computed eigenvalues
 * @param eigvecs Output matrix of computed eigenvectors
 */
template <class TScalar, int StorageOrder, class TIndex, int UpLo = Eigen::Lower | Eigen::Upper>
void LargestEigenvaluesOf(
    Eigen::SparseMatrix<TScalar, StorageOrder, TIndex> const& A,
    Eigen::Index nEigenvalues,
    Eigen::Index nMaxIters,
    TScalar tol,
    Eigen::Vector<TScalar, Eigen::Dynamic>& eigvals,
    Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic>& eigvecs)
{
    Spectra::SparseSymMatProd<TScalar, UpLo, StorageOrder, TIndex> op(A);
    Spectra::SymEigsSolver<decltype(op)> eigs{op, nEigenvalues, 2 * nEigenvalues};
    eigs.init();
    eigs.compute(Spectra::SortRule::LargestMagn, nMaxIters, tol);
    if (eigs.info() != Spectra::CompInfo::Successful)
    {
        throw std::runtime_error("Eigenvalue computation did not converge");
    }
    eigvals = eigs.eigenvalues();
    eigvecs = eigs.eigenvectors();
}

} // namespace pbat::math::linalg

#endif // PBAT_MATH_LINALG_EIGENSOLVER_H