/**
 * @file FilterEigenvalues.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Filter eigenvalues of a symmetric matrix
 * @version 0.1
 * @date 2025-10-14
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_MATH_LINALG_FILTEREIGENVALUES_H
#define PBAT_MATH_LINALG_FILTEREIGENVALUES_H

#include "mini/BinaryOperations.h"
#include "mini/Concepts.h"
#include "mini/Eigenvalues.h"
#include "mini/Matrix.h"
#include "pbat/common/ConstexprFor.h"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <cmath>

namespace pbat::math::linalg {

/**
 * @brief Bit-flag enum for SPD projection type
 */
enum class EEigenvalueFilter {
    None,          // No filter
    SpdProjection, // Project to nearest (in the 2-norm) SPD matrix
    FlipNegative,  // Flip negative eigenvalue signs
};

/**
 * @brief Filter eigenvalues of a symmetric matrix A and store the result in B
 *
 * @tparam TDerivedA Type of the input matrix A
 * @tparam TDerivedB Type of the output matrix B
 * @param A Input (symmetric) matrix
 * @param mode Eigenvalue filtering mode
 * @param B Output matrix
 * @return true if filtering occured successfully
 * @return false if failed (e.g. eigen decomposition failed)
 */
template <class TDerivedA, class TDerivedB>
bool FilterEigenvalues(
    Eigen::MatrixBase<TDerivedA> const& A,
    EEigenvalueFilter mode,
    Eigen::MatrixBase<TDerivedB>& B)
{
    switch (mode)
    {
        using ScalarType = typename TDerivedA::Scalar;
        using MatrixType =
            Eigen::Matrix<ScalarType, TDerivedA::RowsAtCompileTime, TDerivedA::ColsAtCompileTime>;
        case EEigenvalueFilter::None: {
            B = A;
            return true;
        }
        case EEigenvalueFilter::SpdProjection: {
            Eigen::SelfAdjointEigenSolver<MatrixType> eig{};
            eig.compute(A, Eigen::ComputeEigenvectors);
            if (eig.info() != Eigen::Success)
            {
                return false;
            }
            auto D = eig.eigenvalues();
            auto V = eig.eigenvectors();
            for (auto i = 0; i < D.size(); ++i)
            {
                if (D(i) >= 0)
                    break;
                D(i) = ScalarType(0);
            }
            B = V * D.asDiagonal() * V.transpose();
            return true;
        }
        case EEigenvalueFilter::FlipNegative: {
            Eigen::SelfAdjointEigenSolver<MatrixType> eig{};
            eig.compute(A, Eigen::ComputeEigenvectors);
            if (eig.info() != Eigen::Success)
            {
                return false;
            }
            auto D = eig.eigenvalues();
            auto V = eig.eigenvectors();
            for (auto i = 0; i < D.size(); ++i)
            {
                if (D(i) >= 0)
                    break;
                D(i) = -D(i);
            }
            B = V * D.asDiagonal() * V.transpose();
            return true;
        }
        default: return false;
    }
}

/**
 * @brief Filter eigenvalues of a symmetric matrix A and store the result in B
 *
 * @note This function does NOT check for convergence of the eigen decomposition.
 *
 * @tparam TMatrixA Type of the input matrix A
 * @tparam TMatrixB Type of the output matrix B
 * @param A Input (square symmetric) matrix
 * @param mode Eigenvalue filtering mode
 * @param nMaxIters Maximum number of iterations. If -1, use default.
 * @param eps Tolerance for convergence.
 * @return Filtered matrix
 */
template <mini::CMatrix TMatrixA>
auto FilterEigenvalues(
    TMatrixA const& A,
    EEigenvalueFilter mode,
    int nMaxIters = -1,
    typename TMatrixA::ScalarType eps =
        std::numeric_limits<typename TMatrixA::ScalarType>::epsilon())
    -> mini::SMatrix<typename TMatrixA::ScalarType, TMatrixA::kRows, TMatrixA::kCols>
{
    static_assert(TMatrixA::kRows == TMatrixA::kCols, "Matrix A must be square.");
    using ScalarType = typename TMatrixA::ScalarType;
    if (mode == EEigenvalueFilter::None)
        return A;
    mini::SMatrix<ScalarType, TMatrixA::kRows, TMatrixA::kCols> B{};
    B.SetZero();
    auto eigs     = mini::SymmetricEigen(A, false /*bSortEigenvalues*/, nMaxIters, eps);
    auto const& D = eigs.lambda;
    auto const& V = eigs.V;
    switch (mode)
    {
        using namespace std;
        case EEigenvalueFilter::SpdProjection: {
            common::ForRange<0, TMatrixA::kRows>(
                [&]<auto i>() { B += max(D(i), ScalarType(0)) * V.Col(i) * V.Col(i).Transpose(); });
            break;
        }
        case EEigenvalueFilter::FlipNegative: {
            common::ForRange<0, TMatrixA::kRows>(
                [&]<auto i>() { B += abs(D(i)) * V.Col(i) * V.Col(i).Transpose(); });
            break;
        }
        default: break;
    }
    return B;
}

} // namespace pbat::math::linalg

#endif // PBAT_MATH_LINALG_FILTEREIGENVALUES_H
