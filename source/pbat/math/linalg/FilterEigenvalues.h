#ifndef PBAT_MATH_LINALG_FILTEREIGENVALUES_H
#define PBAT_MATH_LINALG_FILTEREIGENVALUES_H

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

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
 * @param A Input matrix
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
            return Eigen::Success;
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

} // namespace pbat::math::linalg

#endif // PBAT_MATH_LINALG_FILTEREIGENVALUES_H
