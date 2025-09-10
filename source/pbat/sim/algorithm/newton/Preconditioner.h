#ifndef PBAT_SIM_ALGORITHM_NEWTON_PRECONDITIONER_H
#define PBAT_SIM_ALGORITHM_NEWTON_PRECONDITIONER_H

#include "HessianProduct.h"
#include "pbat/Aliases.h"

#ifdef PBAT_USE_SUITESPARSE
    #include <Eigen/CholmodSupport>
#else
    #include <Eigen/SparseCholesky>
#endif

namespace pbat::sim::algorithm::newton {

/**
 * @brief Cholesky preconditioner
 *
 * This preconditioner is the full Cholesky (or Bunch-Kaufmann) decomposition of the elasto-dynamics
 * hessian, ignoring the off-diagonals of the contact hessian.
 */
struct Preconditioner
{
#ifdef PBAT_USE_SUITESPARSE
    using DecompositionType = Eigen::CholmodDecomposition<CSCMatrix, Eigen::Upper | Eigen::Lower>;
#else
    using DecompositionType = Eigen::SimplicialLDLT<CSCMatrix, Eigen::Upper | Eigen::Lower>;
#endif // PBAT_USE_SUITESPARSE

    Preconditioner()                                 = default;
    Preconditioner(Preconditioner const&)            = delete;
    Preconditioner(Preconditioner&&)                 = default;
    Preconditioner& operator=(Preconditioner const&) = delete;
    Preconditioner& operator=(Preconditioner&&)      = default;

    DecompositionType mLLT;         ///< Cholesky decomposition
    bool mIsPatternAnalyzed{false}; ///< Flag to indicate if the pattern is analyzed
};

/**
 * @brief Preconditioner operator
 * Non-owning view over stateful preconditioner, to use with Eigen's matrix-free solvers.
 */
struct PreconditionerOperator
{
    Preconditioner* mImpl; ///< Pointer to the stateful preconditioner
    /**
     * @brief Construct an empty Preconditioner Operator object
     */
    PreconditionerOperator() = default;
    /**
     * @brief Construct a new Preconditioner Operator object
     * @warning no-op here to satisfy Eigen's preconditioner API
     * @param A The matrix to precondition
     */
    explicit PreconditionerOperator(HessianOperator const& A);
    /**
     * @brief Symbolic factorization
     * @param A The matrix to precondition
     * @return Reference to this
     */
    PreconditionerOperator& analyzePattern(HessianOperator const& A);
    /**
     * @brief Numeric factorization
     * @param A The matrix to precondition
     * @return Reference to this
     */
    PreconditionerOperator& factorize(HessianOperator const& A);
    /**
     * @brief Symbolic and numeric factorization
     * @param A The matrix to precondition
     * @return Reference to this
     */
    PreconditionerOperator& compute(HessianOperator const& A);
    /**
     * @brief Solve the system AX = B
     *
     * @tparam Rhs Eigen type of the right-hand side
     * @param b Right-hand side vector/matrix
     * @return Solve expression
     */
    template <typename Rhs>
    inline auto solve(const Rhs& b) const
    {
        return mImpl->mLLT.solve(b);
    }
    /**
     * @brief Preconditioner's status
     * @return Eigen::ComputationInfo::Success if the factorization was successful, otherwise
     * one of Eigen::ComputationInfo::NumericalIssue, Eigen::ComputationInfo::NoConvergence or
     * Eigen::ComputationInfo::InvalidInput.
     */
    Eigen::ComputationInfo info() const;
};

} // namespace pbat::sim::algorithm::newton

#endif // PBAT_SIM_ALGORITHM_NEWTON_PRECONDITIONER_H
