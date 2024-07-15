#ifndef PBA_MATH_LINALG_CHOLMOD_H
#define PBA_MATH_LINALG_CHOLMOD_H

#include "pbat/Aliases.h"

#include <algorithm>
#include <exception>
#include <suitesparse/cholmod.h>

namespace pbat {
namespace math {
namespace linalg {

class Cholmod
{
  public:
    static_assert(
        std::is_same_v<Scalar, double>,
        "Cholmod implementation uses double floating point coefficients");
    static_assert(
        std::is_same_v<CSCMatrix::StorageIndex, std::int32_t>,
        "Cholmod implementation uses 32-bit integer indices");

    Cholmod();

    template <class Derived>
    Cholmod(Eigen::SparseCompressedBase<Derived> const& A);

    template <class Derived>
    void analyze(Eigen::SparseCompressedBase<Derived> const& A);

    template <class Derived>
    bool compute(Eigen::SparseCompressedBase<Derived> const& A);

    void analyze();
    bool factorize();

    MatrixX solve(Eigen::Ref<MatrixX const> const& B) const;

    ~Cholmod();

  private:
    template <class Derived>
    void allocate(Eigen::SparseCompressedBase<Derived> const& A);

    void deallocate();

    cholmod_common mCholmodCommon;
    cholmod_sparse* mCholmodA;
    cholmod_factor* mCholmodL;
    bool mbFactorized;
};

template <class Derived>
inline Cholmod::Cholmod(Eigen::SparseCompressedBase<Derived> const& A) : Cholmod()
{
    allocate(A);
}

template <class Derived>
inline void Cholmod::analyze(Eigen::SparseCompressedBase<Derived> const& A)
{
    deallocate();
    allocate(A);
    analyze();
}

template <class Derived>
inline bool Cholmod::compute(Eigen::SparseCompressedBase<Derived> const& A)
{
    analyze(A);
    return factorize();
}

template <class Derived>
inline void Cholmod::allocate(Eigen::SparseCompressedBase<Derived> const& A)
{
    if (A.rows() != A.cols())
    {
        throw std::invalid_argument("Input sparse matrix must be square");
    }
    if (!A.isCompressed())
    {
        throw std::invalid_argument("Input sparse matrix must be compressed");
    }

    // Store only the lower triangular part of A
    mCholmodA = cholmod_allocate_sparse(
        static_cast<size_t>(A.innerSize()),
        static_cast<size_t>(A.outerSize()),
        static_cast<size_t>(A.nonZeros()),
        true /*columns are sorted*/,
        true /*A is packed (i.e. compressed)*/,
        -1 /*tril, i.e. lower triangular part of A is stored*/,
        CHOLMOD_DOUBLE + CHOLMOD_REAL,
        &mCholmodCommon);
    if (mCholmodA == NULL)
    {
        throw std::runtime_error("Failed to allocate sparse matrix");
    }
    // cholmod_allocate_sparse probably already set this as default, but I still set it to be
    // sure.
    mCholmodA->itype = CHOLMOD_INT;

    Scalar const* values = A.valuePtr();
    using IndexType      = Eigen::SparseCompressedBase<Derived>::StorageIndex;
    static_assert(std::is_same_v<IndexType, std::int32_t>, "CHOLMOD compiled with int32 indices");
    IndexType const* innerInds = A.innerIndexPtr();
    IndexType const* outerInds = A.outerIndexPtr();

    std::memcpy(mCholmodA->p, outerInds, sizeof(IndexType) * (A.outerSize() + 1));
    std::memcpy(mCholmodA->i, innerInds, sizeof(IndexType) * A.nonZeros());
    std::memcpy(mCholmodA->x, values, sizeof(Scalar) * A.nonZeros());
}

} // namespace linalg
} // namespace math
} // namespace pbat

#endif // PBA_MATH_LINALG_CHOLMOD_H