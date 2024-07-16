// clang-format off
#ifdef PBAT_USE_SUITESPARSE

#ifndef PBAT_MATH_LINALG_CHOLMOD_H
#define PBAT_MATH_LINALG_CHOLMOD_H

#include "pbat/Aliases.h"
#include "PhysicsBasedAnimationToolkitExport.h"

#include <exception>
#include <suitesparse/cholmod.h>
#include <type_traits>
// clang-format on

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

    enum class ESparseStorage {
        SymmetricLowerTriangular = -1,
        Unsymmetric              = 0,
        SymmetricUpperTriangular = 1
    };

    PBAT_API Cholmod();

    template <class Derived>
    void Analyze(
        Eigen::SparseCompressedBase<Derived> const& A,
        ESparseStorage storage = ESparseStorage::SymmetricLowerTriangular);

    template <class Derived>
    bool Factorize(
        Eigen::SparseCompressedBase<Derived> const& A,
        ESparseStorage storage = ESparseStorage::SymmetricLowerTriangular);

    template <class Derived>
    bool Compute(
        Eigen::SparseCompressedBase<Derived> const& A,
        ESparseStorage storage = ESparseStorage::SymmetricLowerTriangular);

    template <class Derived>
    bool Update(Eigen::SparseCompressedBase<Derived> const& U);

    template <class Derived>
    bool Downdate(Eigen::SparseCompressedBase<Derived> const& U);

    PBAT_API MatrixX Solve(Eigen::Ref<MatrixX const> const& B) const;

    PBAT_API ~Cholmod();

  private:
    template <class Derived>
    void ToCholmodView(
        Eigen::SparseCompressedBase<Derived> const& A,
        ESparseStorage storage,
        cholmod_sparse& cholmod_A) const;

    void Deallocate();

    cholmod_common mCholmodCommon;
    cholmod_factor* mCholmodL;
};

template <class Derived>
inline void Cholmod::Analyze(Eigen::SparseCompressedBase<Derived> const& A, ESparseStorage storage)
{
    Deallocate();
    cholmod_sparse cholmod_A{};
    ToCholmodView(A, storage, cholmod_A);
    mCholmodL = cholmod_analyze(&cholmod_A, &mCholmodCommon);
    if (mCholmodL == NULL)
        throw std::runtime_error("Symbolic analysis of Cholesky factor failed");
}

template <class Derived>
inline bool
Cholmod::Factorize(Eigen::SparseCompressedBase<Derived> const& A, ESparseStorage storage)
{
    if (mCholmodL == NULL)
        Analyze(A, storage);

    cholmod_sparse cholmod_A;
    ToCholmodView(A, storage, cholmod_A);
    int const ec = cholmod_factorize(&cholmod_A, mCholmodL, &mCholmodCommon);
    return ec == 1;
}

template <class Derived>
inline bool Cholmod::Compute(Eigen::SparseCompressedBase<Derived> const& A, ESparseStorage storage)
{
    return Factorize(A, storage);
}

template <class Derived>
inline bool Cholmod::Update(Eigen::SparseCompressedBase<Derived> const& U)
{
    cholmod_sparse cholmod_U{};
    ToCholmodView(U, ESparseStorage::Unsymmetric, cholmod_U);
    cholmod_sparse* cholmod_PU = cholmod_submatrix(
        &cholmod_U,
        static_cast<std::int32_t*>(mCholmodL->Perm) /*Fill-reducing permutation*/,
        static_cast<long long>(mCholmodL->n) /*Number of rows*/,
        NULL /*No column permutation*/,
        -1 /*all columns*/,
        1 /*TRUE*/,
        1 /*TRUE*/,
        &mCholmodCommon);
    int const ec = cholmod_updown(1 /*TRUE == update*/, cholmod_PU, mCholmodL, &mCholmodCommon);
    cholmod_free_sparse(&cholmod_PU, &mCholmodCommon);
    return ec == 1 /*success*/;
}

template <class Derived>
inline bool Cholmod::Downdate(Eigen::SparseCompressedBase<Derived> const& U)
{
    // NOTE: Implementation copied from Update, except that FALSE is passed to cholmod_update(...)
    cholmod_sparse cholmod_U{};
    ToCholmodView(U, ESparseStorage::Unsymmetric, cholmod_U);
    cholmod_sparse* cholmod_PU = cholmod_submatrix(
        &cholmod_U,
        static_cast<std::int32_t*>(mCholmodL->Perm) /*Fill-reducing permutation*/,
        static_cast<long long>(mCholmodL->n) /*Number of rows*/,
        NULL /*No column permutation*/,
        -1 /*all columns*/,
        1 /*TRUE*/,
        1 /*TRUE*/,
        &mCholmodCommon);
    int const ec = cholmod_updown(0 /*FALSE == downdate*/, cholmod_PU, mCholmodL, &mCholmodCommon);
    cholmod_free_sparse(&cholmod_PU, &mCholmodCommon);
    return ec == 1 /*success*/;
}

template <class Derived>
inline void Cholmod::ToCholmodView(
    Eigen::SparseCompressedBase<Derived> const& A,
    ESparseStorage storage,
    cholmod_sparse& cholmod_A) const
{
    // Store only the lower triangular part of A
    cholmod_A.nrow   = static_cast<size_t>(A.innerSize());
    cholmod_A.ncol   = static_cast<size_t>(A.outerSize());
    cholmod_A.nzmax  = static_cast<size_t>(A.nonZeros());
    cholmod_A.p      = const_cast<std::int32_t*>(A.outerIndexPtr());
    cholmod_A.i      = const_cast<std::int32_t*>(A.innerIndexPtr());
    cholmod_A.nz     = NULL;
    cholmod_A.x      = const_cast<Scalar*>(A.valuePtr());
    cholmod_A.z      = NULL;
    cholmod_A.stype  = static_cast<std::int32_t>(storage);
    cholmod_A.itype  = CHOLMOD_INT;
    cholmod_A.xtype  = CHOLMOD_REAL;
    cholmod_A.sorted = 1 /*TRUE*/;
    cholmod_A.packed = 1 /*TRUE*/;
}

} // namespace linalg
} // namespace math
} // namespace pbat

    // clang-format off
#endif // PBAT_MATH_LINALG_CHOLMOD_H
#endif // PBAT_USE_SUITESPARSE
// clang-format on