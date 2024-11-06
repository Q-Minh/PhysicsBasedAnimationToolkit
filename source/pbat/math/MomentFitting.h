#ifndef PBAT_MATH_MOMENT_FITTING_H
#define PBAT_MATH_MOMENT_FITTING_H

#include "Concepts.h"
#include "pbat/Aliases.h"
#include "pbat/common/Eigen.h"

#include <limits>
#include <tbb/parallel_for.h>
#include <unsupported/Eigen/NNLS>

namespace pbat {
namespace math {

template <int Dims, int Order>
struct DynamicQuadrature
{
    static auto constexpr kOrder = Order;
    static auto constexpr kDims  = Dims;

    MatrixX points;  ///< |kDims| x |#weights| array of quadrature points Xg
    VectorX weights; ///< Array of quadrature weights wg associated with Xg
};

template <CPolynomialBasis TBasis, CFixedPointPolynomialQuadratureRule TQuad>
Matrix<TBasis::kSize, TQuad::kPoints> ReferenceMomentFittingMatrix(TBasis const& Pb, TQuad const& Q)
{
    static_assert(
        TBasis::kDims == TQuad::kDims,
        "Dimensions of the quadrature rule and the polynomial basis must match, i.e. a k-D "
        "polynomial must be fit in a k-D integration domain.");
    Matrix<TBasis::kSize, TQuad::kPoints> P{};
    auto Xg = common::ToEigen(Q.points).reshaped(TQuad::kDims + 1, TQuad::kPoints);
    // Eigen::Map<Matrix<TQuad::kDims + 1, TQuad::kPoints> const> Xg(Q.points.data());
    for (auto g = 0u; g < TQuad::kPoints; ++g)
        P.col(g) = Pb.eval(Xg.col(g).template segment<TQuad::kDims>(1));
    return P;
}

template <CPolynomialBasis TBasis, CPolynomialQuadratureRule TQuad>
Matrix<TBasis::kSize, Eigen::Dynamic> ReferenceMomentFittingMatrix(TBasis const& Pb, TQuad const& Q)
{
    static_assert(
        TBasis::kDims == TQuad::kDims,
        "Dimensions of the quadrature rule and the polynomial basis must match, i.e. a k-D "
        "polynomial must be fit in a k-D integration domain.");
    Matrix<TBasis::kSize, Eigen::Dynamic> P(TBasis::kSize, Q.weights.size());
    auto Xg = common::ToEigen(Q.points).reshaped(TQuad::kDims + 1, Q.weights.size());
    for (auto g = 0u; g < Xg.cols(); ++g)
        P.col(g) = Pb.eval(Xg.col(g).template segment<TQuad::kDims>(1));
    return P;
}

template <CPolynomialBasis TBasis, int Dims, int Order>
Matrix<TBasis::kSize, Eigen::Dynamic>
ReferenceMomentFittingMatrix(TBasis const& Pb, DynamicQuadrature<Dims, Order> const& Q)
{
    using QuadratureType = DynamicQuadrature<Dims, Order>;
    static_assert(
        TBasis::kDims == QuadratureType::kDims,
        "Dimensions of the quadrature rule and the polynomial basis must match, i.e. a k-D "
        "polynomial must be fit in a k-D integration domain.");
    Matrix<TBasis::kSize, Eigen::Dynamic> P(TBasis::kSize, Q.weights.size());
    auto Xg = common::ToEigen(Q.points).reshaped(QuadratureType::kDims, Q.weights.size());
    for (auto g = 0u; g < Xg.cols(); ++g)
        P.col(g) = Pb.eval(Xg.col(g));
    return P;
}

template <class TDerivedP, class TDerivedB>
VectorX MomentFittedWeights(
    Eigen::MatrixBase<TDerivedP> const& P,
    Eigen::DenseBase<TDerivedB> const& b,
    Index maxIterations = 10,
    Scalar precision    = std::numeric_limits<Scalar>::epsilon())
{
    using MatrixType = TDerivedP;
    Eigen::NNLS<MatrixType> nnls{};
    nnls.compute(P.derived());
    nnls.setMaxIterations(maxIterations);
    nnls.setTolerance(precision);
    auto w = nnls.solve(b);
    return w;
}

} // namespace math
} // namespace pbat

#endif // PBAT_MATH_MOMENT_FITTING_H