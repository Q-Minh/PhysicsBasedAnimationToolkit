#ifndef PBAT_MATH_MOMENT_FITTING_H
#define PBAT_MATH_MOMENT_FITTING_H

#include "Concepts.h"
#include "PolynomialBasis.h"
#include "pbat/Aliases.h"
#include "pbat/common/ArgSort.h"
#include "pbat/common/Eigen.h"
#include "pbat/common/Indexing.h"

#include <algorithm>
#include <exception>
#include <limits>
#include <tbb/parallel_for.h>
#include <unsupported/Eigen/NNLS>
#include <utility>

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

template <CPolynomialBasis TBasis, class TDerivedXg>
Matrix<TBasis::kSize, Eigen::Dynamic>
ReferenceMomentFittingMatrix(TBasis const& Pb, Eigen::MatrixBase<TDerivedXg> const& Xg)
{
    using QuadratureType = DynamicQuadrature<TBasis::kDims, TBasis::kOrder>;
    static_assert(
        TBasis::kDims == QuadratureType::kDims,
        "Dimensions of the quadrature rule and the polynomial basis must match, i.e. a k-D "
        "polynomial must be fit in a k-D integration domain.");
    Matrix<TBasis::kSize, Eigen::Dynamic> P(TBasis::kSize, Xg.cols());
    for (auto g = 0u; g < Xg.cols(); ++g)
        P.col(g) = Pb.eval(Xg.col(g));
    return P;
}

/**
 * @brief Computes non-negative quadrature weights wg by moment fitting.
 * @tparam TDerivedP
 * @tparam TDerivedB
 * @param P Moment fitting matrix
 * @param b Target integrated polynomials
 * @param maxIterations Maximum number of non-negative least-squares active set solver
 * @param precision Convergence threshold
 * @return
 */
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

/**
 * @brief
 * @tparam Polynomial
 * @tparam TDerivedXg
 * @tparam TDerivedWg
 * @param P
 * @param Xg |#dims|x|#quad.pts.|
 * @param wg |#quad.pts.|x1
 * @return
 */
template <CPolynomialBasis Polynomial, class TDerivedXg, class TDerivedWg>
Vector<Polynomial::kSize> Integrate(
    Polynomial const& P,
    Eigen::MatrixBase<TDerivedXg> const& Xg,
    Eigen::DenseBase<TDerivedWg> const& wg)
{
    Vector<Polynomial::kSize> b{};
    b.setZero();
    for (auto g = 0; g < wg.size(); ++g)
        b += wg(g) * P.eval(Xg.col(g));
    return b;
}

/**
 * @brief
 * @tparam TDerivedXg1
 * @tparam TDerivedXg2
 * @tparam TDerivedWg2
 * @tparam Polynomial
 * @param P
 * @param Xg1 |#dims|x|#new quad.pts.|
 * @param Xg2 |#dims|x|#old quad.pts.|
 * @param wg2 |#old quad.pts.|x1
 * @param maxIterations Maximum number of non-negative least-squares active set solver
 * @param precision Convergence threshold
 * @return
 */
template <CPolynomialBasis Polynomial, class TDerivedXg1, class TDerivedXg2, class TDerivedWg2>
Vector<TDerivedXg1::ColsAtCompileTime> TransferQuadrature(
    Polynomial const& P,
    Eigen::MatrixBase<TDerivedXg1> const& Xg1,
    Eigen::MatrixBase<TDerivedXg2> const& Xg2,
    Eigen::DenseBase<TDerivedWg2> const& wg2,
    Index maxIterations = 10,
    Scalar precision    = std::numeric_limits<Scalar>::epsilon())
{
    auto b = Integrate(P, Xg2, wg2);
    auto M = ReferenceMomentFittingMatrix(P, Xg1);
    auto w = MomentFittedWeights(M, b, maxIterations, precision);
    return w;
}

/**
 * @brief Obtain weights wi1 by transferring an existing quadrature rule (Xi2,wi2) defined on a
 * domain composed of simplices onto a new quadrature rule (Xi1,wi1) defined on the same domain,
 * given fixed quadrature points Xi1.
 *
 * @tparam Order Order of the quadrature rules
 * @tparam TDerivedS1
 * @tparam TDerivedXi1
 * @tparam TDerivedS2
 * @tparam TDerivedXi2
 * @tparam TDerivedWg2
 * @param S1 1x|Xi1.cols()| Index array giving the simplex containing the corresponding quadrature
 * point in columns of Xi1.
 * @param Xi1 |#dims|x|#new quad.pts.| array of quadrature point positions defined in reference
 * simplex space
 * @param S2 |Xi2.cols()|x1 index array giving the simplex containing the corresponding quadrature
 * point in columns of Xi2.
 * @param Xi2 |#dims|x|#old quad.pts.| array of quadrature point positions defined in reference
 * simplex space
 * @param wi2 |#old quad.pts.|x1 array of quadrature weights
 * @param bEvaluateError Whether to compute the integration error on the new quadrature rule
 * @param maxIterations Maximum number of non-negative least-squares active set solver
 * @param precision Convergence threshold
 * @return (w, e) where w are the quadrature weights associated with points Xi1, and e is the
 * integration error in each simplex (zeros if not bEvaluateError).
 */
template <
    auto Order,
    class TDerivedS1,
    class TDerivedXi1,
    class TDerivedS2,
    class TDerivedXi2,
    class TDerivedWg2>
std::pair<VectorX, VectorX> TransferQuadrature(
    Eigen::DenseBase<TDerivedS1> const& S1,
    Eigen::MatrixBase<TDerivedXi1> const& Xi1,
    Eigen::DenseBase<TDerivedS2> const& S2,
    Eigen::MatrixBase<TDerivedXi2> const& Xi2,
    Eigen::MatrixBase<TDerivedWg2> const& wi2,
    bool bEvaluateError = false,
    Index maxIterations = 10,
    Scalar precision    = std::numeric_limits<Scalar>::epsilon())
{
    // Compute adjacency graph from simplices s to their quadrature points Xi
    using common::ArgSort;
    using common::Counts;
    using common::CumSum;
    using common::ToEigen;
    auto nsimplices =
        std::max(*std::max_element(S1.begin(), S1.end()), *std::max_element(S2.begin(), S2.end())) +
        1;
    std::vector<Index> S1P = CumSum(Counts<Index>(S1.begin(), S1.end(), nsimplices));
    std::vector<Index> S2P = CumSum(Counts<Index>(S2.begin(), S2.end(), nsimplices));
    IndexVectorX S1N =
        ToEigen(ArgSort<Index>(S1.size(), [&](auto si, auto sj) { return S1(si) < S1(sj); }));
    IndexVectorX S2N =
        ToEigen(ArgSort<Index>(S2.size(), [&](auto si, auto sj) { return S2(si) < S2(sj); }));
    // Find weights wg1 that fit the given quadrature rule Xi2, wi2 on simplices S2
    auto fSolveWeights = [maxIterations, precision, bEvaluateError](
                             MatrixX const& Xg1,
                             MatrixX const& Xg2,
                             VectorX const& wg2) {
        if (Xg1.rows() == 1)
        {
            OrthonormalPolynomialBasis<1, Order> P{};
            auto w = TransferQuadrature(P, Xg1, Xg2, wg2, maxIterations, precision);
            Scalar error(0);
            if (bEvaluateError)
            {
                auto b1 = math::Integrate(P, Xg1, w);
                auto b2 = math::Integrate(P, Xg2, wg2);
                error   = (b1 - b2).squaredNorm();
            }
            return std::make_pair(w, error);
        }
        if (Xg1.rows() == 2)
        {
            OrthonormalPolynomialBasis<2, Order> P{};
            auto w = TransferQuadrature(P, Xg1, Xg2, wg2, maxIterations, precision);
            Scalar error(0);
            if (bEvaluateError)
            {
                auto b1 = math::Integrate(P, Xg1, w);
                auto b2 = math::Integrate(P, Xg2, wg2);
                error   = (b1 - b2).squaredNorm();
            }
            return std::make_pair(w, error);
        }
        if (Xg1.rows() == 3)
        {
            OrthonormalPolynomialBasis<3, Order> P{};
            auto w = TransferQuadrature(P, Xg1, Xg2, wg2, maxIterations, precision);
            Scalar error(0);
            if (bEvaluateError)
            {
                auto b1 = math::Integrate(P, Xg1, w);
                auto b2 = math::Integrate(P, Xg2, wg2);
                error   = (b1 - b2).squaredNorm();
            }
            return std::make_pair(w, error);
        }
        throw std::invalid_argument(
            "Expected quadrature points in reference simplex space of dimensions (i.e. rows) 1,2 "
            "or 3.");
    };

    VectorX error = VectorX::Zero(nsimplices);
    VectorX wi1   = VectorX::Zero(Xi1.cols());
    tbb::parallel_for(Index(0), nsimplices, [&](Index s) {
        auto S1begin = S1P[s];
        auto S1end   = S1P[s + 1];
        if (S1end > S1begin)
        {
            auto s1inds     = S1N(Eigen::seq(S1begin, S1end - 1));
            MatrixX Xg1     = Xi1(Eigen::placeholders::all, s1inds);
            auto S2begin    = S2P[s];
            auto S2end      = S2P[s + 1];
            auto s2inds     = S2N(Eigen::seq(S2begin, S2end - 1));
            MatrixX Xg2     = Xi2(Eigen::placeholders::all, s2inds);
            VectorX wg2     = wi2(s2inds);
            auto [wg1, err] = fSolveWeights(Xg1, Xg2, wg2);
            wi1(s1inds)     = wg1;
            error(s)        = err;
        }
    });
    return {wi1, error};
}

} // namespace math
} // namespace pbat

#endif // PBAT_MATH_MOMENT_FITTING_H