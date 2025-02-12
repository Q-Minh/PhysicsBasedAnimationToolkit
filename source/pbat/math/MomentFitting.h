/**
 * @file MomentFitting.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Moment fitting for polynomial quadrature rules
 * @date 2025-02-11
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_MATH_MOMENTFITTING_H
#define PBAT_MATH_MOMENTFITTING_H

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
#include <tuple>
#include <unsupported/Eigen/NNLS>
#include <utility>

namespace pbat {
namespace math {

/**
 * @brief Quadrature rule with variable points and weights.
 *
 * @tparam Dims Spatial dimensions
 * @tparam Order Polynomial order
 */
template <int Dims, int Order>
struct DynamicQuadrature
{
    static auto constexpr kOrder = Order; ///< Polynomial order
    static auto constexpr kDims  = Dims;  ///< Spatial dimensions

    MatrixX points;  ///< \f$ d \times n \f$ array of quadrature points \f$ Xg \f$, where \f$ n \f$
                     ///< is the number of quadrature points
    VectorX weights; ///< \f$ n \times 1 \f$ array of quadrature weights \f$ wg \f$ associated with
                     ///< \f$ Xg \f$
};

/**
 * @brief Compute the moment fitting matrix in the reference simplex space.
 *
 * @tparam TBasis Polynomial basis type
 * @tparam TQuad Quadrature rule type
 * @param Pb Polynomial basis
 * @param Q Quadrature rule
 * @return Moment fitting matrix
 */
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

/**
 * @brief Compute the moment fitting matrix in the reference simplex space.
 *
 * @tparam TBasis Polynomial basis type
 * @tparam TQuad Quadrature rule type
 * @param Pb Polynomial basis
 * @param Q Quadrature rule
 * @return Moment fitting matrix
 */
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

/**
 * @brief Compute the moment fitting matrix in the reference simplex space.
 *
 * @tparam TBasis Polynomial basis type
 * @tparam TDerivedXg Eigen expression for quadrature points
 * @param Pb Polynomial basis
 * @param Xg Quadrature points
 * @return Moment fitting matrix
 */
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
 * @brief Computes non-negative quadrature weights \f$ w \f$ by moment fitting.
 * @tparam TDerivedP Eigen matrix expression for moment fitting matrix
 * @tparam TDerivedB Eigen vector expression for target integrated polynomials
 * @param P Moment fitting matrix \f$ \mathbf{P} \in \mathbb{R}^{s \times n} \f$, where \f$ s \f$ is
 * the polynomial basis' size and \f$ n \f$ is the number of quadrature points
 * @param b Target integrated polynomials \f$ \mathbf{b} \in \mathbb{R}^s \f$
 * @param maxIterations Maximum number of non-negative least-squares active set solver
 * @param precision Convergence threshold
 * @return Non-negative quadrature weights \f$ \mathbf{w} \in \mathbb{R}^n \f$
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
    nnls.setMaxIterations(maxIterations);
    nnls.setTolerance(precision);
    nnls.compute(P.derived());
    auto w = nnls.solve(b);
    if (nnls.info() != Eigen::ComputationInfo::Success)
    {
        std::string what = "Moment fitting's non-negative least-squares failed with error: ";
        if (nnls.info() == Eigen::ComputationInfo::InvalidInput)
            what += "InvalidInput";
        if (nnls.info() == Eigen::ComputationInfo::NoConvergence)
            what += "NoConvergence";
        if (nnls.info() == Eigen::ComputationInfo::NumericalIssue)
            what += "NumericalIssue";
        throw std::invalid_argument(what);
    }
    return w;
}

/**
 * @brief Computes \f$ \int_{\Omega} P(x) \, d\Omega \f$ given an existing quadrature rule \f$
 * (Xg,wg) \f$.
 * @tparam Polynomial Polynomial basis type
 * @tparam TDerivedXg Eigen matrix expression for quadrature points
 * @tparam TDerivedWg Eigen vector expression for quadrature weights
 * @param P Polynomial basis
 * @param Xg \f$ d \times n \f$ array of quadrature point positions defined in reference space
 * @param wg \f$ n \times 1 \f$ array of quadrature weights
 * @return \f$ s \times 1 \f$ array of integrated polynomials
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
 * @brief Obtain weights \f$ w_g^1 \f$ by transferring an existing quadrature rule \f$ X_g^2, w_g^2
 * \f$ defined on a simplex onto a new quadrature rule \f$ X_g^1, w_g^1 \f$ defined on the same
 * simplex.
 *
 * @tparam TDerivedXg1 Eigen matrix expression for quadrature points
 * @tparam TDerivedXg2 Eigen matrix expression for existing quadrature points
 * @tparam TDerivedWg2 Eigen vector expression for existing quadrature weights
 * @tparam Polynomial Polynomial basis type
 * @param P Polynomial basis
 * @param Xg1 \f$ d \times n_1 \f$ array \f$ X_g^1 \f$ of quadrature point positions defined in
 * reference space
 * @param Xg2 \f$ d \times n_2 \f$ array \f$ X_g^2 \f$ of quadrature point positions defined in
 * reference space
 * @param wg2 \f$ n_2 \times 1 \f$ array \f$ w_g^2 \f$ of existing quadrature weights
 * @param maxIterations Maximum number of non-negative least-squares active set solver
 * @param precision Convergence threshold
 * @return \f$ n_1 \times 1 \f$ array of quadrature weights
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
 * @brief Obtain weights \f$ w_g^1 \f$ by transferring an existing quadrature rule \f$ X_g^2, w_g^2
 * \f$ defined on a domain of simplices onto a new quadrature rule \f$ X_g^1, w_g^1 \f$ defined on
 * the same domain.
 *
 * @tparam Order Order of the quadrature rules
 * @tparam TDerivedS1 Eigen vector expression for simplex indices
 * @tparam TDerivedXi1 Eigen matrix expression for quadrature points
 * @tparam TDerivedS2 Eigen vector expression for simplex indices
 * @tparam TDerivedXi2 Eigen matrix expression for existing quadrature points
 * @tparam TDerivedWg2 Eigen vector expression for existing quadrature weights
 * @param S1 \f$ |S_1| \times 1 \f$ index array giving the simplex containing the corresponding
 * quadrature point in columns of Xi1, i.e. `S1(g)` is the simplex containing `Xi1.col(g)`
 * @param Xi1 \f$ d \times n_1 \f$ array \f$ \xi_g^1 \f$ of quadrature point positions defined in
 * simplex space
 * @param S2 \f$ |S_2| \times 1 \f$ index array giving the simplex containing the corresponding
 * quadrature point in columns of Xi2, i.e. `S2(g)` is the simplex containing `Xi2.col(g)`
 * @param Xi2 \f$ d \times n_2 \f$ array of quadrature point positions defined in reference
 * simplex space
 * @param wi2 \f$ n_2 \times 1 \f$ array \f$ w_g^2 \f$ of existing quadrature weights
 * @param nSimplices Number of simplices in the domain. If `nSimplices < 1`, the number of simplices
 * is inferred from S1 and S2.
 * @param bEvaluateError Whether to compute the integration error on the new quadrature rule
 * @param maxIterations Maximum number of non-negative least-squares active set solver
 * @param precision Convergence threshold
 * @return `(w, e)` where `w` are the quadrature weights associated with
 * points `Xi1`, and `e` is the integration error in each simplex (zeros if `not bEvaluateError`).
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
    Eigen::DenseBase<TDerivedWg2> const& wi2,
    Index nSimplices    = -1,
    bool bEvaluateError = false,
    Index maxIterations = 10,
    Scalar precision    = std::numeric_limits<Scalar>::epsilon())
{
    // Compute adjacency graph from simplices s to their quadrature points Xi
    using common::ArgSort;
    using common::Counts;
    using common::CumSum;
    using common::ToEigen;
    if (nSimplices < 0)
        nSimplices = std::max(
                         *std::max_element(S1.begin(), S1.end()),
                         *std::max_element(S2.begin(), S2.end())) +
                     1;
    IndexVectorX S1P = CumSum(Counts(S1.begin(), S1.end(), nSimplices));
    IndexVectorX S2P = CumSum(Counts(S2.begin(), S2.end(), nSimplices));
    IndexVectorX S1N = ArgSort<Index>(S1.size(), [&](auto si, auto sj) { return S1(si) < S1(sj); });
    IndexVectorX S2N = ArgSort<Index>(S2.size(), [&](auto si, auto sj) { return S2(si) < S2(sj); });
    // Find weights wg1 that fit the given quadrature rule Xi2, wi2 on simplices S2
    auto fSolveWeights = [maxIterations,
                          precision,
                          bEvaluateError](auto const& Xg1, auto const& Xg2, auto const& wg2) {
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
    // Solve moment fitting on each simplex
    VectorX error = VectorX::Zero(nSimplices);
    VectorX wi1   = VectorX::Zero(Xi1.cols());
    tbb::parallel_for(Index(0), nSimplices, [&](Index s) {
        auto S1begin = S1P(s);
        auto S1end   = S1P(s + 1);
        if (S1end > S1begin)
        {
            auto s1inds     = S1N(Eigen::seq(S1begin, S1end - 1));
            MatrixX Xg1     = Xi1(Eigen::placeholders::all, s1inds);
            auto S2begin    = S2P(s);
            auto S2end      = S2P(s + 1);
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

/**
 * @brief Computes reference moment fitting systems for all simplices \f$ S_1 \f$, given an existing
 * quadrature rule \f$ (X_g^2, w_g^2) \f$ defined on a domain of simplices. The quadrature points of
 * the new rule \f$ X_g^1 \f$ are given and fixed.
 *
 * @tparam Order Order of the quadrature rules
 * @tparam TDerivedS1 Eigen vector expression for simplex indices
 * @tparam TDerivedX1 Eigen matrix expression for quadrature points
 * @tparam TDerivedS2 Eigen vector expression for simplex indices
 * @tparam TDerivedX2 Eigen matrix expression for existing quadrature points
 * @tparam TDerivedW2 Eigen vector expression for existing quadrature weights
 * @param S1 \f$ |S_1| \times 1 \f$ index array giving the simplex containing the corresponding
 * quadrature point in columns of X1, i.e. `S1(g)` is the simplex containing `X1.col(g)`
 * @param X1 \f$ d \times n_1 \f$ array \f$ X_g^1 \f$ of quadrature point positions defined in
 * reference space
 * @param S2 \f$ |S_2| \times 1 \f$ index array giving the simplex containing the corresponding
 * quadrature point in columns of X2, i.e. `S2(g)` is the simplex containing `X2.col(g)`
 * @param X2 \f$ d \times n_2 \f$ array of quadrature point positions defined in reference simplex
 * space
 * @param w2 \f$ n_2 \times 1 \f$ array \f$ w_g^2 \f$ of existing quadrature weights
 * @param nSimplices Number of simplices in the domain. If `nSimplices < 1`, the number of simplices
 * is inferred from S1 and S2.
 * @return `(P, B, prefix)` where `P` is the moment
 * fitting matrix, `B` is the target integrated polynomials, and `prefix` is the prefix into columns
 * of `P` for each simplex, i.e. the block `P[:,prefix[s]:prefix[s+1]] = B[:,s]` represents the
 * moment fitting system for simplex `s`.
 */
template <
    int Order,
    class TDerivedS1,
    class TDerivedX1,
    class TDerivedS2,
    class TDerivedX2,
    class TDerivedW2>
std::tuple<MatrixX /*P*/, MatrixX /*B*/, IndexVectorX /*prefix into columns of P*/>
ReferenceMomentFittingSystems(
    Eigen::DenseBase<TDerivedS1> const& S1,
    Eigen::MatrixBase<TDerivedX1> const& X1,
    Eigen::DenseBase<TDerivedS2> const& S2,
    Eigen::MatrixBase<TDerivedX2> const& X2,
    Eigen::DenseBase<TDerivedW2> const& w2,
    Index nSimplices = Index(-1))
{
    // Compute adjacency graph from simplices s to their quadrature points Xi
    using common::ArgSort;
    using common::Counts;
    using common::CumSum;
    using common::ToEigen;
    if (nSimplices < 0)
        nSimplices = std::max(
                         *std::max_element(S1.begin(), S1.end()),
                         *std::max_element(S2.begin(), S2.end())) +
                     1;
    IndexVectorX S1P = CumSum(Counts<Index>(S1.begin(), S1.end(), nSimplices));
    IndexVectorX S2P = CumSum(Counts<Index>(S2.begin(), S2.end(), nSimplices));
    IndexVectorX S1N = ArgSort<Index>(S1.size(), [&](auto si, auto sj) { return S1(si) < S1(sj); });
    IndexVectorX S2N = ArgSort<Index>(S2.size(), [&](auto si, auto sj) { return S2(si) < S2(sj); });
    // Assemble moment fitting matrices and their rhs
    auto fPolyRows = [](MatrixX const& Xg) {
        if (Xg.rows() == 1)
            return OrthonormalPolynomialBasis<1, Order>::kSize;
        if (Xg.rows() == 2)
            return OrthonormalPolynomialBasis<2, Order>::kSize;
        if (Xg.rows() == 3)
            return OrthonormalPolynomialBasis<3, Order>::kSize;
        throw std::invalid_argument(
            "Expected quadrature points in reference simplex space of dimensions (i.e. rows) 1,2 "
            "or 3.");
    };
    auto fAssembleSystem =
        [](auto const& Xg1, auto const& Xg2, auto const& wg2) -> std::pair<MatrixX, VectorX> {
        if (Xg1.rows() == 1)
        {
            OrthonormalPolynomialBasis<1, Order> P{};
            auto M = ReferenceMomentFittingMatrix(P, Xg1);
            auto b = Integrate(P, Xg2, wg2);
            return {M, b};
        }
        if (Xg1.rows() == 2)
        {
            OrthonormalPolynomialBasis<2, Order> P{};
            auto M = ReferenceMomentFittingMatrix(P, Xg1);
            auto b = Integrate(P, Xg2, wg2);
            return {M, b};
        }
        if (Xg1.rows() == 3)
        {
            OrthonormalPolynomialBasis<3, Order> P{};
            auto M = ReferenceMomentFittingMatrix(P, Xg1);
            auto b = Integrate(P, Xg2, wg2);
            return {M, b};
        }
        throw std::invalid_argument(
            "Expected quadrature points in reference simplex space of dimensions (i.e. rows) 1,2 "
            "or 3.");
    };
    auto nrows = fPolyRows(X1);

    // Count actual number of systems and their dimensions
    Index nSystems{0};
    for (Index s = 0; s < nSimplices; ++s)
        if (S1P(s + 1) > S1P(s))
            ++nSystems;
    IndexVectorX nQuads(nSystems);
    nQuads.setZero();
    for (Index s = 0, sy = 0; s < nSimplices; ++s)
    {
        if (S1P(s + 1) > S1P(s))
        {
            nQuads(sy++) = S1P(s + 1) - S1P(s);
        }
    }
    IndexVectorX const prefix = CumSum(nQuads);

    MatrixX P(nrows, prefix(Eigen::placeholders::last));
    MatrixX B(nrows, nSystems);
    B.setZero();
    for (Index s = 0, sy = 0; s < nSimplices; ++s)
    {
        auto S1begin = S1P(s);
        auto S1end   = S1P(s + 1);
        if (S1end > S1begin)
        {
            auto s1inds                               = S1N(Eigen::seq(S1begin, S1end - 1));
            MatrixX Xg1                               = X1(Eigen::placeholders::all, s1inds);
            auto S2begin                              = S2P(s);
            auto S2end                                = S2P(s + 1);
            auto s2inds                               = S2N(Eigen::seq(S2begin, S2end - 1));
            MatrixX Xg2                               = X2(Eigen::placeholders::all, s2inds);
            VectorX wg2                               = w2(s2inds);
            auto [Ps, bs]                             = fAssembleSystem(Xg1, Xg2, wg2);
            P.block(0, prefix(sy), nrows, nQuads(sy)) = Ps;
            B.col(sy)                                 = bs;
            ++sy;
        }
    }
    return std::make_tuple(P, B, prefix);
}

/**
 * @brief Block diagonalizes the given reference moment fitting systems (see
 * ReferenceMomentFittingSystems()) into a large sparse matrix.
 *
 * @tparam TDerivedM Eigen matrix expression for moment fitting matrices
 * @tparam TDerivedP Eigen vector expression for prefix into columns of moment fitting matrices
 * @param M Moment fitting matrices
 * @param P Prefix into columns of moment fitting matrices
 * @return The block diagonal row sparse matrix \f$ \mathbf{A} \f$, whose diagonal blocks
 * are the individual reference moment fitting matrices in M, such that \f$ \mathbf{A} \mathbf{w} =
 * \text{vec}(\mathbf{B}) \f$ is the global sparse linear system to solve for quadrature weights \f$
 * \mathbf{w} \f$.
 */
template <class TDerivedM, class TDerivedP>
CSRMatrix BlockDiagonalReferenceMomentFittingSystem(
    Eigen::MatrixBase<TDerivedM> const& M,
    Eigen::DenseBase<TDerivedP> const& P)
{
    auto const nblocks    = P.size() - 1;
    auto const nblockrows = M.rows();
    auto const nrows      = nblockrows * nblocks;
    auto const ncols      = P(Eigen::placeholders::last);
    CSRMatrix GM(nrows, ncols);
    IndexVectorX reserves(nrows);
    for (auto b = 0; b < nblocks; ++b)
    {
        auto begin            = P(b);
        auto end              = P(b + 1);
        auto const nblockcols = end - begin;
        auto const offset     = b * nblockrows;
        for (auto i = 0; i < nblockrows; ++i)
            reserves(offset + i) = nblockcols;
    }
    GM.reserve(reserves);
    for (auto b = 0; b < nblocks; ++b)
    {
        auto begin            = P(b);
        auto end              = P(b + 1);
        auto const nblockcols = end - begin;
        auto Mb               = M.block(0, begin, nblockrows, nblockcols);
        auto const roffset    = b * nblockrows;
        auto const coffset    = begin;
        for (auto i = 0; i < nblockrows; ++i)
        {
            for (auto j = 0; j < nblockcols; ++j)
            {
                GM.insert(roffset + i, coffset + j) = Mb(i, j);
            }
        }
    }
    return GM;
}

} // namespace math
} // namespace pbat

#endif // PBAT_MATH_MOMENTFITTING_H
