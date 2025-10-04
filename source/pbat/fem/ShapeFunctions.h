/**
 * @file ShapeFunctions.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief FEM shape functions and gradients
 * @date 2025-02-11
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_FEM_SHAPEFUNCTIONS_H
#define PBAT_FEM_SHAPEFUNCTIONS_H

#include "Concepts.h"
#include "Jacobian.h"
#include "pbat/Aliases.h"
#include "pbat/common/Concepts.h"
#include "pbat/common/Eigen.h"
#include "pbat/profiling/Profiling.h"

#include <Eigen/LU>
#include <Eigen/SVD>
#include <exception>
#include <fmt/core.h>
#include <string>
#include <tbb/parallel_for.h>

namespace pbat::fem {

/**
 * @brief Computes shape functions at element quadrature points for a polynomial quadrature rule of
 * order QuadratureOrder
 * @tparam TElement Element type
 * @tparam QuadratureOrder Quadrature order
 * @tparam TScalar Floating point type, defaults to Scalar
 * @return `|# element nodes| x |# elem. quad.pts.|` matrix of nodal shape function values at
 * quadrature points
 */
template <CElement TElement, int QuadratureOrder, common::CFloatingPoint TScalar = Scalar>
auto ElementShapeFunctions() -> Eigen::Matrix<
    TScalar,
    TElement::kNodes,
    TElement::template QuadratureType<QuadratureOrder, TScalar>::kPoints>
{
    using QuadratureRuleType = typename TElement::template QuadratureType<QuadratureOrder, TScalar>;
    using ElementType        = TElement;
    auto const Xg            = common::ToEigen(QuadratureRuleType::points)
                        .reshaped(QuadratureRuleType::kDims + 1, QuadratureRuleType::kPoints)
                        .template bottomRows<QuadratureRuleType::kDims>();
    Eigen::Matrix<TScalar, ElementType::kNodes, QuadratureRuleType::kPoints> Ng{};
    for (auto g = 0; g < QuadratureRuleType::kPoints; ++g)
    {
        Ng.col(g) = ElementType::N(Xg.col(g));
    }
    return Ng;
}

/**
 * @brief Constructs a shape function matrix \f$ \mathbf{N} \f$ for a given mesh, i.e. at the
 * element quadrature points.
 * @tparam TElement Element type
 * @tparam QuadratureOrder Quadrature order
 * @tparam TScalar Floating point type
 * @tparam TIndex Index type, defaults to Index
 * @param E `|# nodes|x|# elements|` array of element node indices
 * @param nNodes Number of nodes in the mesh
 * @return `|# elements * # quad.pts.| x |# nodes|` shape function matrix
 */
template <CElement TElement, int QuadratureOrder, common::CFloatingPoint TScalar, class TDerivedE>
auto ShapeFunctionMatrix(Eigen::DenseBase<TDerivedE> const& E, Eigen::Index nNodes)
    -> Eigen::SparseMatrix<TScalar, Eigen::RowMajor, typename TDerivedE::Scalar>
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.fem.ShapeFunctionMatrix");
    using ElementType     = TElement;
    using ScalarType      = TScalar;
    using IndexType       = typename TDerivedE::Scalar;
    using IndexVectorType = Eigen::Vector<IndexType, Eigen::Dynamic>;
    auto const Ng         = ElementShapeFunctions<ElementType, QuadratureOrder, ScalarType>();
    auto const nElements  = E.cols();
    auto const nQuadPointsPerElement = Ng.cols();
    auto const m                     = nQuadPointsPerElement * nElements;
    auto const n                     = nNodes;
    auto constexpr kNodesPerElement  = ElementType::kNodes;
    Eigen::SparseMatrix<ScalarType, Eigen::RowMajor, IndexType> N(m, n);
    N.reserve(IndexVectorType::Constant(m, kNodesPerElement));
    for (auto e = 0; e < nElements; ++e)
    {
        auto const nodes = E.col(e);
        for (auto g = 0; g < nQuadPointsPerElement; ++g)
        {
            auto const row = e * nQuadPointsPerElement + g;
            for (auto i = 0; i < Ng.rows(); ++i)
            {
                auto const col     = nodes(i);
                N.insert(row, col) = Ng(i, g);
            }
        }
    }
    return N;
}

/**
 * @brief Constructs a shape function matrix \f$ \mathbf{N} \f$ for a given mesh, i.e. at the
 * element quadrature points.
 * @tparam QuadratureOrder Quadrature order
 * @tparam TMesh Mesh type
 * @param mesh FEM mesh
 * @return `|# elements * # quad.pts.| x |# nodes|` shape function matrix
 */
template <int QuadratureOrder, CMesh TMesh>
auto ShapeFunctionMatrix(TMesh const& mesh)
    -> Eigen::SparseMatrix<typename TMesh::ScalarType, Eigen::RowMajor, typename TMesh::IndexType>
{
    return ShapeFunctionMatrix<
        typename TMesh::ElementType,
        QuadratureOrder,
        typename TMesh::ScalarType>(mesh.E, mesh.X.cols());
}

/**
 * @brief Constructs a shape function matrix \f$ \mathbf{N} \f$ for a given mesh, i.e. at the
 * element quadrature points.
 * @tparam TElement Element type
 * @tparam TDerivedE Eigen dense expression type for element indices
 * @tparam TDerivedXi Eigen dense expression type for quadrature points
 * @tparam TIndex Index type, defaults to coefficient type of `TDerivedE`
 * @param E `|# nodes| x |# quad.pts.|` array of element node indices
 * @param nNodes Number of nodes in the mesh
 * @param Xi `|# dims| x |# quad.pts.|` array of quadrature points in reference space
 * @return `|# quad.pts.| x |# nodes|` shape function matrix
 */
template <
    CElement TElement,
    class TDerivedE,
    class TDerivedXi,
    common::CIndex TIndex = typename TDerivedE::Scalar>
auto ShapeFunctionMatrixAt(
    Eigen::DenseBase<TDerivedE> const& E,
    TIndex nNodes,
    Eigen::MatrixBase<TDerivedXi> const& Xi)
    -> Eigen::SparseMatrix<typename TDerivedXi::Scalar, Eigen::RowMajor, TIndex>
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.fem.ShapeFunctionMatrix");
    using ElementType               = TElement;
    using ScalarType                = typename TDerivedXi::Scalar;
    using IndexType                 = TIndex;
    using IndexVectorType           = Eigen::Vector<IndexType, Eigen::Dynamic>;
    using SparseMatrixType          = Eigen::SparseMatrix<ScalarType, Eigen::RowMajor, IndexType>;
    auto const numberOfQuadPoints   = Xi.cols();
    auto constexpr kNodesPerElement = ElementType::kNodes;
    SparseMatrixType N(numberOfQuadPoints, nNodes);
    N.reserve(IndexVectorType::Constant(numberOfQuadPoints, kNodesPerElement));
    for (auto g = 0; g < numberOfQuadPoints; ++g)
    {
        auto const nodes = E.col(g);
        auto Ng          = ElementType::N(Xi.col(g));
        for (auto i = 0; i < kNodesPerElement; ++i)
        {
            N.insert(g, nodes(i)) = Ng(i);
        }
    }
    return N;
}

/**
 * @brief Constructs a shape function matrix \f$ \mathbf{N} \f$ for a given mesh, i.e. at the
 * given evaluation points.
 * @tparam TMesh Mesh type
 * @tparam TDerivedEg Eigen type
 * @tparam TDerivedXi Eigen type
 * @param mesh FEM mesh
 * @param eg `|# quad.pts.|` array of elements associated with quadrature points
 * @param Xi `|# dims|x|# quad.pts.|` array of quadrature points in reference space
 * @return `|# quad.pts.| x |# nodes|` shape function matrix
 */
template <CMesh TMesh, class TDerivedEg, class TDerivedXi>
auto ShapeFunctionMatrixAt(
    TMesh const& mesh,
    Eigen::DenseBase<TDerivedEg> const& eg,
    Eigen::MatrixBase<TDerivedXi> const& Xi)
    -> Eigen::SparseMatrix<typename TMesh::ScalarType, Eigen::RowMajor, typename TMesh::IndexType>
{
    using ScalarType = typename TMesh::ScalarType;
    using IndexType  = typename TMesh::IndexType;
    static_assert(
        std::is_same_v<ScalarType, typename TDerivedXi::Scalar>,
        "Scalar type of Xi must match mesh scalar type");
    return ShapeFunctionMatrixAt<typename TMesh::ElementType>(
        mesh.E(Eigen::placeholders::all, eg.derived()),
        static_cast<IndexType>(mesh.X.cols()),
        Xi.derived());
}

/**
 * @brief Compute shape functions at the given reference positions
 * @tparam TElement Element type
 * @tparam TDerivedXi Eigen dense expression type for reference positions
 * @param Xi `|# dims|x|# quad.pts.|` evaluation points
 * @return `|# element nodes| x |Xi.cols()|` matrix of nodal shape functions at reference
 * points Xi
 */
template <CElement TElement, class TDerivedXi>
auto ShapeFunctionsAt(Eigen::DenseBase<TDerivedXi> const& Xi)
    -> Eigen::Matrix<typename TDerivedXi::Scalar, TElement::kNodes, Eigen::Dynamic>
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.fem.ShapeFunctionsAt");
    using ElementType = TElement;
    using ScalarType  = typename TDerivedXi::Scalar;
    using MatrixType  = Eigen::Matrix<ScalarType, ElementType::kNodes, Eigen::Dynamic>;
    if (Xi.rows() != ElementType::kDims)
    {
        std::string const what = fmt::format(
            "Expected evaluation points in d={} dimensions, but got Xi.rows()={}",
            ElementType::kDims,
            Xi.rows());
        throw std::invalid_argument(what);
    }
    MatrixType N(ElementType::kNodes, Xi.cols());
    tbb::parallel_for(Index{0}, Index{Xi.cols()}, [&](Index i) {
        N.col(i) = ElementType::N(Xi.col(i));
    });
    return N;
}

/**
 * @brief Computes gradients \f$ \mathbf{G}(X) = \nabla_X \mathbf{N}_e(X) \f$ of FEM basis
 * functions. Given some FEM function
 * \f[
 * \begin{align*}
 * f(X) &=
 * \begin{bmatrix} \dots & \mathbf{f}_i & \dots \end{bmatrix}
 * \begin{bmatrix} \vdots \\ \phi_i(X) \\ \vdots \end{bmatrix} \\
 * &= \mathbf{F}_e \mathbf{N}_e(X)
 * \end{align*}
 * \f]
 * where \f$ \phi_i(X) = N_i(\xi(X)) \f$ are the nodal basis functions, we have that
 * \f[
 * \begin{align*}
 * \nabla_X f(X) &= \mathbf{F}_e \nabla_X \mathbf{N}_e(X) \\
 * &= \mathbf{F}_e \nabla_\xi \mathbf{N}_e(\xi(X)) \frac{\partial \xi}{\partial X}
 * \end{align*}
 * \f]
 * Recall that
 * \f[
 * \begin{align*}
 * \frac{\partial \xi}{\partial X} &= (\frac{\partial X}{\partial \xi})^{-1} \\
 * &= \left( \mathbf{X}_e \nabla_\xi \mathbf{N}_e \right)^{-1}
 * \end{align*}
 * \f]
 * where \f$ \mathbf{X}_e = \begin{bmatrix} \dots & \mathbf{X}_i & \dots \end{bmatrix} \f$ are the
 * element nodal positions. We thus have that
 * \f[
 * \mathbf{G}(X)^T = \nabla_X \mathbf{N}_e(X)^T = ( \mathbf{X}_e \nabla_\xi \mathbf{N}_e )^{-T}
 * \nabla_\xi \mathbf{N}_e^T
 * \f]
 * where the left-multiplication by \f$ ( \mathbf{X}_e \nabla_\xi \mathbf{N}_e )^{-T} \f$ is carried
 * out via LU/SVD decomposition for a square/rectangular Jacobian.
 *
 * @tparam TDerivedXi Eigen dense expression type for reference position
 * @tparam TDerivedX Eigen dense expression type for element nodal positions
 * @tparam TElement Element type
 * @param Xi `|# elem dims| x 1` point in reference element at which to evaluate the gradients
 * @param X `|# dims| x |# nodes|` element vertices, i.e. nodes of affine element
 * @return `|# nodes| x |# dims|` matrix of basis function gradients in rows
 */
template <CElement TElement, class TDerivedXi, class TDerivedX>
auto ElementShapeFunctionGradients(
    Eigen::MatrixBase<TDerivedXi> const& Xi,
    Eigen::MatrixBase<TDerivedX> const& X)
    -> Eigen::Matrix<typename TDerivedX::Scalar, TElement::kNodes, TDerivedX::RowsAtCompileTime>
{
    auto constexpr kInputDims  = TElement::kDims;
    auto constexpr kOutputDims = TDerivedX::RowsAtCompileTime;
    auto constexpr kNodes      = TElement::kNodes;
    using ScalarType           = typename TDerivedX::Scalar;
    static_assert(
        std::is_same_v<ScalarType, typename TDerivedXi::Scalar>,
        "Scalar type of Xi must match X's scalar type");
    static_assert(kOutputDims != Eigen::Dynamic, "Rows of X must be fixed at compile time");

    Eigen::Matrix<ScalarType, kNodes, kInputDims> const GN = TElement::GradN(Xi);
    Eigen::Matrix<ScalarType, kNodes, kOutputDims> GP;
    Eigen::Matrix<ScalarType, kOutputDims, kInputDims> J = X * GN;
    // Compute \nabla_\xi N (X \nabla_\xi N)^{-1}
    bool constexpr bIsJacobianSquare = kInputDims == kOutputDims;
    if constexpr (bIsJacobianSquare)
    {
        GP.transpose() = J.transpose().fullPivLu().solve(GN.transpose());
    }
    else
    {
        GP.transpose() =
            J.transpose().template jacobiSvd<Eigen::ComputeThinU | Eigen::ComputeThinV>().solve(
                GN.transpose());
    }
    return GP;
}

/**
 * @brief Computes nodal shape function gradients at each element quadrature points
 * @tparam TElement Element type
 * @tparam Dims Number of dimensions of the element
 * @tparam QuadratureOrder Quadrature order
 * @tparam TDerivedE Eigen dense expression type for element indices
 * @tparam TDerivedX Eigen dense expression type for element nodal positions
 * @tparam TScalar Floating point type, defaults to Scalar
 * @tparam TIndex Index type, defaults to coefficient type of `TDerivedE`
 * @param E `|# elem nodes| x |# elements|` array of element node indices
 * @param X `|# dims| x |# nodes|` array of mesh nodal positions
 * @return `|# elem nodes| x |# dims * # elem quad.pts. * # elems|` matrix of nodal shape function
 * gradients
 */
template <
    CElement TElement,
    int Dims,
    int QuadratureOrder,
    class TDerivedE,
    class TDerivedX,
    common::CFloatingPoint TScalar = typename TDerivedX::Scalar,
    common::CIndex TIndex          = typename TDerivedE::Scalar>
auto ShapeFunctionGradients(
    Eigen::MatrixBase<TDerivedE> const& E,
    Eigen::MatrixBase<TDerivedX> const& X)
    -> Eigen::Matrix<TScalar, TElement::kNodes, Eigen::Dynamic>
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.fem.ShapeFunctionGradients");
    using ScalarType = TScalar;
    using QuadratureRuleType =
        typename TElement::template QuadratureType<QuadratureOrder, ScalarType>;
    using ElementType = TElement;
    using MatrixType  = Eigen::Matrix<ScalarType, ElementType::kNodes, Eigen::Dynamic>;
    auto const Xg     = common::ToEigen(QuadratureRuleType::points)
                        .reshaped(QuadratureRuleType::kDims + 1, QuadratureRuleType::kPoints)
                        .template bottomRows<QuadratureRuleType::kDims>();
    auto const numberOfElements     = E.cols();
    auto constexpr kNodesPerElement = ElementType::kNodes;
    MatrixType GNe(kNodesPerElement, numberOfElements * Dims * QuadratureRuleType::kPoints);
    tbb::parallel_for(Index{0}, Index{numberOfElements}, [&](Index e) {
        auto const nodes = E.col(e);
        for (auto g = 0; g < QuadratureRuleType::kPoints; ++g)
        {
            auto const GP = ElementShapeFunctionGradients<ElementType>(
                Xg.col(g),
                X(Eigen::placeholders::all, nodes)
                    .template topLeftCorner<Dims, kNodesPerElement>());
            auto constexpr kStride = Dims * QuadratureRuleType::kPoints;
            GNe.template block<kNodesPerElement, Dims>(0, e * kStride + g * Dims) = GP;
        }
    });
    return GNe;
}

/**
 * @brief Computes nodal shape function gradients at each element quadrature points
 * @tparam Order Quadrature order
 * @tparam TMesh Mesh type
 * @param mesh FEM mesh
 * @return `|# element nodes| x |# dims * # quad.pts. * # elements|` matrix of shape function
 * gradients
 */
template <int QuadratureOrder, CMesh TMesh>
auto ShapeFunctionGradients(TMesh const& mesh)
    -> Eigen::Matrix<typename TMesh::ScalarType, TMesh::ElementType::kNodes, Eigen::Dynamic>
{
    using MeshType    = TMesh;
    using ElementType = typename MeshType::ElementType;
    return ShapeFunctionGradients<ElementType, MeshType::kDims, QuadratureOrder>(mesh.E, mesh.X);
}

/**
 * @brief Computes nodal shape function gradients at evaluation points Xi.
 * @tparam TElement Element type
 * @tparam Dims Number of dimensions of the element
 * @tparam TDerivedE Eigen dense expression type for element indices
 * @tparam TDerivedX Eigen dense expression type for element nodal positions
 * @tparam TDerivedXi Eigen dense expression type for evaluation points
 * @param Eg `|# elem nodes| x |# eval.pts.|` array of element node indices at evaluation points
 * @param X `|# dims| x |# elem nodes|` array of element nodal positions
 * @param Xi `|# dims| x |# eval.pts.|` evaluation points in reference space
 * @return `|# elem nodes| x |# dims * # eval.pts.|` matrix of nodal shape function gradients
 */
template <CElement TElement, int Dims, class TDerivedEg, class TDerivedX, class TDerivedXi>
auto ShapeFunctionGradientsAt(
    Eigen::DenseBase<TDerivedEg> const& Eg,
    Eigen::MatrixBase<TDerivedX> const& X,
    Eigen::MatrixBase<TDerivedXi> const& Xi)
    -> Eigen::Matrix<typename TDerivedXi::Scalar, TElement::kNodes, Eigen::Dynamic>
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.fem.ShapeFunctionGradientsAt");
    using ElementType = TElement;
    using ScalarType  = typename TDerivedXi::Scalar;
    using MatrixType  = Eigen::Matrix<ScalarType, ElementType::kNodes, Eigen::Dynamic>;
    if (Eg.cols() != Xi.cols())
    {
        std::string const what = fmt::format(
            "Expected Eg.cols() == Xi.cols(), but got Eg.cols()={} and Xi.cols()={}",
            Eg.cols(),
            Xi.cols());
        throw std::invalid_argument(what);
    }
    auto const numberOfEvaluationPoints = Xi.cols();
    MatrixType GNe(ElementType::kNodes, numberOfEvaluationPoints * Dims);
    tbb::parallel_for(Index{0}, Index{numberOfEvaluationPoints}, [&](Index g) {
        auto const nodes = Eg.col(g);
        auto GP          = ElementShapeFunctionGradients<ElementType>(
            Xi.col(g),
            X(Eigen::placeholders::all, nodes).template topLeftCorner<Dims, ElementType::kNodes>());
        GNe.template block<ElementType::kNodes, Dims>(0, g * Dims) = GP;
    });
    return GNe;
}

/**
 * @brief Computes nodal shape function gradients at evaluation points Xg.
 * @tparam TMesh Mesh type
 * @tparam TDerivedEg Eigen dense expression type for element indices
 * @tparam TDerivedXi Eigen dense expression type for evaluation points
 * @param mesh FEM mesh
 * @param E `|# eval.pts.|` array of elements associated with evaluation points
 * @param Xi `|# dims|x|# eval.pts.|` evaluation points
 * @return `|# element nodes| x |eg.size() * mesh.dims|` nodal shape function gradients at
 * evaluation points Xi
 */
template <CMesh TMesh, class TDerivedEg, class TDerivedXi>
auto ShapeFunctionGradientsAt(
    TMesh const& mesh,
    Eigen::DenseBase<TDerivedEg> const& eg,
    Eigen::MatrixBase<TDerivedXi> const& Xi)
    -> Eigen::Matrix<typename TMesh::ScalarType, TMesh::ElementType::kNodes, Eigen::Dynamic>
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.fem.ShapeFunctionGradientsAt");
    using ScalarType  = typename TMesh::ScalarType;
    using MeshType    = TMesh;
    using ElementType = typename MeshType::ElementType;
    static_assert(
        std::is_same_v<ScalarType, typename TDerivedXi::Scalar>,
        "Scalar type of Xi must match mesh scalar type");
    return ShapeFunctionGradientsAt<ElementType, MeshType::kDims>(
        mesh.E(Eigen::placeholders::all, eg.derived()),
        mesh.X,
        Xi.derived());
}

} // namespace pbat::fem

#endif // PBAT_FEM_SHAPEFUNCTIONS_H
