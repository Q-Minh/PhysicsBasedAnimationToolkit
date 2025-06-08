/**
 * @file ShapeFunctions.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief FEM shape functions and gradients
 * @date 2025-02-11
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_FEM_SHAPE_FUNCTIONS_H
#define PBAT_FEM_SHAPE_FUNCTIONS_H

#include "Concepts.h"
#include "Jacobian.h"
#include "pbat/Aliases.h"
#include "pbat/common/Concepts.h"
#include "pbat/profiling/Profiling.h"

#include <Eigen/Cholesky>
#include <Eigen/LU>
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
 * @return The shape function values of each node at quadrature points, stored in a matrix
 * of dimensions `|# element nodes| x |# quad.pts.|`
 */
template <CElement TElement, int QuadratureOrder, common::CFloatingPoint TScalar = Scalar>
auto ShapeFunctions() -> Eigen::Matrix<
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
 * @param TScalar Floating point type, defaults to Scalar
 * @param TIndex Index type, defaults to Index
 * @param E `|# nodes|x|# elements|` array of element node indices
 * @param nNodes Number of nodes in the mesh
 * @return `|# elements * # quad.pts.| x |# nodes|` shape function matrix
 */
template <
    CElement TElement,
    int QuadratureOrder,
    common::CFloatingPoint TScalar = Scalar,
    common::CIndex TIndex          = Index>
auto ShapeFunctionMatrix(
    Eigen::Ref<Eigen::Matrix<TIndex, TElement::kNodes, Eigen::Dynamic> const> const& E,
    TIndex nNodes) -> Eigen::SparseMatrix<TScalar, Eigen::RowMajor, TIndex>
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.fem.ShapeFunctionMatrix");
    using ElementType                = TElement;
    using ScalarType                 = TScalar;
    using IndexType                  = TIndex;
    using IndexVectorType            = Eigen::Vector<IndexType, Eigen::Dynamic>;
    auto const Ng                    = ShapeFunctions<ElementType, QuadratureOrder, ScalarType>();
    auto const nElements             = E.cols();
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
        typename TMesh::ScalarType,
        typename TMesh::IndexType>(mesh.E, mesh.X.cols());
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
 * @brief Computes gradients of FEM basis functions in reference element.
 *
 * Only works for linear maps, but we do not emit an error when `TElement::bHasConstantJacobian` is
 * `false`, since the element's function space might be non-linear, while its current configuration
 * induces a linear map. I.e., a \f$ k^\text{th} \f$ order element that is purely a rigid
 transformation on the
 * reference element still induces a linear map, even if the element's function space is \f$
 k^\text{th} \f$ order.
 * It is up to the user to give the right inputs, and we cannot/won't check those.
 *
 * @note
 * Since \f$ \phi(X) = N(\xi(X)) \f$, to be mathematically precise, what we should compute is:
 * \f{eqnarray*}{
 * \nabla \phi(X) &= \nabla \xi N(\xi(X)) J_X \xi(X) \\
 *                &= \nabla N * J_X \xi(X)
 * \f}
 * This requires the Jacobian of the inverse map taking domain element to reference element.
 * Because this map is potentially non-linear, we compute it via Gauss-Newton iterations in
 * Jacobian.h. Hence, to get the jacobian of that map, we also need to compute derivatives of
 * the Gauss-Newton iterations in Jacobian.h.
 * @note
 * However, we assume that domain elements are linear transformations of reference elements,
 * so that the inverse map is linear, i.e. the Jacobian is constant. Hence,
 * \f{eqnarray*}{
 * X(\xi) &= X * N(\xi) \\
 * J &= X * \nabla_\xi N \\
 * X(\xi) &= X_0 + J*\xi \\
 * J \xi &= X - X_0
 * \f}
 * @note
 * If J is square, then
 * \f{eqnarray*}{
 * \xi &= J^{-1} (X - X_0) \\
 * \nabla_X N(\xi(X)) &= \nabla_\xi N * J^{-1} \\
 * \nabla_X \phi^T  &= J^{-T} \left[ \nabla_\xi N \right]^T
 * \f}
 * @note
 * If J is rectangular, then
 * \f{eqnarray*}{
 * (J^T J) \xi      &= J^T (X - X_0) \\
 * \xi              &= (J^T J)^{-1} J^T (X - X_0) \\
 * \nabla_X N(\xi(X)) &= \nabla_\xi N * (J^T J)^{-1} J^T \\
 * \left[ \nabla_X \phi \right]^T  &= J (J^T J)^{-1} \left[ \nabla_\xi N \right]^T
 * \f}
 * @note
 * For non-linear elements, like hexahedra or quadrilaterals, the accuracy of the gradients
 * might be unacceptable, but will be exact, if domain hex or quad elements are linear
 * transformations on reference hex/quad elements. This is the case for axis-aligned elements,
 * for example, which would arise when constructing a mesh from an octree or quadtree.
 *
 * @tparam TDerivedXi Eigen dense expression type for reference position
 * @tparam TDerivedX Eigen dense expression type for element nodal positions
 * @tparam TElement Element type
 * @param Xi `|# elem dims| x 1` point in reference element at which to evaluate the gradients
 * @param X `|# dims| x |# nodes|` element vertices, i.e. nodes of affine element
 * @return `|# nodes|x|Dims|` matrix of basis function gradients in rows
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
    using AffineElementType    = typename TElement::AffineBaseType;
    using ScalarType           = typename TDerivedX::Scalar;
    using MatrixType           = Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic>;
    static_assert(
        std::is_same_v<ScalarType, typename TDerivedXi::Scalar>,
        "Scalar type of Xi must match X's scalar type");
    static_assert(kOutputDims != Eigen::Dynamic, "Rows of X must be fixed at compile time");

    Eigen::Matrix<ScalarType, kNodes, kInputDims> const GN = TElement::GradN(Xi);
    Eigen::Matrix<ScalarType, kOutputDims, kInputDims> J;
    Eigen::Matrix<ScalarType, kNodes, kOutputDims> GP;

    bool constexpr bIsElementLinear = std::is_same_v<TElement, AffineElementType>;
    if constexpr (bIsElementLinear)
    {
        J = X * GN;
    }
    else
    {
        J = X * AffineElementType::GradN(Xi);
    }
    bool constexpr bIsJacobianSquare = kInputDims == kOutputDims;
    if (bIsJacobianSquare)
    {
        GP.transpose() = J.transpose().fullPivLu().solve(GN.transpose());
    }
    else
    {
        Eigen::Matrix<ScalarType, kInputDims, kInputDims> const JTJ = J.transpose() * J;
        GP.transpose() = J * JTJ.ldlt().solve(GN.transpose());
    }
    return GP;
}

/**
 * @brief Computes nodal shape function gradients at each element quadrature points
 * @tparam TElement Element type
 * @tparam Dims Number of dimensions of the element
 * @tparam QuadratureOrder Quadrature order
 * @tparam TDerivedE Eigen dense expression type for element indices
 * @tparam TScalar Floating point type, defaults to Scalar
 * @tparam TIndex Index type, defaults to coefficient type of `TDerivedE`
 * @param E `|# elem nodes| x |# elements|` array of element node indices
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
    using ElementType       = TElement;
    using AffineElementType = typename ElementType::AffineBaseType;
    using MatrixType        = Eigen::Matrix<ScalarType, ElementType::kNodes, Eigen::Dynamic>;
    auto const Xg           = common::ToEigen(QuadratureRuleType::points)
                        .reshaped(QuadratureRuleType::kDims + 1, QuadratureRuleType::kPoints)
                        .template bottomRows<QuadratureRuleType::kDims>();
    auto const numberOfElements     = E.cols();
    auto constexpr kNodesPerElement = ElementType::kNodes;
    MatrixType GNe(kNodesPerElement, numberOfElements * Dims * QuadratureRuleType::kPoints);
    tbb::parallel_for(Index{0}, Index{numberOfElements}, [&](Index e) {
        auto const nodes                                   = E.col(e);
        auto const vertices                                = nodes(ElementType::Vertices);
        auto constexpr kRowsJ                              = Dims;
        auto constexpr kColsJ                              = AffineElementType::kNodes;
        Eigen::Matrix<ScalarType, kRowsJ, kColsJ> const Ve = X(Eigen::placeholders::all, vertices);
        for (auto g = 0; g < QuadratureRuleType::kPoints; ++g)
        {
            auto const GP          = ElementShapeFunctionGradients<ElementType>(Xg.col(g), Ve);
            auto constexpr kStride = Dims * QuadratureRuleType::kPoints;
            GNe.block<kNodesPerElement, Dims>(0, e * kStride + g * Dims) = GP;
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
    using ScalarType  = typename MeshType::ScalarType;
    using ElementType = typename MeshType::ElementType;
    return ShapeFunctionGradients<ElementType, MeshType::kDims, QuadratureOrder>(mesh.E, mesh.X);
}

/**
 * @brief Computes nodal shape function gradients at evaluation points Xi.
 * @param TElement Element type
 * @param Dims Number of dimensions of the element
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
    using ElementType       = TElement;
    using AffineElementType = typename ElementType::AffineBaseType;
    using ScalarType        = typename TDerivedXi::Scalar;
    using MatrixType        = Eigen::Matrix<ScalarType, ElementType::kNodes, Eigen::Dynamic>;
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
        auto const nodes                                   = Eg.col(g);
        auto const vertices                                = nodes(ElementType::Vertices);
        auto constexpr kRowsJ                              = Dims;
        auto constexpr kColsJ                              = AffineElementType::kNodes;
        Eigen::Matrix<ScalarType, kRowsJ, kColsJ> const Ve = X(Eigen::placeholders::all, vertices);
        auto GP = ElementShapeFunctionGradients<ElementType>(Xi.col(g), Ve);
        GNe.block<ElementType::kNodes, Dims>(0, g * Dims) = GP;
    });
    return GNe;
}

/**
 * @brief Computes nodal shape function gradients at evaluation points Xg.
 * @tparam TDerivedEg Eigen dense expression type for element indices
 * @tparam TDerivedXi Eigen dense expression type for evaluation points
 * @tparam TMesh Mesh type
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

#endif // PBAT_FEM_SHAPE_FUNCTIONS_H
