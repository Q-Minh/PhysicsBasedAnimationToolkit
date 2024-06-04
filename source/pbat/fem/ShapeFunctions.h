#ifndef PBAT_FEM_SHAPE_FUNCTIONS_H
#define PBAT_FEM_SHAPE_FUNCTIONS_H

#include "Concepts.h"
#include "Jacobian.h"
#include "pbat/aliases.h"
#include "pbat/profiling/Profiling.h"

#include <Eigen/SVD>
#include <tbb/parallel_for.h>

namespace pbat {
namespace fem {

/**
 * @brief Computes shape functions at element quadrature points for a polynomial quadrature rule of
 * order QuadratureOrder
 * @tparam TElement
 * @tparam QuadratureOrder
 * @return The shape function values of each node at quadrature points, stored in a matrix
 * of dimensions |#element nodes| x |#quad.pts.|
 */
template <CElement TElement, int QuadratureOrder>
Matrix<TElement::kNodes, TElement::template QuadratureType<QuadratureOrder>::kPoints>
ShapeFunctions()
{
    using QuadratureRuleType = TElement::template QuadratureType<QuadratureOrder>;
    using ElementType        = TElement;
    auto const Xg            = common::ToEigen(QuadratureRuleType::points)
                        .reshaped(QuadratureRuleType::kDims + 1, QuadratureRuleType::kPoints)
                        .bottomRows<QuadratureRuleType::kDims>();
    Matrix<ElementType::kNodes, QuadratureRuleType::kPoints> Ng{};
    for (auto g = 0; g < QuadratureRuleType::kPoints; ++g)
    {
        Ng.col(g) = ElementType::N(Xg.col(g));
    }
    return Ng;
}

/**
 * @brief Computes gradients of FEM basis functions in reference element
 * @tparam TDerivedXi
 * @tparam TDerivedX
 * @tparam TElement
 * @param Xi Point in reference element at which to evaluate the gradients
 * @param X Element vertices, i.e. nodes of affine element
 * @return |#nodes|x|Dims| matrix of basis function gradients in rows
 */
template <CElement TElement, class TDerivedXi, class TDerivedX>
Matrix<TElement::kNodes, TDerivedX::RowsAtCompileTime> ShapeFunctionGradients(
    Eigen::MatrixBase<TDerivedXi> const& Xi,
    Eigen::MatrixBase<TDerivedX> const& X)
{
    // \phi(X) = N(J^{-1} X) = N(\Xi)
    // grad_X \phi(X) = d N(\Xi) / d\Xi d \Xi / dX
    //                = grad_\Xi N * J^{-1}
    // If we transpose that equation, we get
    // [ grad_X \phi(X) ]^T = J^{-T} * grad_\Xi N^T
    // Recall that the pseudoinverse of J is J^{-1} = U \Sigma^{-1} V^T
    // We pseudoinvert its transpose directly, J^{-T} = V \Sigma^{-1} U^T
    auto constexpr kInputDims                     = TElement::kDims;
    auto constexpr kOutputDims                    = TDerivedX::RowsAtCompileTime;
    using AffineElementType                       = typename TElement::AffineBaseType;
    Matrix<TElement::kNodes, kInputDims> const GN = TElement::GradN(Xi);
    Matrix<kInputDims, kOutputDims> const JT      = Jacobian<AffineElementType>(Xi, X).transpose();
    auto JinvT = JT.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
    Matrix<TElement::kNodes, kOutputDims> const GP = JinvT.solve(GN.transpose()).transpose();
    return GP;
}

/**
 * @brief Computes nodal shape function gradients at each element quadrature point.
 * @tparam Order
 * @tparam TMesh
 * @param mesh
 * @return |ElementType::kNodes| x |MeshType::kDims *
 * ElementType::QuadratureType<QuadratureOrder>::kPoints * #elements| matrix of shape functions
 */
template <int QuadratureOrder, CMesh TMesh>
MatrixX ShapeFunctionGradients(TMesh const& mesh)
{
    PBA_PROFILE_SCOPE;
    using MeshType              = TMesh;
    using ElementType           = typename MeshType::ElementType;
    using QuadratureRuleType    = typename ElementType::template QuadratureType<QuadratureOrder>;
    using AffineElementType     = typename ElementType::AffineBaseType;
    auto const numberOfElements = mesh.E.cols();
    auto constexpr kNodesPerElement = ElementType::kNodes;
    auto const Xg                   = common::ToEigen(QuadratureRuleType::points)
                        .reshaped(QuadratureRuleType::kDims + 1, QuadratureRuleType::kPoints)
                        .bottomRows<ElementType::kDims>();
    MatrixX GNe(kNodesPerElement, numberOfElements * MeshType::kDims * QuadratureRuleType::kPoints);
    tbb::parallel_for(Index{0}, Index{numberOfElements}, [&](Index e) {
        auto const nodes                = mesh.E.col(e);
        auto const vertices             = nodes(ElementType::Vertices);
        auto constexpr kRowsJ           = MeshType::kDims;
        auto constexpr kColsJ           = AffineElementType::kNodes;
        Matrix<kRowsJ, kColsJ> const Ve = mesh.X(Eigen::all, vertices);
        for (auto g = 0; g < QuadratureRuleType::kPoints; ++g)
        {
            auto const gradPhi     = ShapeFunctionGradients<ElementType>(Xg.col(g), Ve);
            auto constexpr kStride = MeshType::kDims * QuadratureRuleType::kPoints;
            GNe.block<kNodesPerElement, MeshType::kDims>(0, e * kStride + g * MeshType::kDims) =
                gradPhi;
        }
    });
    return GNe;
}

} // namespace fem
} // namespace pbat

#endif // PBAT_FEM_SHAPE_FUNCTIONS_H
