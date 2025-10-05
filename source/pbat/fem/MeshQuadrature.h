/**
 * @file MeshQuadrature.h
 * @brief Utility functions computing common mesh quadrature quantities.
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @date 2025-02-11
 * @copyright Copyright (c) 2025
 */

#ifndef PBAT_FEM_MESHQUADRATURE_H
#define PBAT_FEM_MESHQUADRATURE_H

#include "Concepts.h"
#include "Jacobian.h"
#include "ShapeFunctions.h"
#include "pbat/Aliases.h"
#include "pbat/common/Concepts.h"

#include <exception>
#include <fmt/format.h>

namespace pbat::fem {

/**
 * @brief Computes the inner product weights \f$ \mathbf{w}_{ge} \in \mathbb{R}^{|G^e| \times |E|}
 * \f$ such that \f$ \int_\Omega \cdot d\Omega = \sum_e \sum_g w_{ge} \cdot \f$.
 *
 * In other words, \f$ w_{ge} = w_g \det(J^e_g) \f$ where \f$ J^e_g \f$ is the Jacobian of the
 * element map at the \f$ g^\text{{th} \f$ quadrature point and \f$ w_g \f$ is the \f$ g^\text{th}
 * \f$ quadrature weight.
 *
 * @tparam QuadratureOrder Quadrature order
 * @tparam TMesh Mesh type
 * @param mesh FEM mesh
 * @param detJeThenWg `|# quad.pts.| x |# elements|` matrix of jacobian determinants at element
 * quadrature points
 * @return `|# quad.pts.|x|# elements|` matrix of quadrature weights multiplied by jacobian
 * determinants at element quadrature points
 */
template <CElement TElement, int QuadratureOrder, class TDerivedDetJe>
void ToMeshQuadratureWeights(Eigen::MatrixBase<TDerivedDetJe>& detJeThenWg)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.fem.ToMeshQuadratureWeights");
    using ScalarType     = typename TDerivedDetJe::Scalar;
    using QuadratureType = typename TElement::template QuadratureType<QuadratureOrder, ScalarType>;
    if (detJeThenWg.rows() != QuadratureType::kPoints)
    {
        throw std::invalid_argument(
            fmt::format(
                "detJeThenWg must have {} rows, but has {}",
                QuadratureType::kPoints,
                detJeThenWg.rows()));
    }
    auto const wg = common::ToEigen(QuadratureType::weights);
    detJeThenWg.array().colwise() *= wg.array();
}

/**
 * @brief Computes the inner product weights \f$ \mathbf{w}_{ge} \in \mathbb{R}^{|G^e| \times |E|}
 * \f$ such that \f$ \int_\Omega \cdot d\Omega = \sum_e \sum_g w_{ge} \cdot \f$.
 *
 * In other words, \f$ w_{ge} = w_g \det(J^e_g) \f$ where \f$ J^e_g \f$ is the Jacobian of the
 * element map at the \f$ g^\text{{th} \f$ quadrature point and \f$ w_g \f$ is the \f$ g^\text{th}
 * \f$ quadrature weight.
 *
 * @tparam TElement FEM element type
 * @tparam QuadratureOrder Quadrature order
 * @tparam TDerivedE Eigen matrix expression for element matrix
 * @tparam TDerivedX Eigen matrix expression for mesh nodal positions
 * @param E `|# elem nodes| x |# elems|` mesh element matrix
 * @param X `|# dims| x |# nodes|` mesh nodal position matrix
 * @return `|# quad.pts.| x |# elements|` matrix of quadrature weights multiplied by jacobian
 * determinants at element quadrature points
 */
template <CElement TElement, int QuadratureOrder, class TDerivedE, class TDerivedX>
auto MeshQuadratureWeights(
    Eigen::MatrixBase<TDerivedE> const& E,
    Eigen::MatrixBase<TDerivedX> const& X)
    -> Eigen::Matrix<
        typename TDerivedX::Scalar,
        TElement::template QuadratureType<QuadratureOrder, typename TDerivedX::Scalar>::kPoints,
        Eigen::Dynamic>
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.fem.MeshQuadratureWeights");
    using ElementType = TElement;
    auto detJeThenWg = DeterminantOfJacobian<ElementType, QuadratureOrder>(E, X);
    ToMeshQuadratureWeights<ElementType, QuadratureOrder>(detJeThenWg);
    return detJeThenWg;
}

/**
 * @brief Computes the inner product weights \f$ \mathbf{w}_{ge} \in \mathbb{R}^{|G^e| \times |E|}
 * \f$ such that \f$ \int_\Omega \cdot d\Omega = \sum_e \sum_g w_{ge} \cdot \f$.
 *
 * In other words, \f$ w_{ge} = w_g \det(J^e_g) \f$ where \f$ J^e_g \f$ is the Jacobian of the
 * element map at the \f$ g^\text{{th} \f$ quadrature point and \f$ w_g \f$ is the \f$ g^\text{th}
 * \f$ quadrature weight.
 *
 * @tparam QuadratureOrder Quadrature order
 * @tparam TMesh Mesh type
 * @param mesh FEM mesh
 * @return `|# quad.pts.|x|# elements|` matrix of quadrature weights multiplied by jacobian
 * determinants at element quadrature points
 */
template <int QuadratureOrder, CMesh TMesh>
auto MeshQuadratureWeights(TMesh const& mesh) -> Eigen::Matrix<
    typename TMesh::ScalarType,
    TMesh::ElementType::template QuadratureType<QuadratureOrder, typename TMesh::ScalarType>::
        kPoints,
    Eigen::Dynamic>
{
    return MeshQuadratureWeights<typename TMesh::ElementType, QuadratureOrder>(mesh.E, mesh.X);
}

/**
 * @brief Computes the element quadrature points indices for each quadrature point,
 * given the number of elements and the quadrature weights matrix.
 *
 * @tparam TIndex Index type
 * @tparam TDerivedwg Eigen matrix expression for quadrature weights
 * @param nElements Number of elements
 * @param wg `|# quad.pts.| x |# elems|` mesh quadrature weights matrix
 * @return `|# quad.pts.| x |# elems|` matrix of element indices at quadrature points
 */
template <common::CIndex TIndex>
auto MeshQuadratureElements(TIndex nElements, TIndex nQuadPtsPerElement)
{
    using IndexVectorType = Eigen::Vector<TIndex, Eigen::Dynamic>;
    return IndexVectorType::LinSpaced(nElements, TIndex(0), nElements - 1)
        .replicate(TIndex(1), nQuadPtsPerElement)
        .transpose();
}

/**
 * @brief Computes the element quadrature points indices for each quadrature point,
 * given the number of elements and the quadrature weights matrix.
 *
 * @tparam TIndex Index type
 * @tparam TDerivedwg Eigen matrix expression for quadrature weights
 * @param nElements Number of elements
 * @param wg `|# quad.pts.| x |# elems|` mesh quadrature weights matrix
 * @return `|# quad.pts.| x |# elems|` matrix of element indices at quadrature points
 */
template <common::CIndex TIndex, class TDerivedwg>
auto MeshQuadratureElements(TIndex nElements, Eigen::DenseBase<TDerivedwg> const& wg)
{
    bool const bAreDimensionsCorrect = (wg.size() % nElements) == 0;
    if (not bAreDimensionsCorrect)
    {
        throw std::invalid_argument(
            fmt::format(
                "nElements must match the number of columns in wg, or wg's flattened size must be "
                "a multiple of nElements but got {} and {}",
                nElements,
                wg.size()));
    }
    auto const nQuadPtsPerElement = wg.size() / nElements;
    return MeshQuadratureElements(nElements, nQuadPtsPerElement);
}

/**
 * @brief Computes the element quadrature points indices for each quadrature point.
 *
 * @tparam TDerivedE Eigen matrix expression for element matrix
 * @tparam TDerivedwg Eigen matrix expression for quadrature weights
 * @param E `|# elem nodes| x |# elems|` mesh element matrix
 * @param wg `|# quad.pts.| x |# elems|` mesh quadrature weights matrix
 * @return `|# quad.pts.| x |# elems|` matrix of element indices at quadrature points
 */
template <class TDerivedE, class TDerivedwg>
auto MeshQuadratureElements(
    Eigen::DenseBase<TDerivedE> const& E,
    Eigen::DenseBase<TDerivedwg> const& wg)
{
    return MeshQuadratureElements(E.cols(), wg);
}

/**
 * @brief Computes the element quadrature points in reference element space.
 *
 * @tparam TElement FEM element type
 * @tparam QuadratureOrder Quadrature order
 * @tparam TScalar Scalar type
 * @tparam TIndex Index type
 * @param nElements Number of elements
 * @return `|# ref. dims| x |# quad.pts. * # elems|` matrix of quadrature points in reference
 * element space
 */
template <
    CElement TElement,
    int QuadratureOrder,
    common::CArithmetic TScalar = Scalar,
    common::CIndex TIndex       = Index>
auto MeshReferenceQuadraturePoints(TIndex nElements)
{
    using QuadratureType = typename TElement::template QuadratureType<QuadratureOrder, TScalar>;
    auto const Xi =
        common::ToEigen(QuadratureType::points).template bottomRows<QuadratureType::kDims>();
    return Xi.replicate(1, nElements);
}

/**
 * @brief Computes the element quadrature points in reference element space.
 *
 * @tparam QuadratureOrder Quadrature order
 * @tparam TMesh Mesh type
 * @param mesh FEM mesh
 * @return `|# ref. dims| x |# quad.pts. * # elems|` matrix of quadrature points in reference
 * element space
 */
template <int QuadratureOrder, CMesh TMesh>
auto MeshReferenceQuadraturePoints(TMesh const& mesh)
{
    using ElementType = typename TMesh::ElementType;
    using ScalarType  = typename TMesh::ScalarType;
    return MeshReferenceQuadraturePoints<ElementType, QuadratureOrder, ScalarType>(mesh.E.cols());
}

} // namespace pbat::fem

#endif // PBAT_FEM_MESHQUADRATURE_H