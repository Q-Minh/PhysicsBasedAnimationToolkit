/**
 * @file LoadVector.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Functions to compute load vectors for FEM elements.
 * @date 2025-02-11
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_FEM_LOAD_VECTOR_H
#define PBAT_FEM_LOAD_VECTOR_H

#include "Concepts.h"
#include "ShapeFunctions.h"
#include "pbat/profiling/Profiling.h"

#include <Eigen/Core>

namespace pbat::fem {

/**
 * @brief Computes the load vector for a given FEM mesh.
 *
 * @tparam TElement Element type
 * @tparam TDerivedE Eigen matrix expression for element matrix
 * @tparam TDerivedeg Eigen vector expression for elements at quadrature points
 * @tparam TDerivedwg Eigen vector expression for quadrature weights
 * @tparam TDerivedNeg Eigen matrix expression for nodal shape function values at quadrature points
 * @tparam TDerivedFeg Eigen matrix expression for external load values at quadrature points
 * @param E `|# elem. nodes| x |# elements|` element matrix
 * @param nNodes Number of mesh nodes
 * @param eg `|# quad.pts.| x 1` array of elements associated with quadrature points
 * @param wg `|# quad.pts.| x 1` quadrature weights
 * @param Neg `|# elem. nodes| x |# quad.pts.|` nodal shape function values at quadrature points
 * @param Feg `|# dims| x |# quad.pts.|` load vector values at quadrature points
 * @return `|# dims| x |# nodes|` matrix s.t. each output dimension's load vectors are stored in
 * rows
 */
template <
    CElement TElement,
    class TDerivedE,
    class TDerivedeg,
    class TDerivedwg,
    class TDerivedNeg,
    class TDerivedFeg>
auto LoadVectors(
    Eigen::DenseBase<TDerivedE> const& E,
    Eigen::Index nNodes,
    Eigen::DenseBase<TDerivedeg> const& eg,
    Eigen::DenseBase<TDerivedwg> const& wg,
    Eigen::MatrixBase<TDerivedNeg> const& Neg,
    Eigen::MatrixBase<TDerivedFeg> const& Feg)
    -> Eigen::Matrix<typename TDerivedFeg::Scalar, TDerivedFeg::RowsAtCompileTime, Eigen::Dynamic>
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.fem.LoadVector");
    using ScalarType  = typename TDerivedFeg::Scalar;
    using ElementType = TElement;

    using MatrixType =
        Eigen::Matrix<typename TDerivedFeg::Scalar, TDerivedFeg::RowsAtCompileTime, Eigen::Dynamic>;
    MatrixType F = MatrixType::Zero(Feg.rows(), nNodes);
    for (auto g = 0; g < wg.size(); ++g)
    {
        auto const nodes = E.col(eg(g));
        F(Eigen::placeholders::all, nodes) += wg(g) * Feg.col(g) * Neg.col(g).transpose();
    }
    return F;
}

} // namespace pbat::fem

#endif // PBAT_FEM_LOAD_VECTOR_H