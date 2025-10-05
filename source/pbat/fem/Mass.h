/**
 * @file Mass.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief FEM mass matrix operator
 * @date 2025-02-11
 *
 * @details The mass matrix \f$ \mathbf{M}_{ij} = \int_\Omega \rho(X) \phi_i(X) \phi_j(X) \f$
 * of a finite element discretized function under Galerkin projection.
 *
 * This file provides functions to compute mass matrix related quantities.
 *
 * @copyright Copyright (c) 2025
 */

#ifndef PBAT_FEM_MASS_H
#define PBAT_FEM_MASS_H

#include "Concepts.h"
#include "ShapeFunctions.h"
#include "pbat/Aliases.h"
#include "pbat/common/Eigen.h"
#include "pbat/profiling/Profiling.h"

#include <exception>
#include <fmt/core.h>
#include <tbb/parallel_for.h>

namespace pbat::fem {

/**
 * @brief Compute the element mass matrix at a single quadrature point.
 *
 * @tparam TElement Element type
 * @tparam TN Type of the shape function vector at the quadrature point
 * @tparam TScalar Scalar type (e.g., double)
 * @param N Shape function vector at the quadrature point (size: |# nodes per element|)
 * @param w Quadrature weight (including Jacobian determinant)
 * @param rho Density at the quadrature point
 * @return Element mass matrix (size: |# nodes per element| x |# nodes per element|)
 */
template <typename TElement, typename TN>
inline auto
ElementMassMatrix(Eigen::MatrixBase<TN> const& N, typename TN::Scalar w, typename TN::Scalar rho)
{
    static_assert(
        TElement::kNodes == TN::RowsAtCompileTime || TN::RowsAtCompileTime == Eigen::Dynamic,
        "Shape function vector size must match number of element nodes");
    return w * rho * N * N.transpose();
}

/**
 * @brief Compute a matrix of horizontally stacked element mass matrices for all quadrature points.
 *
 * @tparam TElement Element type
 * @tparam TN Type of the shape function vector at the quadrature point
 * @tparam TDerivedwg Type of the quadrature weights
 * @tparam TDerivedrhog Type of the density at quadrature points
 * @tparam TDerivedOut Type of the output matrix
 * @param Neg `|# nodes per element| x |# quad.pts.|` shape function matrix at all quadrature points
 * @param wg `|# quad.pts.| x 1` quadrature weights (including Jacobian determinant)
 * @param rhog `|# quad.pts.| x 1` mass density at quadrature points
 * @param M `|# elem nodes| x |# elem nodes * # quad.pts.|` out matrix of stacked element mass
 * matrices
 */
template <
    typename TElement,
    typename TN,
    typename TDerivedwg,
    typename TDerivedrhog,
    typename TDerivedOut>
inline void ToElementMassMatrices(
    Eigen::MatrixBase<TN> const& Neg,
    Eigen::DenseBase<TDerivedwg> const& wg,
    Eigen::DenseBase<TDerivedrhog> const& rhog,
    Eigen::PlainObjectBase<TDerivedOut>& Meg)
{
    constexpr int kNodes = TElement::kNodes;
    const auto nQuadPts  = Neg.cols();
    Meg.resize(kNodes, kNodes * nQuadPts);
    for (Eigen::Index g = 0; g < nQuadPts; ++g)
    {
        auto const N                                      = Neg.col(g);
        auto const w                                      = wg(g);
        auto const rho                                    = rhog(g);
        auto Me                                           = w * rho * N * N.transpose();
        Meg.template block<kNodes, kNodes>(0, g * kNodes) = Me;
    }
}

/**
 * @brief Compute a matrix of horizontally stacked element mass matrices for all quadrature points
 *
 * @tparam TElement Element type
 * @tparam TN Type of the shape function vector at the quadrature point
 * @tparam TDerivedwg Type of the quadrature weights
 * @tparam TDerivedrhog Type of the density at quadrature points
 * @param Neg `|# nodes per element| x |# quad.pts.|` shape function matrix at all quadrature points
 * @param wg `|# quad.pts.| x 1` quadrature weights (including Jacobian determinant)
 * @param rhog `|# quad.pts.| x 1` mass density at quadrature points
 * @return `|# elem nodes| x |# elem nodes * # quad.pts.|` matrix of stacked element mass matrices
 */
template <typename TElement, typename TN, typename TDerivedwg, typename TDerivedrhog>
inline auto ElementMassMatrices(
    Eigen::MatrixBase<TN> const& Neg,
    Eigen::DenseBase<TDerivedwg> const& wg,
    Eigen::DenseBase<TDerivedrhog> const& rhog)
{
    using ScalarType     = typename TN::Scalar;
    constexpr int kNodes = TElement::kNodes;
    Eigen::Matrix<ScalarType, kNodes, Eigen::Dynamic> Meg;
    ToElementMassMatrices<TElement>(Neg.derived(), wg.derived(), rhog.derived(), Meg);
    return Meg;
}

/**
 * @brief Compute mass matrix-matrix multiply \f$ Y += \mathbf{M} X \f$ using precomputed element
 * mass matrices
 *
 * @tparam TElement Element type
 * @tparam Dims Number of spatial dimensions
 * @tparam TDerivedE Type of the element matrix
 * @tparam TDerivedeg Type of the element indices at quadrature points
 * @tparam TDerivedMe Type of the precomputed element mass matrices
 * @tparam TDerivedIn Type of the input matrix
 * @tparam TDerivedOut Type of the output matrix
 * @param E `|# nodes per element| x |# elements|` matrix of mesh elements
 * @param nNodes Number of mesh nodes
 * @param eg `|# quad.pts.| x 1` vector of element indices at quadrature points
 * @param Me `|# nodes per element| x |# nodes per element * # quad.pts.|` precomputed element mass
 * matrices
 * @param dims Dimensionality of the image of the FEM function space
 * @param X `|# nodes * dims| x |# cols|` input matrix
 * @param Y `|# nodes * dims| x |# cols|` output matrix
 */
template <
    CElement TElement,
    int Dims,
    class TDerivedE,
    class TDerivedeg,
    class TDerivedMe,
    class TDerivedIn,
    class TDerivedOut>
void GemmMass(
    Eigen::DenseBase<TDerivedE> const& E,
    Eigen::Index nNodes,
    Eigen::DenseBase<TDerivedeg> const& eg,
    Eigen::MatrixBase<TDerivedMe> const& Me,
    int dims,
    Eigen::MatrixBase<TDerivedIn> const& X,
    Eigen::DenseBase<TDerivedOut>& Y);

/**
 * @brief Compute mass matrix-matrix multiply \f$ Y += \mathbf{M} X \f$ using mesh and precomputed
 * element mass matrices
 * @tparam TMesh Type of the mesh
 * @tparam TDerivedeg Type of the element indices at quadrature points
 * @tparam TDerivedMe Type of the precomputed element mass matrices
 * @tparam TDerivedIn Type of the input matrix
 * @tparam TDerivedOut Type of the output matrix
 * @param mesh The finite element mesh
 * @param eg `|# quad.pts.| x 1` vector of element indices at quadrature points
 * @param Me `|# nodes per element| x |# nodes per element * # quad.pts.|` precomputed element mass
 * matrices
 * @param dims Dimensionality of the image of the FEM function space
 * @param X `|# nodes * dims| x |# cols|` input matrix
 * @param Y `|# nodes * dims| x |# cols|` output matrix
 */
template <CMesh TMesh, class TDerivedeg, class TDerivedMe, class TDerivedIn, class TDerivedOut>
inline void GemmMass(
    TMesh const& mesh,
    Eigen::DenseBase<TDerivedeg> const& eg,
    Eigen::MatrixBase<TDerivedMe> const& Me,
    int dims,
    Eigen::MatrixBase<TDerivedIn> const& X,
    Eigen::DenseBase<TDerivedOut>& Y)
{
    GemmMass<typename TMesh::ElementType, TMesh::kDims>(
        mesh.E,
        static_cast<typename TMesh::IndexType>(mesh.X.cols()),
        eg.derived(),
        Me.derived(),
        dims,
        X.derived(),
        Y.derived());
}

/**
 * @brief Construct the mass matrix operator's sparse matrix representation
 * @tparam TElement Element type
 * @tparam Dims Number of spatial dimensions
 * @tparam Options Storage options for the matrix (default: Eigen::ColMajor)
 * @tparam TDerivedE Type of the element matrix
 * @tparam TDerivedeg Type of the element indices at quadrature points
 * @tparam TDerivedwg Type of the quadrature weights
 * @tparam TDerivedrhog Type of the density at quadrature points
 * @tparam TDerivedNeg Type of the shape functions at quadrature points
 * @param E `|# nodes per element| x |# elements|` matrix of mesh elements
 * @param nNodes Number of mesh nodes
 * @param eg `|# quad.pts.| x 1` vector of element indices at quadrature points
 * @param wg `|# quad.pts.| x 1` vector of quadrature weights (including Jacobian determinants)
 * @param rhog `|# quad.pts.| x 1` vector of density at quadrature points
 * @param Neg `|# nodes per element| x |# quad.pts.|` shape functions at quadrature points
 * @param dims Dimensionality of the image of the FEM function space
 * @return Sparse matrix representation of the mass matrix operator
 */
template <
    CElement TElement,
    int Dims,
    Eigen::StorageOptions Options,
    class TDerivedE,
    class TDerivedeg,
    class TDerivedwg,
    class TDerivedrhog,
    class TDerivedNeg>
auto MassMatrix(
    Eigen::DenseBase<TDerivedE> const& E,
    Eigen::Index nNodes,
    Eigen::DenseBase<TDerivedeg> const& eg,
    Eigen::DenseBase<TDerivedwg> const& wg,
    Eigen::DenseBase<TDerivedrhog> const& rhog,
    Eigen::MatrixBase<TDerivedNeg> const& Neg,
    int dims = 1)
    -> Eigen::SparseMatrix<typename TDerivedNeg::Scalar, Options, typename TDerivedE::Scalar>;

/**
 * @brief Construct the mass matrix operator's sparse matrix representation from precomputed element
 * mass matrices
 * @tparam TElement Element type
 * @tparam Dims Number of spatial dimensions
 * @tparam Options Storage options for the matrix (default: Eigen::ColMajor)
 * @tparam TDerivedE Type of the element matrix
 * @tparam TDerivedeg Type of the element indices at quadrature points
 * @tparam TDerivedMe Type of the precomputed element mass matrices
 * @param E `|# nodes per element| x |# elements|` matrix of mesh elements
 * @param nNodes Number of mesh nodes
 * @param eg `|# quad.pts.| x 1` vector of element indices at quadrature points
 * @param Meg `|# nodes per element| x |# nodes per element * # quad.pts.|` precomputed element mass
 * matrices
 * @param dims Dimensionality of the image of the FEM function space
 * @return Sparse matrix representation of the mass matrix operator
 */
template <
    CElement TElement,
    int Dims,
    Eigen::StorageOptions Options,
    class TDerivedE,
    class TDerivedeg,
    class TDerivedMe>
auto MassMatrix(
    Eigen::DenseBase<TDerivedE> const& E,
    Eigen::Index nNodes,
    Eigen::DenseBase<TDerivedeg> const& eg,
    Eigen::MatrixBase<TDerivedMe> const& Meg,
    int dims = 1)
    -> Eigen::SparseMatrix<typename TDerivedMe::Scalar, Options, typename TDerivedE::Scalar>;

/**
 * @brief Construct the mass matrix operator's sparse matrix representation using mesh
 * @tparam Options Storage options for the matrix (default: Eigen::ColMajor)
 * @tparam TMesh Type of the mesh
 * @tparam TDerivedeg Type of the element indices at quadrature points
 * @tparam TDerivedwg Type of the quadrature weights
 * @tparam TDerivedrhog Type of the density at quadrature points
 * @tparam TDerivedNeg Type of the shape functions at quadrature points
 * @param mesh The finite element mesh
 * @param eg `|# quad.pts.| x 1` vector of element indices at quadrature points
 * @param wg `|# quad.pts.| x 1` vector of quadrature weights (including Jacobian determinants)
 * @param rhog `|# quad.pts.| x 1` vector of density at quadrature points
 * @param Neg `|# nodes per element| x |# quad.pts.|` shape functions at quadrature points
 * @param dims Dimensionality of the image of the FEM function space
 * @return Sparse matrix representation of the mass matrix operator
 */
template <
    Eigen::StorageOptions Options,
    CMesh TMesh,
    class TDerivedeg,
    class TDerivedwg,
    class TDerivedrhog,
    class TDerivedNeg>
auto MassMatrix(
    TMesh const& mesh,
    Eigen::DenseBase<TDerivedeg> const& eg,
    Eigen::DenseBase<TDerivedwg> const& wg,
    Eigen::DenseBase<TDerivedrhog> const& rhog,
    Eigen::MatrixBase<TDerivedNeg> const& Neg,
    int dims = 1)
    -> Eigen::SparseMatrix<typename TDerivedNeg::Scalar, Options, typename TMesh::IndexType>
{
    return MassMatrix<typename TMesh::ElementType, TMesh::kDims, Options>(
        mesh.E,
        static_cast<typename TMesh::IndexType>(mesh.X.cols()),
        eg.derived(),
        wg.derived(),
        rhog.derived(),
        Neg.derived(),
        dims);
}

/**
 * @brief Construct the mass matrix operator's sparse matrix representation using mesh and
 * precomputed element mass matrices
 * @tparam Options Storage options for the matrix (default: Eigen::ColMajor)
 * @tparam TMesh Type of the mesh
 * @tparam TDerivedeg Type of the element indices at quadrature points
 * @tparam TDerivedMe Type of the precomputed element mass matrices
 * @param mesh The finite element mesh
 * @param eg `|# quad.pts.| x 1` vector of element indices at quadrature points
 * @param Meg `|# nodes per element| x |# nodes per element * # quad.pts.|` precomputed element mass
 * matrices
 * @param dims Dimensionality of the image of the FEM function space
 * @return Sparse matrix representation of the mass matrix operator
 */
template <Eigen::StorageOptions Options, CMesh TMesh, class TDerivedeg, class TDerivedMe>
auto MassMatrix(
    TMesh const& mesh,
    Eigen::DenseBase<TDerivedeg> const& eg,
    Eigen::MatrixBase<TDerivedMe> const& Meg,
    int dims = 1)
    -> Eigen::SparseMatrix<typename TDerivedMe::Scalar, Options, typename TMesh::IndexType>
{
    return MassMatrix<typename TMesh::ElementType, TMesh::kDims, Options>(
        mesh.E,
        static_cast<typename TMesh::IndexType>(mesh.X.cols()),
        eg.derived(),
        Meg.derived(),
        dims);
}

/**
 * @brief Compute lumped mass matrix's diagonal vector into existing output vector
 * @tparam TElement Element type
 * @tparam Dims Number of spatial dimensions
 * @tparam TDerivedE Type of the element matrix
 * @tparam TDerivedeg Type of the element indices at quadrature points
 * @tparam TDerivedwg Type of the quadrature weights
 * @tparam TDerivedrhog Type of the density at quadrature points
 * @tparam TDerivedNeg Type of the shape functions at quadrature points
 * @tparam TDerivedOut Type of the output vector
 * @param E `|# nodes per element| x |# elements|` matrix of mesh elements
 * @param nNodes Number of mesh nodes
 * @param eg `|# quad.pts.| x 1` vector of element indices at quadrature points
 * @param wg `|# quad.pts.| x 1` vector of quadrature weights (including Jacobian determinants)
 * @param rhog `|# quad.pts.| x 1` vector of density at quadrature points
 * @param Neg `|# nodes per element| x |# quad.pts.|` shape functions at quadrature points
 * @param dims Dimensionality of the image of the FEM function space
 * @param m Output vector of lumped masses `|# nodes * dims| x 1`
 */
template <
    CElement TElement,
    int Dims,
    class TDerivedE,
    class TDerivedeg,
    class TDerivedwg,
    class TDerivedrhog,
    class TDerivedNeg,
    class TDerivedOut>
void ToLumpedMassMatrix(
    Eigen::DenseBase<TDerivedE> const& E,
    Eigen::Index nNodes,
    Eigen::DenseBase<TDerivedeg> const& eg,
    Eigen::DenseBase<TDerivedwg> const& wg,
    Eigen::DenseBase<TDerivedrhog> const& rhog,
    Eigen::MatrixBase<TDerivedNeg> const& Neg,
    int dims,
    Eigen::PlainObjectBase<TDerivedOut>& m);

/**
 * @brief Compute lumped mass vector from precomputed element mass matrices into existing output
 * vector
 * @tparam TElement Element type
 * @tparam Dims Number of spatial dimensions
 * @tparam TDerivedE Type of the element matrix
 * @tparam TDerivedeg Type of the element indices at quadrature points
 * @tparam TDerivedMe Type of the precomputed element mass matrices
 * @tparam TDerivedOut Type of the output vector
 * @param E `|# nodes per element| x |# elements|` matrix of mesh elements
 * @param nNodes Number of mesh nodes
 * @param eg `|# quad.pts.| x 1` vector of element indices at quadrature points
 * @param Meg `|# nodes per element| x |# nodes per element * # quad.pts.|` precomputed element mass
 * matrices
 * @param dims Dimensionality of the image of the FEM function space
 * @param m Output vector of lumped masses `|# nodes * dims| x 1`
 */
template <
    CElement TElement,
    int Dims,
    class TDerivedE,
    class TDerivedeg,
    class TDerivedMe,
    class TDerivedOut>
void ToLumpedMassMatrix(
    Eigen::DenseBase<TDerivedE> const& E,
    Eigen::Index nNodes,
    Eigen::DenseBase<TDerivedeg> const& eg,
    Eigen::MatrixBase<TDerivedMe> const& Meg,
    int dims,
    Eigen::PlainObjectBase<TDerivedOut>& m);

/**
 * @brief Compute lumped mass vector using mesh and precomputed element mass matrices into
 * existing output vector
 * @tparam TMesh Type of the mesh
 * @tparam TDerivedeg Type of the element indices at quadrature points
 * @tparam TDerivedwg Type of the quadrature weights
 * @tparam TDerivedrhog Type of the density at quadrature points
 * @tparam TDerivedNeg Type of the shape functions at quadrature points
 * @tparam TDerivedOut Type of the output vector
 * @param mesh The finite element mesh
 * @param eg `|# quad.pts.| x 1` vector of element indices at quadrature points
 * @param wg `|# quad.pts.| x 1` vector of quadrature weights (including Jacobian determinants)
 * @param rhog `|# quad.pts.| x 1` vector of density at quadrature points
 * @param Neg `|# nodes per element| x |# quad.pts.|` shape functions at quadrature points
 * @param dims Dimensionality of the image of the FEM function space
 * @param m Output vector of lumped masses `|# nodes * dims| x 1`
 */
template <
    CMesh TMesh,
    class TDerivedeg,
    class TDerivedwg,
    class TDerivedrhog,
    class TDerivedNeg,
    class TDerivedOut>
inline void ToLumpedMassMatrix(
    TMesh const& mesh,
    Eigen::DenseBase<TDerivedeg> const& eg,
    Eigen::DenseBase<TDerivedwg> const& wg,
    Eigen::DenseBase<TDerivedrhog> const& rhog,
    Eigen::MatrixBase<TDerivedNeg> const& Neg,
    int dims,
    Eigen::PlainObjectBase<TDerivedOut>& m)
{
    ToLumpedMassMatrix<typename TMesh::ElementType, TMesh::kDims>(
        mesh.E,
        static_cast<typename TMesh::IndexType>(mesh.X.cols()),
        eg.derived(),
        wg.derived(),
        rhog.derived(),
        Neg.derived(),
        dims,
        m.derived());
}

/**
 * @brief Compute lumped mass vector using mesh and precomputed element mass matrices into
 * existing output vector
 * @tparam TMesh Type of the mesh
 * @tparam TDerivedeg Type of the element indices at quadrature points
 * @tparam TDerivedMe Type of the precomputed element mass matrices
 * @tparam TDerivedOut Type of the output vector
 * @param mesh The finite element mesh
 * @param eg `|# quad.pts.| x 1` vector of element indices at quadrature points
 * @param Meg `|# nodes per element| x |# nodes per element * # quad.pts.|` precomputed element mass
 * matrices
 * @param dims Dimensionality of the image of the FEM function space
 * @param m Output vector of lumped masses `|# nodes * dims| x 1`
 */
template <CMesh TMesh, class TDerivedeg, class TDerivedMe, class TDerivedOut>
inline void ToLumpedMassMatrix(
    TMesh const& mesh,
    Eigen::DenseBase<TDerivedeg> const& eg,
    Eigen::MatrixBase<TDerivedMe> const& Meg,
    int dims,
    Eigen::PlainObjectBase<TDerivedOut>& m)
{
    ToLumpedMassMatrix<typename TMesh::ElementType, TMesh::kDims>(
        mesh.E,
        static_cast<typename TMesh::IndexType>(mesh.X.cols()),
        eg.derived(),
        Meg.derived(),
        dims,
        m.derived());
}

/**
 * @brief Compute lumped mass matrix's diagonal vector (allocates output vector)
 * @tparam TElement Element type
 * @tparam Dims Number of spatial dimensions
 * @tparam TDerivedE Type of the element matrix
 * @tparam TDerivedeg Type of the element indices at quadrature points
 * @tparam TDerivedwg Type of the quadrature weights
 * @tparam TDerivedrhog Type of the density at quadrature points
 * @tparam TDerivedNeg Type of the shape functions at quadrature points
 * @param E `|# nodes per element| x |# elements|` matrix of mesh elements
 * @param nNodes Number of mesh nodes
 * @param eg `|# quad.pts.| x 1` vector of element indices at quadrature points
 * @param wg `|# quad.pts.| x 1` vector of quadrature weights (including Jacobian determinants)
 * @param rhog `|# quad.pts.| x 1` vector of density at quadrature points
 * @param Neg `|# nodes per element| x |# quad.pts.|` shape functions at quadrature points
 * @param dims Dimensionality of the image of the FEM function space
 * @return Vector of lumped masses
 */
template <
    CElement TElement,
    int Dims,
    class TDerivedE,
    class TDerivedeg,
    class TDerivedwg,
    class TDerivedrhog,
    class TDerivedNeg>
auto LumpedMassMatrix(
    Eigen::DenseBase<TDerivedE> const& E,
    Eigen::Index nNodes,
    Eigen::DenseBase<TDerivedeg> const& eg,
    Eigen::DenseBase<TDerivedwg> const& wg,
    Eigen::DenseBase<TDerivedrhog> const& rhog,
    Eigen::MatrixBase<TDerivedNeg> const& Neg,
    int dims = 1) -> Eigen::Vector<typename TDerivedNeg::Scalar, Eigen::Dynamic>
{
    using ScalarType        = typename TDerivedNeg::Scalar;
    auto const numberOfDofs = dims * nNodes;
    Eigen::Vector<ScalarType, Eigen::Dynamic> m(numberOfDofs);
    ToLumpedMassMatrix<TElement, Dims>(E, nNodes, eg, wg, rhog, Neg, dims, m);
    return m;
}

/**
 * @brief Compute lumped mass vector from precomputed element mass matrices (allocates output
 * vector)
 * @tparam TElement Element type
 * @tparam Dims Number of spatial dimensions
 * @tparam TDerivedE Type of the element matrix
 * @tparam TDerivedeg Type of the element indices at quadrature points
 * @tparam TDerivedMe Type of the precomputed element mass matrices
 * @param E `|# nodes per element| x |# elements|` matrix of mesh elements
 * @param nNodes Number of mesh nodes
 * @param eg `|# quad.pts.| x 1` vector of element indices at quadrature points
 * @param Meg `|# nodes per element| x |# nodes per element * # quad.pts.|` precomputed element mass
 * matrices
 * @param dims Dimensionality of the image of the FEM function space
 * @return Vector of lumped masses
 */
template <CElement TElement, int Dims, class TDerivedE, class TDerivedeg, class TDerivedMe>
auto LumpedMassMatrix(
    Eigen::DenseBase<TDerivedE> const& E,
    Eigen::Index nNodes,
    Eigen::DenseBase<TDerivedeg> const& eg,
    Eigen::MatrixBase<TDerivedMe> const& Meg,
    int dims = 1) -> Eigen::Vector<typename TDerivedMe::Scalar, Eigen::Dynamic>
{
    using ScalarType        = typename TDerivedMe::Scalar;
    auto const numberOfDofs = dims * nNodes;
    Eigen::Vector<ScalarType, Eigen::Dynamic> m(numberOfDofs);
    ToLumpedMassMatrix<TElement, Dims>(E, nNodes, eg, Meg, dims, m);
    return m;
}

/**
 * @brief Compute lumped mass vector using mesh (allocates output vector)
 */
template <CMesh TMesh, class TDerivedeg, class TDerivedwg, class TDerivedrhog, class TDerivedNeg>
auto LumpedMassMatrix(
    TMesh const& mesh,
    Eigen::DenseBase<TDerivedeg> const& eg,
    Eigen::DenseBase<TDerivedwg> const& wg,
    Eigen::DenseBase<TDerivedrhog> const& rhog,
    Eigen::MatrixBase<TDerivedNeg> const& Neg,
    int dims = 1)
{
    return LumpedMassMatrix<typename TMesh::ElementType, TMesh::kDims>(
        mesh.E,
        static_cast<typename TMesh::IndexType>(mesh.X.cols()),
        eg.derived(),
        wg.derived(),
        rhog.derived(),
        Neg.derived(),
        dims);
}

/**
 * @brief Compute lumped mass vector using mesh and precomputed element mass matrices (allocates
 * output vector)
 */
template <CMesh TMesh, class TDerivedeg, class TDerivedMe>
auto LumpedMassMatrix(
    TMesh const& mesh,
    Eigen::DenseBase<TDerivedeg> const& eg,
    Eigen::MatrixBase<TDerivedMe> const& Meg,
    int dims = 1)
{
    return LumpedMassMatrix<typename TMesh::ElementType, TMesh::kDims>(
        mesh.E,
        static_cast<typename TMesh::IndexType>(mesh.X.cols()),
        eg.derived(),
        Meg.derived(),
        dims);
}

// Implementation of GemmMass
template <
    CElement TElement,
    int Dims,
    class TDerivedE,
    class TDerivedeg,
    class TDerivedMe,
    class TDerivedIn,
    class TDerivedOut>
inline void GemmMass(
    Eigen::DenseBase<TDerivedE> const& E,
    Eigen::Index nNodes,
    Eigen::DenseBase<TDerivedeg> const& eg,
    Eigen::MatrixBase<TDerivedMe> const& Meg,
    int dims,
    Eigen::MatrixBase<TDerivedIn> const& X,
    Eigen::DenseBase<TDerivedOut>& Y)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.fem.GemmMass");

    // Check inputs
    auto const numberOfDofs = dims * nNodes;
    bool const bDimensionsMatch =
        (X.cols() == Y.cols()) and (X.rows() == numberOfDofs) and (Y.rows() == numberOfDofs);
    if (not bDimensionsMatch)
    {
        std::string const what = fmt::format(
            "Expected input and output to have {} rows and same number of columns, but got "
            "dimensions X,Y=({} x {}), ({} x {})",
            numberOfDofs,
            X.rows(),
            X.cols(),
            Y.rows(),
            Y.cols());
        throw std::invalid_argument(what);
    }

    if (dims < 1)
    {
        std::string const what = fmt::format("Expected dims >= 1, got {} instead", dims);
        throw std::invalid_argument(what);
    }

    auto constexpr kNodesPerElement = TElement::kNodes;
    auto const nQuadPts             = eg.size();

    // Apply precomputed element mass matrices
    for (auto c = 0; c < X.cols(); ++c)
    {
        for (auto g = 0; g < nQuadPts; ++g)
        {
            auto const e     = eg(g);
            auto const nodes = E.col(e);

            // Get precomputed element mass matrix for this quadrature point
            auto const Me =
                Meg.template block<kNodesPerElement, kNodesPerElement>(0, g * kNodesPerElement);

            // Apply to each dimension
            auto ye       = Y.col(c).reshaped(dims, nNodes)(Eigen::placeholders::all, nodes);
            auto const xe = X.col(c).reshaped(dims, nNodes)(Eigen::placeholders::all, nodes);
            ye += xe * Me.transpose(); // Mass matrix is symmetric, so transpose doesn't matter
        }
    }
}

// Implementation of MassMatrix
template <
    CElement TElement,
    int Dims,
    Eigen::StorageOptions Options,
    class TDerivedE,
    class TDerivedeg,
    class TDerivedwg,
    class TDerivedrhog,
    class TDerivedNeg>
auto MassMatrix(
    Eigen::DenseBase<TDerivedE> const& E,
    Eigen::Index nNodes,
    Eigen::DenseBase<TDerivedeg> const& eg,
    Eigen::DenseBase<TDerivedwg> const& wg,
    Eigen::DenseBase<TDerivedrhog> const& rhog,
    Eigen::MatrixBase<TDerivedNeg> const& Neg,
    int dims)
    -> Eigen::SparseMatrix<typename TDerivedNeg::Scalar, Options, typename TDerivedE::Scalar>
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.fem.MassMatrix");
    using ScalarType       = typename TDerivedNeg::Scalar;
    using IndexType        = typename TDerivedE::Scalar;
    using SparseMatrixType = Eigen::SparseMatrix<ScalarType, Options, IndexType>;
    using Triplet          = Eigen::Triplet<ScalarType, IndexType>;

    auto constexpr kNodesPerElement = TElement::kNodes;
    auto const nQuadPts             = eg.size();

    std::vector<Triplet> triplets{};
    triplets.reserve(
        static_cast<std::size_t>(kNodesPerElement * kNodesPerElement * nQuadPts * dims));

    auto const numberOfDofs = dims * nNodes;
    SparseMatrixType M(numberOfDofs, numberOfDofs);

    for (auto g = 0; g < nQuadPts; ++g)
    {
        auto const e     = eg(g);
        auto const nodes = E.col(e);
        auto const w     = wg(g); // Includes Jacobian determinant
        auto const rho   = rhog(g);

        // Get shape functions at this quadrature point
        auto const N = Neg.col(g);

        // Compute element mass matrix: w * rho * N * N^T
        auto const Me = w * rho * N * N.transpose();

        // Add contributions for each dimension
        for (auto i = 0; i < kNodesPerElement; ++i)
        {
            for (auto j = 0; j < kNodesPerElement; ++j)
            {
                for (auto d = 0; d < dims; ++d)
                {
                    auto const ni = static_cast<IndexType>(dims * nodes(i) + d);
                    auto const nj = static_cast<IndexType>(dims * nodes(j) + d);
                    triplets.emplace_back(ni, nj, Me(i, j));
                }
            }
        }
    }

    M.setFromTriplets(triplets.begin(), triplets.end());
    return M;
}

template <
    CElement TElement,
    int Dims,
    Eigen::StorageOptions Options,
    class TDerivedE,
    class TDerivedeg,
    class TDerivedMe>
auto MassMatrix(
    Eigen::DenseBase<TDerivedE> const& E,
    Eigen::Index nNodes,
    Eigen::DenseBase<TDerivedeg> const& eg,
    Eigen::MatrixBase<TDerivedMe> const& Meg,
    int dims)
    -> Eigen::SparseMatrix<typename TDerivedMe::Scalar, Options, typename TDerivedE::Scalar>
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.fem.MassMatrix");
    using ScalarType       = typename TDerivedMe::Scalar;
    using IndexType        = typename TDerivedE::Scalar;
    using SparseMatrixType = Eigen::SparseMatrix<ScalarType, Options, IndexType>;
    using Triplet          = Eigen::Triplet<ScalarType, IndexType>;

    auto constexpr kNodesPerElement = TElement::kNodes;
    auto const nQuadPts             = eg.size();

    std::vector<Triplet> triplets{};
    triplets.reserve(
        static_cast<std::size_t>(kNodesPerElement * kNodesPerElement * nQuadPts * dims));

    auto const numberOfDofs = dims * nNodes;
    SparseMatrixType M(numberOfDofs, numberOfDofs);

    for (auto g = 0; g < nQuadPts; ++g)
    {
        auto const e     = eg(g);
        auto const nodes = E.col(e);

        // Get precomputed element mass matrix for this quadrature point
        auto const Me =
            Meg.template block<kNodesPerElement, kNodesPerElement>(0, g * kNodesPerElement);

        // Add contributions for each dimension
        for (auto i = 0; i < kNodesPerElement; ++i)
        {
            for (auto j = 0; j < kNodesPerElement; ++j)
            {
                for (auto d = 0; d < dims; ++d)
                {
                    auto const ni = static_cast<IndexType>(dims * nodes(i) + d);
                    auto const nj = static_cast<IndexType>(dims * nodes(j) + d);
                    triplets.emplace_back(ni, nj, Me(i, j));
                }
            }
        }
    }

    M.setFromTriplets(triplets.begin(), triplets.end());
    return M;
}

// Implementation of ToLumpedMassMatrix
template <
    CElement TElement,
    int Dims,
    class TDerivedE,
    class TDerivedeg,
    class TDerivedwg,
    class TDerivedrhog,
    class TDerivedNeg,
    class TDerivedOut>
inline void ToLumpedMassMatrix(
    Eigen::DenseBase<TDerivedE> const& E,
    Eigen::Index nNodes,
    Eigen::DenseBase<TDerivedeg> const& eg,
    Eigen::DenseBase<TDerivedwg> const& wg,
    Eigen::DenseBase<TDerivedrhog> const& rhog,
    Eigen::MatrixBase<TDerivedNeg> const& Neg,
    int dims,
    Eigen::PlainObjectBase<TDerivedOut>& m)
{
    auto const numberOfDofs = dims * nNodes;
    m.resize(numberOfDofs, 1);
    m.setZero();
    auto constexpr kNodesPerElement = TElement::kNodes;
    auto const nQuadPts             = eg.size();
    for (auto g = 0; g < nQuadPts; ++g)
    {
        auto const e     = eg(g);
        auto const nodes = E.col(e);
        auto const w     = wg(g); // Includes Jacobian determinant
        auto const rho   = rhog(g);

        // Get shape functions at this quadrature point
        auto const N = Neg.col(g);

        // Compute element mass matrix: w * rho * N * N^T
        auto const Me = (w * rho * N * N.transpose()).eval();

        // Add lumped contributions (sum rows)
        for (auto i = 0; i < kNodesPerElement; ++i)
        {
            for (auto j = 0; j < kNodesPerElement; ++j)
            {
                for (auto d = 0; d < dims; ++d)
                {
                    auto const ni = dims * nodes(i) + d;
                    m(ni) += Me(i, j);
                }
            }
        }
    }
}

template <
    CElement TElement,
    int Dims,
    class TDerivedE,
    class TDerivedeg,
    class TDerivedMe,
    class TDerivedOut>
inline void ToLumpedMassMatrix(
    Eigen::DenseBase<TDerivedE> const& E,
    Eigen::Index nNodes,
    Eigen::DenseBase<TDerivedeg> const& eg,
    Eigen::MatrixBase<TDerivedMe> const& Meg,
    int dims,
    Eigen::PlainObjectBase<TDerivedOut>& m)
{
    auto const numberOfDofs = dims * nNodes;
    m.resize(numberOfDofs, 1);
    m.setZero();
    auto constexpr kNodesPerElement = TElement::kNodes;
    auto const nQuadPts             = eg.size();
    for (auto g = 0; g < nQuadPts; ++g)
    {
        auto const e     = eg(g);
        auto const nodes = E.col(e);

        // Get precomputed element mass matrix for this quadrature point
        auto const Me =
            Meg.template block<kNodesPerElement, kNodesPerElement>(0, g * kNodesPerElement);

        // Add lumped contributions (sum rows)
        for (auto i = 0; i < kNodesPerElement; ++i)
        {
            for (auto j = 0; j < kNodesPerElement; ++j)
            {
                for (auto d = 0; d < dims; ++d)
                {
                    auto const ni = dims * nodes(i) + d;
                    m(ni) += Me(i, j);
                }
            }
        }
    }
}

} // namespace pbat::fem

#endif // PBAT_FEM_MASS_H