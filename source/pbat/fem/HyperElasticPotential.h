/**
 * @file HyperElasticPotential.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Hyper elastic potential energy
 * @date 2025-02-11
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_FEM_HYPERELASTICPOTENTIAL_H
#define PBAT_FEM_HYPERELASTICPOTENTIAL_H

#include "Concepts.h"
#include "DeformationGradient.h"
#include "pbat/Aliases.h"
#include "pbat/common/Eigen.h"
#include "pbat/math/linalg/SparsityPattern.h"
#include "pbat/math/linalg/mini/Eigen.h"
#include "pbat/math/linalg/mini/Product.h"
#include "pbat/physics/HyperElasticity.h"
#include "pbat/profiling/Profiling.h"

#include <Eigen/Eigenvalues>
#include <exception>
#include <fmt/core.h>
#include <span>
#include <string>
#include <tbb/parallel_for.h>

namespace pbat {
namespace fem {

/**
 * @brief Bit-flag enum for SPD projection type
 */
enum class EHyperElasticSpdCorrection : std::uint32_t {
    None,       // No projection
    Projection, // Project element hessian to nearest (in the 2-norm) SPD matrix
    Absolute,   // Flip negative eigenvalue signs
};

/**
 * @brief Bit-flag enum for element elasticity computation flags
 */
enum EElementElasticityComputationFlags : int {
    Potential = 1 << 0, // Compute element elastic potential at quadrature points
    Gradient  = 1 << 1, // Compute element gradient vectors at quadrature points
    Hessian   = 1 << 2, // Compute element hessian matrices at quadrature points
};

/**
 * @brief Compute element elasticity and its derivatives at the given shape
 * @tparam TElement Element type
 * @tparam Dims Number of spatial dimensions
 * @tparam THyperElasticEnergy Hyper elastic energy type
 * @tparam TDerivedE Type of the element matrix
 * @tparam TDerivedeg Type of the element indices at quadrature points
 * @tparam TDerivedwg Type of the quadrature weights
 * @tparam TDerivedGNeg Type of the shape function gradients at quadrature points
 * @tparam TDerivedlameg Type of the Lame coefficients at quadrature points
 * @tparam TDerivedx Type of the deformed nodal positions
 * @tparam TDerivedUg Type of the elastic potentials at quadrature points
 * @tparam TDerivedGg Type of the element gradient vectors
 * @tparam TDerivedHg Type of the element hessian matrices
 * @param E `|# nodes per element| x |# elements|` matrix of mesh elements
 * @param nNodes Number of mesh nodes
 * @param eg `|# quad.pts.| x 1` vector of element indices at quadrature points
 * @param wg `|# quad.pts.| x 1` vector of quadrature weights
 * @param GNeg `|# nodes per element| x |# dims * # quad.pts.|` shape function gradients
 * @param mug `|# quad.pts.| x 1` first Lame coefficients at quadrature points
 * @param lambdag `|# quad.pts.| x 1` second Lame coefficients at quadrature points
 * @param x `|# dims * # nodes| x 1` deformed nodal positions
 * @param Ug Optional `|# quad.pts.| x 1` elastic potentials at quadrature points
 * @param Gg Optional `|# dims * # elem nodes| x |# quad.pts.|` element elastic potential
 * gradients at quadrature points.
 * @param Hg Optional `|# dims * # elem nodes| x |# dims * # elem nodes * # quad.pts.|`
 * element elastic hessian matrices at quadrature points.
 * @param eFlags Flags for the computation (see EElementElasticityComputationFlags)
 * @param eSpdCorrection SPD correction mode (see EHyperElasticSpdCorrection)
 */
template <
    CElement TElement,
    int Dims,
    physics::CHyperElasticEnergy THyperElasticEnergy,
    class TDerivedE,
    class TDerivedeg,
    class TDerivedwg,
    class TDerivedGNeg,
    class TDerivedmug,
    class TDerivedlambdag,
    class TDerivedx,
    class TDerivedUg,
    class TDerivedGg,
    class TDerivedHg>
void ToElementElasticity(
    Eigen::DenseBase<TDerivedE> const& E,
    typename TDerivedE::Scalar nNodes,
    Eigen::DenseBase<TDerivedeg> const& eg,
    Eigen::DenseBase<TDerivedwg> const& wg,
    Eigen::MatrixBase<TDerivedGNeg> const& GNeg,
    Eigen::DenseBase<TDerivedmug> const& mug,
    Eigen::DenseBase<TDerivedlambdag> const& lambdag,
    Eigen::MatrixBase<TDerivedx> const& x,
    Eigen::PlainObjectBase<TDerivedUg>& Ug,
    Eigen::PlainObjectBase<TDerivedGg>& Gg,
    Eigen::PlainObjectBase<TDerivedHg>& Hg,
    int eFlags                                = EElementElasticityComputationFlags::Potential,
    EHyperElasticSpdCorrection eSpdCorrection = EHyperElasticSpdCorrection::None);

/**
 * @brief Compute element elasticity using mesh
 * @tparam THyperElasticEnergy Hyper elastic energy type
 * @tparam TMesh Type of the mesh
 * @tparam TDerivedeg Type of the element indices at quadrature points
 * @tparam TDerivedwg Type of the quadrature weights
 * @tparam TDerivedGNeg Type of the shape function gradients at quadrature points
 * @tparam TDerivedmug Type of the first Lame coefficients at quadrature points
 * @tparam TDerivedlambdag Type of the second Lame coefficients at quadrature points
 * @tparam TDerivedx Type of the deformed nodal positions
 * @tparam TDerivedUg Type of the elastic potentials at quadrature points
 * @tparam TDerivedGg Type of the element gradient vectors at quadrature points
 * @tparam TDerivedHg Type of the element hessian matrices at quadrature points
 * @param mesh Mesh containing element connectivity and node positions
 * @param eg `|# quad.pts.| x 1` vector of element indices at quadrature points
 * @param wg `|# quad.pts.| x 1` vector of quadrature weights
 * @param GNeg `|# nodes per element| x |# dims * # quad.pts.|` shape function gradients
 * @param mug `|# quad.pts.| x 1` first Lame coefficients at quadrature points
 * @param lambdag `|# quad.pts.| x 1` second Lame coefficients at quadrature points
 * @param x `|# dims * # nodes| x 1` deformed nodal positions
 * @param Ug `|# quad.pts.| x 1` elastic potentials at quadrature points
 * @param Gg Optional `|# dims * # elem nodes| x |# quad.pts.|` element elastic potential
 * gradients at quadrature points.
 * @param Hg Optional `|# dims * # elem nodes| x |# dims * # elem nodes * # quad.pts.|`
 * element elastic hessian matrices at quadrature points.
 * @param eFlags Flags for the computation (see EElementElasticityComputationFlags)
 * @param eSpdCorrection SPD correction mode (see EHyperElasticSpdCorrection)
 */
template <
    physics::CHyperElasticEnergy THyperElasticEnergy,
    CMesh TMesh,
    class TDerivedeg,
    class TDerivedwg,
    class TDerivedGNeg,
    class TDerivedmug,
    class TDerivedlambdag,
    class TDerivedx,
    class TDerivedUg,
    class TDerivedGg,
    class TDerivedHg>
inline void ToElementElasticity(
    TMesh const& mesh,
    Eigen::DenseBase<TDerivedeg> const& eg,
    Eigen::DenseBase<TDerivedwg> const& wg,
    Eigen::MatrixBase<TDerivedGNeg> const& GNeg,
    Eigen::DenseBase<TDerivedmug> const& mug,
    Eigen::DenseBase<TDerivedlambdag> const& lambdag,
    Eigen::MatrixBase<TDerivedx> const& x,
    Eigen::PlainObjectBase<TDerivedUg>& Ug,
    Eigen::PlainObjectBase<TDerivedGg>& Gg,
    Eigen::PlainObjectBase<TDerivedHg>& Hg,
    int eFlags                                = EElementElasticityComputationFlags::Potential,
    EHyperElasticSpdCorrection eSpdCorrection = EHyperElasticSpdCorrection::None)
{
    ToElementElasticity<typename TMesh::ElementType, TMesh::kDims, THyperElasticEnergy>(
        mesh.E,
        static_cast<typename TMesh::IndexType>(mesh.X.cols()),
        eg.derived(),
        wg.derived(),
        GNeg.derived(),
        mug.derived(),
        lambdag.derived(),
        x.derived(),
        Ug,
        Gg,
        Hg,
        eFlags,
        eSpdCorrection);
}

/**
 * @brief Apply the hessian matrix as a linear operator \f$ Y += \mathbf{H} X \f$
 * @tparam TElement Element type
 * @tparam Dims Number of spatial dimensions
 * @tparam TDerivedE Type of the element matrix
 * @tparam TDerivedeg Type of the element indices at quadrature points
 * @tparam TDerivedHg Type of the element hessian matrices
 * @tparam TDerivedIn Type of the input matrix
 * @tparam TDerivedOut Type of the output matrix
 * @param E `|# nodes per element| x |# elements|` matrix of mesh elements
 * @param nNodes Number of mesh nodes
 * @param eg `|# quad.pts.| x 1` vector of element indices at quadrature points
 * @param Hg `|# dims * # elem nodes| x |# dims * # elem nodes * # quad.pts.|` element elastic
 * hessian matrices at quadrature points
 * @param X `|# dims * # nodes| x |# cols|` input matrix
 * @param Y `|# dims * # nodes| x |# cols|` output matrix
 */
template <
    CElement TElement,
    int Dims,
    class TDerivedE,
    class TDerivedeg,
    class TDerivedHg,
    class TDerivedIn,
    class TDerivedOut>
void GemmHyperElastic(
    Eigen::DenseBase<TDerivedE> const& E,
    typename TDerivedE::Scalar nNodes,
    Eigen::DenseBase<TDerivedeg> const& eg,
    Eigen::MatrixBase<TDerivedHg> const& Hg,
    Eigen::MatrixBase<TDerivedIn> const& X,
    Eigen::DenseBase<TDerivedOut>& Y);

/**
 * @brief Apply the hessian matrix as a linear operator \f$ Y += \mathbf{H} X \f$ using mesh
 * @tparam TMesh Type of the mesh
 * @tparam TDerivedeg Type of the element indices at quadrature points
 * @tparam TDerivedHg Type of the element hessian matrices
 * @tparam TDerivedIn Type of the input matrix
 * @tparam TDerivedOut Type of the output matrix
 * @param mesh Mesh containing element connectivity and node positions
 * @param eg `|# quad.pts.| x 1` vector of element indices at quadrature points
 * @param Hg `|# dims * # elem nodes| x |# dims * # elem nodes * # quad.pts.|` element elastic
 * hessian matrices at quadrature points
 * @param X `|# dims * # nodes| x |# cols|` input matrix
 * @param Y `|# dims * # nodes| x |# cols|` output matrix
 */
template <CMesh TMesh, class TDerivedeg, class TDerivedHg, class TDerivedIn, class TDerivedOut>
inline void GemmHyperElastic(
    TMesh const& mesh,
    Eigen::DenseBase<TDerivedeg> const& eg,
    Eigen::MatrixBase<TDerivedHg> const& Hg,
    Eigen::MatrixBase<TDerivedIn> const& X,
    Eigen::DenseBase<TDerivedOut>& Y)
{
    GemmHyperElastic<typename TMesh::ElementType, TMesh::kDims>(
        mesh.E,
        static_cast<typename TMesh::IndexType>(mesh.X.cols()),
        eg.derived(),
        Hg.derived(),
        X.derived(),
        Y.derived());
}

/**
 * @brief Construct the hessian matrix's sparsity pattern
 * @tparam TElement Element type
 * @tparam Dims Number of spatial dimensions
 * @tparam Options Storage options for the matrix
 * @tparam TDerivedE Type of the element matrix
 * @tparam TDerivedeg Type of the element indices at quadrature points
 * @param E `|# nodes per element| x |# elements|` matrix of mesh elements
 * @param nNodes Number of mesh nodes
 * @param eg `|# quad.pts.| x 1` vector of element indices at quadrature points
 * @return Elastic hessian's sparsity pattern
 */
template <
    CElement TElement,
    int Dims,
    Eigen::StorageOptions Options,
    class TDerivedE,
    class TDerivedeg>
auto ElasticHessianSparsity(
    Eigen::DenseBase<TDerivedE> const& E,
    typename TDerivedE::Scalar nNodes,
    Eigen::DenseBase<TDerivedeg> const& eg)
    -> math::linalg::SparsityPattern<typename TDerivedE::Scalar, Options>;

/**
 * @brief Construct the hessian matrix's sparsity pattern
 * @param mesh Mesh containing element connectivity and node positions
 * @return Elastic hessian's sparsity pattern
 */
template <Eigen::StorageOptions Options, CMesh TMesh>
auto ElasticHessianSparsity(TMesh const& mesh)
    -> math::linalg::SparsityPattern<typename TMesh::IndexType, Options>
{
    return ElasticHessianSparsity<typename TMesh::ElementType, TMesh::kDims, Options>(
        mesh.E,
        static_cast<typename TMesh::IndexType>(mesh.X.cols()),
        mesh.eg.derived());
}

/**
 * @brief Construct the hessian matrix's sparse representation
 * @tparam TElement Element type
 * @tparam Dims Number of spatial dimensions
 * @tparam Options Storage options for the matrix
 * @tparam TDerivedE Type of the element matrix
 * @tparam TDerivedeg Type of the element indices at quadrature points
 * @tparam TDerivedHg Type of the element hessian matrices
 * @param E `|# nodes per element| x |# elements|` matrix of mesh elements
 * @param nNodes Number of mesh nodes
 * @param eg `|# quad.pts.| x 1` vector of element indices at quadrature points
 * @param Hg `|# dims * # elem nodes| x |# dims * # elem nodes * # quad.pts.|` array of element
 * elastic hessian matrices at quadrature points
 * @return Sparse matrix representation of the hessian
 */
template <
    CElement TElement,
    int Dims,
    Eigen::StorageOptions Options,
    class TDerivedE,
    class TDerivedeg,
    class TDerivedHg>
auto HyperElasticHessian(
    Eigen::DenseBase<TDerivedE> const& E,
    typename TDerivedE::Scalar nNodes,
    Eigen::DenseBase<TDerivedeg> const& eg,
    Eigen::MatrixBase<TDerivedHg> const& Hg)
    -> Eigen::SparseMatrix<typename TDerivedHg::Scalar, Options, typename TDerivedE::Scalar>;

/**
 * @brief Construct the hessian matrix using mesh
 * @tparam TMesh Type of the mesh
 * @tparam TDerivedeg Type of the element indices at quadrature points
 * @tparam TDerivedHg Type of the element hessian matrices
 * @param mesh Mesh containing element connectivity and node positions
 * @param eg `|# quad.pts.| x 1` vector of element indices at quadrature points
 * @param Hg `|# dims * # elem nodes| x |# dims * # elem nodes * # quad.pts.|` element elastic
 * hessian matrices at quadrature points
 * @return Sparse matrix representation of the hessian
 */
template <Eigen::StorageOptions Options, CMesh TMesh, class TDerivedeg, class TDerivedHg>
auto HyperElasticHessian(
    TMesh const& mesh,
    Eigen::DenseBase<TDerivedeg> const& eg,
    Eigen::MatrixBase<TDerivedHg> const& Hg)
    -> Eigen::SparseMatrix<typename TDerivedHg::Scalar, Options, typename TMesh::IndexType>
{
    return HyperElasticHessian<typename TMesh::ElementType, TMesh::kDims, Options>(
        mesh.E,
        static_cast<typename TMesh::IndexType>(mesh.X.cols()),
        eg.derived(),
        Hg.derived());
}

/**
 * @brief Construct the hessian matrix using mesh with sparsity pattern
 * @tparam TElement Element type
 * @tparam Dims Number of spatial dimensions
 * @tparam TDerivedE Type of the element matrix
 * @tparam TDerivedHg Type of the element hessian matrices
 * @tparam Options Storage options for the matrix
 * @tparam TDerivedH Type of the output sparse matrix
 * @param E `|# nodes per element| x |# elements|` matrix of mesh elements
 * @param nNodes Number of mesh nodes
 * @param Hg `|# dims * # elem nodes| x |# dims * # elem nodes * # quad.pts.|` element elastic
 * hessian matrices at quadrature points
 * @param sparsity Sparsity pattern for the hessian matrix
 * @param H Output sparse matrix for the hessian
 */
template <
    CElement TElement,
    int Dims,
    class TDerivedE,
    class TDerivedHg,
    Eigen::StorageOptions Options,
    class TDerivedH>
void ToHyperElasticHessian(
    Eigen::DenseBase<TDerivedE> const& E,
    typename TDerivedE::Scalar nNodes,
    Eigen::DenseBase<TDerivedHg> const& Hg,
    math::linalg::SparsityPattern<typename TDerivedE::Scalar, Options> const& sparsity,
    Eigen::SparseCompressedBase<TDerivedH>& H);

/**
 * @brief Construct the hessian matrix using mesh with sparsity pattern
 * @tparam TMesh Type of the mesh
 * @tparam TDerivedHg Type of the element hessian matrices
 * @tparam Options Storage options for the matrix
 * @tparam TDerivedH Type of the output sparse matrix
 * @param mesh Mesh containing element connectivity and node positions
 * @param Hg `|# dims * # elem nodes| x |# dims * # elem nodes * # quad.pts.|` element elastic
 * hessian matrices at quadrature points
 * @param sparsity Sparsity pattern for the hessian matrix
 * @param H Output sparse matrix for the hessian
 */
template <CMesh TMesh, class TDerivedHg, Eigen::StorageOptions Options, class TDerivedH>
void ToHyperElasticHessian(
    TMesh const& mesh,
    Eigen::DenseBase<TDerivedHg> const& Hg,
    math::linalg::SparsityPattern<typename TMesh::IndexType, Options> const& sparsity,
    Eigen::SparseCompressedBase<typename TDerivedHg::Scalar>& H)
{
    ToHyperElasticHessian<typename TMesh::ElementType, TMesh::kDims, TDerivedHg, Options>(
        mesh.E,
        static_cast<typename TMesh::IndexType>(mesh.X.cols()),
        Hg.derived(),
        sparsity,
        H.derived());
}

template <
    CElement TElement,
    int Dims,
    Eigen::StorageOptions Options,
    class TDerivedE,
    class TDerivedHg>
auto HyperElasticHessian(
    Eigen::DenseBase<TDerivedE> const& E,
    typename TDerivedE::Scalar nNodes,
    Eigen::DenseBase<TDerivedHg> const& Hg,
    math::linalg::SparsityPattern<typename TDerivedE::Scalar, Options> const& sparsity)
    -> Eigen::SparseMatrix<typename TDerivedHg::Scalar, Options, typename TDerivedE::Scalar>;

/**
 * @brief Construct the hessian matrix using mesh with sparsity pattern
 * @tparam TMesh Type of the mesh
 * @tparam TDerivedHg Type of the element hessian matrices
 * @param mesh Mesh containing element connectivity and node positions
 * @param Hg `|# dims * # elem nodes| x |# dims * # elem nodes * # quad.pts.|` element elastic
 * hessian matrices at quadrature points
 * @param sparsity Sparsity pattern for the hessian matrix
 * @return Sparse matrix representation of the hessian
 */
template <Eigen::StorageOptions Options, CMesh TMesh, class TDerivedHg>
auto HyperElasticHessian(
    TMesh const& mesh,
    Eigen::DenseBase<TDerivedHg> const& Hg,
    math::linalg::SparsityPattern<typename TMesh::IndexType, Options> const& sparsity)
    -> Eigen::SparseMatrix<typename TDerivedHg::Scalar, Options, typename TMesh::IndexType>
{
    return HyperElasticHessian<typename TMesh::ElementType, TMesh::kDims, Options>(
        mesh.E,
        static_cast<typename TMesh::IndexType>(mesh.X.cols()),
        Hg.derived(),
        sparsity);
}

/**
 * @brief Compute the gradient vector into existing output
 * @tparam TElement Element type
 * @tparam Dims Number of spatial dimensions
 * @tparam TDerivedE Type of the element matrix
 * @tparam TDerivedeg Type of the element indices at quadrature points
 * @tparam TDerivedGg Type of the element gradient vectors
 * @tparam TDerivedOut Type of the output vector
 * @param E `|# nodes per element| x |# elements|` matrix of mesh elements
 * @param nNodes Number of mesh nodes
 * @param eg `|# quad.pts.| x 1` vector of element indices at quadrature points
 * @param Gg `|# dims * # elem nodes| x |# quad.pts.|` array of element elastic gradient vectors at
 * quadrature points
 * @param G `|# dims * # nodes|` output gradient vector
 */
template <
    CElement TElement,
    int Dims,
    class TDerivedE,
    class TDerivedeg,
    class TDerivedGg,
    class TDerivedG>
void ToHyperElasticGradient(
    Eigen::DenseBase<TDerivedE> const& E,
    typename TDerivedE::Scalar nNodes,
    Eigen::DenseBase<TDerivedeg> const& eg,
    Eigen::MatrixBase<TDerivedGg> const& Gg,
    Eigen::PlainObjectBase<TDerivedG>& G);

/**
 * @brief Compute the gradient vector into existing output
 * @tparam TElement Element type
 * @tparam Dims Number of spatial dimensions
 * @tparam THyperElasticEnergy Hyper elastic energy type
 * @tparam TDerivedE Type of the element matrix
 * @tparam TDerivedeg Type of the element indices at quadrature points
 * @tparam TDerivedwg Type of the quadrature weights
 * @tparam TDerivedGNeg Type of the shape function gradients at quadrature points
 * @tparam TDerivedmug Type of the first Lame coefficients at quadrature points
 * @tparam TDerivedlambdag Type of the second Lame coefficients at quadrature points
 * @tparam TDerivedx Type of the deformed nodal positions
 * @tparam TDerivedGg Type of the element gradient vectors at quadrature points
 * @tparam TDerivedG Type of the output gradient vector
 * @param E `|# nodes per element| x |# elements|` matrix of mesh elements
 * @param nNodes Number of mesh nodes
 * @param eg `|# quad.pts.| x 1` vector of element indices at quadrature points
 * @param wg `|# quad.pts.| x 1` vector of quadrature weights
 * @param GNeg `|# nodes per element| x |# dims * # quad.pts.|` shape function gradients
 * @param mug `|# quad.pts.| x 1` first Lame coefficients at quadrature points
 * @param lambdag `|# quad.pts.| x 1` second Lame coefficients at quadrature points
 * @param x `|# dims * # nodes| x 1` deformed nodal positions
 * @param Gg `|# dims * # elem nodes| x |# quad.pts.|` array of element elastic gradient vectors at
 * quadrature points
 * @param G `|# dims * # nodes|` output gradient vector
 */
template <
    CElement TElement,
    int Dims,
    physics::CHyperElasticEnergy THyperElasticEnergy,
    class TDerivedE,
    class TDerivedeg,
    class TDerivedwg,
    class TDerivedGNeg,
    class TDerivedmug,
    class TDerivedlambdag,
    class TDerivedx,
    class TDerivedGg,
    class TDerivedG>
inline void ToHyperElasticGradient(
    Eigen::DenseBase<TDerivedE> const& E,
    typename TDerivedE::Scalar nNodes,
    Eigen::DenseBase<TDerivedeg> const& eg,
    Eigen::DenseBase<TDerivedwg> const& wg,
    Eigen::MatrixBase<TDerivedGNeg> const& GNeg,
    Eigen::DenseBase<TDerivedmug> const& mug,
    Eigen::DenseBase<TDerivedlambdag> const& lambdag,
    Eigen::MatrixBase<TDerivedx> const& x,
    Eigen::MatrixBase<TDerivedGg> const& Gg,
    Eigen::PlainObjectBase<TDerivedG>& G)
{
    using ScalarType = typename TDerivedx::Scalar;
    Eigen::Matrix<ScalarType, 0, 0> dummyUg, dummyHg;
    ToElementElasticity<TElement, Dims, THyperElasticEnergy>(
        E.derived(),
        nNodes,
        eg.derived(),
        wg.derived(),
        GNeg.derived(),
        mug.derived(),
        lambdag.derived(),
        x.derived(),
        dummyUg,
        Gg.derived(),
        dummyHg,
        EElementElasticityComputationFlags::Gradient,
        EHyperElasticSpdCorrection::None);
    ToHyperElasticGradient<TElement, Dims>(
        E.derived(),
        nNodes,
        eg.derived(),
        Gg.derived(),
        G.derived());
}

/**
 * @brief Compute the gradient vector using mesh (updates existing output)
 * @tparam TMesh Type of the mesh
 * @tparam TDerivedeg Type of the element indices at quadrature points
 * @tparam TDerivedGg Type of the element gradient vectors
 * @tparam TDerivedOut Type of the output vector
 * @param mesh Mesh containing element connectivity and node positions
 * @param eg `|# quad.pts.| x 1` vector of element indices at quadrature points
 * @param Gg `|# dims * # elem nodes| x |# quad.pts.|` array of element elastic gradient vectors at
 * quadrature points
 * @param G `|# dims * # nodes|` output gradient vector
 */
template <CMesh TMesh, class TDerivedeg, class TDerivedGg, class TDerivedOut>
inline void ToHyperElasticGradient(
    TMesh const& mesh,
    Eigen::DenseBase<TDerivedeg> const& eg,
    Eigen::MatrixBase<TDerivedGg> const& Gg,
    Eigen::PlainObjectBase<TDerivedOut>& G)
{
    ToHyperElasticGradient<typename TMesh::ElementType, TMesh::kDims>(
        mesh.E,
        static_cast<typename TMesh::IndexType>(mesh.X.cols()),
        eg.derived(),
        Gg.derived(),
        G.derived());
}

/**
 * @brief Compute the gradient vector into existing output
 * @tparam TMesh Type of the mesh
 * @tparam THyperElasticEnergy Hyper elastic energy type
 * @tparam TDerivedE Type of the element matrix
 * @tparam TDerivedeg Type of the element indices at quadrature points
 * @tparam TDerivedwg Type of the quadrature weights
 * @tparam TDerivedGNeg Type of the shape function gradients at quadrature points
 * @tparam TDerivedmug Type of the first Lame coefficients at quadrature points
 * @tparam TDerivedlambdag Type of the second Lame coefficients at quadrature points
 * @tparam TDerivedx Type of the deformed nodal positions
 * @tparam TDerivedGg Type of the element gradient vectors at quadrature points
 * @tparam TDerivedG Type of the output gradient vector
 * @param E `|# nodes per element| x |# elements|` matrix of mesh elements
 * @param nNodes Number of mesh nodes
 * @param eg `|# quad.pts.| x 1` vector of element indices at quadrature points
 * @param wg `|# quad.pts.| x 1` vector of quadrature weights
 * @param GNeg `|# nodes per element| x |# dims * # quad.pts.|` shape function gradients
 * @param mug `|# quad.pts.| x 1` first Lame coefficients at quadrature points
 * @param lambdag `|# quad.pts.| x 1` second Lame coefficients at quadrature points
 * @param x `|# dims * # nodes| x 1` deformed nodal positions
 * @param Gg `|# dims * # elem nodes| x |# quad.pts.|` array of element elastic gradient vectors at
 * quadrature points
 * @param G `|# dims * # nodes|` output gradient vector
 */
template <
    CMesh TMesh,
    physics::CHyperElasticEnergy THyperElasticEnergy,
    class TDerivedE,
    class TDerivedeg,
    class TDerivedwg,
    class TDerivedGNeg,
    class TDerivedmug,
    class TDerivedlambdag,
    class TDerivedx,
    class TDerivedGg,
    class TDerivedG>
inline void ToHyperElasticGradient(
    TMesh const& mesh,
    Eigen::DenseBase<TDerivedeg> const& eg,
    Eigen::DenseBase<TDerivedwg> const& wg,
    Eigen::MatrixBase<TDerivedGNeg> const& GNeg,
    Eigen::DenseBase<TDerivedmug> const& mug,
    Eigen::DenseBase<TDerivedlambdag> const& lambdag,
    Eigen::MatrixBase<TDerivedx> const& x,
    Eigen::MatrixBase<TDerivedGg> const& Gg,
    Eigen::PlainObjectBase<TDerivedG>& G)
{
    using ScalarType = typename TDerivedx::Scalar;
    Eigen::Matrix<ScalarType, 0, 0> dummyUg, dummyHg;
    using ElementType = typename TMesh::ElementType;
    using Dims        = typename TMesh::kDims;
    ToElementElasticity<THyperElasticEnergy>(
        mesh,
        eg.derived(),
        wg.derived(),
        GNeg.derived(),
        mug.derived(),
        lambdag.derived(),
        x.derived(),
        dummyUg,
        Gg.derived(),
        dummyHg,
        EElementElasticityComputationFlags::Gradient,
        EHyperElasticSpdCorrection::None);
    ToHyperElasticGradient(mesh, eg.derived(), Gg.derived(), G.derived());
}

/**
 * @brief Compute the gradient vector (allocates output)
 * @tparam TElement Element type
 * @tparam Dims Number of spatial dimensions
 * @tparam TDerivedE Type of the element matrix
 * @tparam TDerivedeg Type of the element indices at quadrature points
 * @tparam TDerivedGg Type of the element gradient vectors
 * @param E `|# nodes per element| x |# elements|` matrix of mesh elements
 * @param nNodes Number of mesh nodes
 * @param eg `|# quad.pts.| x 1` vector of element indices at quadrature points
 * @param Gg `|# dims * # elem nodes| x |# quad.pts.|` array of element elastic gradient vectors at
 * quadrature points
 * @return Gradient vector
 */
template <CElement TElement, int Dims, class TDerivedE, class TDerivedeg, class TDerivedGg>
auto HyperElasticGradient(
    Eigen::DenseBase<TDerivedE> const& E,
    typename TDerivedE::Scalar nNodes,
    Eigen::DenseBase<TDerivedeg> const& eg,
    Eigen::MatrixBase<TDerivedGg> const& Gg)
    -> Eigen::Vector<typename TDerivedGg::Scalar, Eigen::Dynamic>
{
    using ScalarType        = typename TDerivedGg::Scalar;
    auto const numberOfDofs = Dims * nNodes;
    Eigen::Vector<ScalarType, Eigen::Dynamic> G(numberOfDofs);
    ToHyperElasticGradient<TElement, Dims>(
        E.derived(),
        nNodes,
        eg.derived(),
        Gg.derived(),
        G.derived());
    return G;
}

/**
 * @brief Compute the gradient vector (allocates output)
 * @tparam TMesh Type of the mesh
 * @tparam THyperElasticEnergy Hyper elastic energy type
 * @tparam TDerivedeg Type of the element indices at quadrature points
 * @tparam TDerivedGg Type of the element gradient vectors
 * @param mesh Mesh containing element connectivity and node positions
 * @param eg `|# quad.pts.| x 1` vector of element indices at quadrature points
 * @param Gg `|# dims * # elem nodes| x |# quad.pts.|` array of element elastic gradient vectors at
 * quadrature points
 * @return Gradient vector
 */
template <CMesh TMesh, class TDerivedeg, class TDerivedGg>
auto HyperElasticGradient(
    TMesh const& mesh,
    Eigen::DenseBase<TDerivedeg> const& eg,
    Eigen::MatrixBase<TDerivedGg> const& Gg)
{
    return HyperElasticGradient<typename TMesh::ElementType, TMesh::kDims>(
        mesh.E,
        static_cast<typename TMesh::IndexType>(mesh.X.cols()),
        eg.derived(),
        Gg.derived());
}

/**
 * @brief Compute the gradient vector (allocates output)
 * @tparam TElement Element type
 * @tparam Dims Number of spatial dimensions
 * @tparam THyperElasticEnergy Hyper elastic energy type
 * @tparam TDerivedE Type of the element matrix
 * @tparam TDerivedeg Type of the element indices at quadrature points
 * @tparam TDerivedwg Type of the quadrature weights
 * @tparam TDerivedGNeg Type of the shape function gradients at quadrature points
 * @tparam TDerivedmug Type of the first Lame coefficients at quadrature points
 * @tparam TDerivedlambdag Type of the second Lame coefficients at quadrature points
 * @tparam TDerivedx Type of the deformed nodal positions
 * @tparam TDerivedGg Type of the element gradient vectors at quadrature points
 * @param E `|# nodes per element| x |# elements|` matrix of mesh elements
 * @param nNodes Number of mesh nodes
 * @param eg `|# quad.pts.| x 1` vector of element indices at quadrature points
 * @param wg `|# quad.pts.| x 1` vector of quadrature weights
 * @param GNeg `|# nodes per element| x |# dims * # quad.pts.|` shape function gradients
 * @param mug `|# quad.pts.| x 1` first Lame coefficients at quadrature points
 * @param lambdag `|# quad.pts.| x 1` second Lame coefficients at quadrature points
 * @param x `|# dims * # nodes| x 1` deformed nodal positions
 * @param Gg `|# dims * # elem nodes| x |# quad.pts.|` array of element gradient vectors at
 * quadrature points
 * @return Gradient vector
 */
template <
    CElement TElement,
    int Dims,
    physics::CHyperElasticEnergy THyperElasticEnergy,
    class TDerivedE,
    class TDerivedeg,
    class TDerivedwg,
    class TDerivedGNeg,
    class TDerivedmug,
    class TDerivedlambdag,
    class TDerivedx,
    class TDerivedGg>
auto HyperElasticGradient(
    Eigen::DenseBase<TDerivedE> const& E,
    typename TDerivedE::Scalar nNodes,
    Eigen::DenseBase<TDerivedeg> const& eg,
    Eigen::DenseBase<TDerivedwg> const& wg,
    Eigen::MatrixBase<TDerivedGNeg> const& GNeg,
    Eigen::DenseBase<TDerivedmug> const& mug,
    Eigen::DenseBase<TDerivedlambdag> const& lambdag,
    Eigen::MatrixBase<TDerivedx> const& x,
    Eigen::MatrixBase<TDerivedGg> const& Gg)
{
    using ScalarType = typename TDerivedx::Scalar;
    Eigen::Matrix<ScalarType, 0, 0> dummyUg, dummyHg;
    ToElementElasticity<TElement, Dims, THyperElasticEnergy>(
        E,
        nNodes,
        eg,
        wg,
        GNeg,
        mug,
        lambdag,
        x,
        dummyUg,
        Gg,
        dummyHg,
        EElementElasticityComputationFlags::Gradient,
        EHyperElasticSpdCorrection::None);
    return HyperElasticGradient<TElement, Dims, THyperElasticEnergy>(E, nNodes, eg, Gg);
}

/**
 * @brief Compute the gradient vector (allocates output)
 * @tparam TMesh Type of the mesh
 * @tparam THyperElasticEnergy Hyper elastic energy type
 * @tparam TDerivedeg Type of the element indices at quadrature points
 * @tparam TDerivedwg Type of the quadrature weights
 * @tparam TDerivedGNeg Type of the shape function gradients at quadrature points
 * @tparam TDerivedmug Type of the first Lame coefficients at quadrature points
 * @tparam TDerivedlambdag Type of the second Lame coefficients at quadrature points
 * @tparam TDerivedx Type of the deformed nodal positions
 * @tparam TDerivedGg Type of the element gradient vectors at quadrature points
 * @param mesh Mesh containing element connectivity and node positions
 * @param eg `|# quad.pts.| x 1` vector of element indices at quadrature points
 * @param wg `|# quad.pts.| x 1` vector of quadrature weights
 * @param GNeg `|# nodes per element| x |# dims * # quad.pts.|` shape function gradients
 * @param mug `|# quad.pts.| x 1` first Lame coefficients at quadrature points
 * @param lambdag `|# quad.pts.| x 1` second Lame coefficients at quadrature points
 * @param x `|# dims * # nodes| x 1` deformed nodal positions
 * @param Gg `|# dims * # elem nodes| x |# quad.pts.|` array of element gradient vectors at
 * quadrature points
 * @return `|# dims| x |# nodes|` gradient vector
 */
template <
    CMesh TMesh,
    physics::CHyperElasticEnergy THyperElasticEnergy,
    class TDerivedeg,
    class TDerivedwg,
    class TDerivedGNeg,
    class TDerivedmug,
    class TDerivedlambdag,
    class TDerivedx,
    class TDerivedGg>
auto HyperElasticGradient(
    TMesh const& mesh,
    Eigen::DenseBase<TDerivedeg> const& eg,
    Eigen::DenseBase<TDerivedwg> const& wg,
    Eigen::MatrixBase<TDerivedGNeg> const& GNeg,
    Eigen::DenseBase<TDerivedmug> const& mug,
    Eigen::DenseBase<TDerivedlambdag> const& lambdag,
    Eigen::MatrixBase<TDerivedx> const& x,
    Eigen::MatrixBase<TDerivedGg> const& Gg)
{
    return HyperElasticGradient<typename TMesh::ElementType, TMesh::kDims, THyperElasticEnergy>(
        mesh.E,
        static_cast<typename TMesh::IndexType>(mesh.X.cols()),
        eg.derived(),
        wg.derived(),
        GNeg.derived(),
        mug.derived(),
        lambdag.derived(),
        x.derived(),
        Gg.derived());
}

/**
 * @brief Compute the gradient vector using mesh (allocates output)
 * @tparam THyperElasticEnergy Hyper elastic energy type
 * @tparam TMesh Type of the mesh
 * @tparam TDerivedeg Type of the element indices at quadrature points
 * @tparam TDerivedGg Type of the element gradient vectors
 * @param mesh Mesh containing element connectivity and node positions
 * @param eg `|# quad.pts.| x 1` vector of element indices at quadrature points
 * @param Gg `|# dims * # elem nodes| x |# quad.pts.|` array of element elastic gradient vectors at
 * quadrature points
 * @return Gradient vector
 */
template <
    physics::CHyperElasticEnergy THyperElasticEnergy,
    CMesh TMesh,
    class TDerivedeg,
    class TDerivedGg>
auto HyperElasticGradient(
    TMesh const& mesh,
    Eigen::DenseBase<TDerivedeg> const& eg,
    Eigen::MatrixBase<TDerivedGg> const& Gg)
{
    return HyperElasticGradient<typename TMesh::ElementType, TMesh::kDims, THyperElasticEnergy>(
        mesh.E,
        static_cast<typename TMesh::IndexType>(mesh.X.cols()),
        eg.derived(),
        Gg.derived());
}

/**
 * @brief Compute the total elastic potential
 * @tparam TDerivedUg Type of the elastic potentials at quadrature points
 * @param Ug `|# quad.pts.| x 1` array of elastic potentials at quadrature points
 * @return Total elastic potential
 */
template <class TDerivedUg>
auto HyperElasticPotential(Eigen::DenseBase<TDerivedUg> const& Ug) -> typename TDerivedUg::Scalar
{
    return Ug.sum();
}

/**
 * @brief Compute the total elastic potential from element data and quadrature
 * @tparam TElement Element type
 * @tparam Dims Number of spatial dimensions
 * @tparam THyperElasticEnergy Hyper elastic energy type
 * @tparam TDerivedE Type of the element matrix
 * @tparam TDerivedeg Type of the element indices at quadrature points
 * @tparam TDerivedwg Type of the quadrature weights
 * @tparam TDerivedGNeg Type of the shape function gradients at quadrature points
 * @tparam TDerivedmug Type of the first Lame coefficients at quadrature points
 * @tparam TDerivedlambdag Type of the second Lame coefficients at quadrature points
 * @tparam TDerivedx Type of the deformed nodal positions
 * @param E `|# nodes per element| x |# elements|` matrix of mesh elements
 * @param nNodes Number of mesh nodes
 * @param eg `|# quad.pts.| x 1` vector of element indices at quadrature points
 * @param wg `|# quad.pts.| x 1` vector of quadrature weights
 * @param GNeg `|# nodes per element| x |# dims * # quad.pts.|` shape function gradients
 * @param mug `|# quad.pts.| x 1` first Lame coefficients at quadrature points
 * @param lambdag `|# quad.pts.| x 1` second Lame coefficients at quadrature points
 * @param x `|# dims * # nodes| x 1` deformed nodal positions
 * @return Total elastic potential
 */
template <
    CElement TElement,
    int Dims,
    physics::CHyperElasticEnergy THyperElasticEnergy,
    class TDerivedE,
    class TDerivedeg,
    class TDerivedwg,
    class TDerivedGNeg,
    class TDerivedmug,
    class TDerivedlambdag,
    class TDerivedx>
auto HyperElasticPotential(
    Eigen::DenseBase<TDerivedE> const& E,
    typename TDerivedE::Scalar nNodes,
    Eigen::DenseBase<TDerivedeg> const& eg,
    Eigen::DenseBase<TDerivedwg> const& wg,
    Eigen::MatrixBase<TDerivedGNeg> const& GNeg,
    Eigen::DenseBase<TDerivedmug> const& mug,
    Eigen::DenseBase<TDerivedlambdag> const& lambdag,
    Eigen::MatrixBase<TDerivedx> const& x)
{
    using ScalarType = typename TDerivedx::Scalar;
    Eigen::Vector<ScalarType, Eigen::Dynamic> Ug;
    Eigen::Matrix<ScalarType, 0, 0> dummyGg, dummyHg;
    ToElementElasticity<TElement, Dims, THyperElasticEnergy>(
        E,
        nNodes,
        eg,
        wg,
        GNeg,
        mug,
        lambdag,
        x,
        Ug,
        dummyGg,
        dummyHg,
        EElementElasticityComputationFlags::Potential,
        EHyperElasticSpdCorrection::None);
    return HyperElasticPotential(Ug);
}

template <
    CElement TElement,
    int Dims,
    physics::CHyperElasticEnergy THyperElasticEnergy,
    class TDerivedE,
    class TDerivedeg,
    class TDerivedwg,
    class TDerivedGNeg,
    class TDerivedmug,
    class TDerivedlambdag,
    class TDerivedx,
    class TDerivedUg,
    class TDerivedGg,
    class TDerivedHg>
void ToElementElasticity(
    Eigen::DenseBase<TDerivedE> const& E,
    typename TDerivedE::Scalar nNodes,
    Eigen::DenseBase<TDerivedeg> const& eg,
    Eigen::DenseBase<TDerivedwg> const& wg,
    Eigen::MatrixBase<TDerivedGNeg> const& GNeg,
    Eigen::DenseBase<TDerivedmug> const& mug,
    Eigen::DenseBase<TDerivedlambdag> const& lambdag,
    Eigen::MatrixBase<TDerivedx> const& x,
    Eigen::PlainObjectBase<TDerivedUg>& Ug,
    Eigen::PlainObjectBase<TDerivedGg>& Gg,
    Eigen::PlainObjectBase<TDerivedHg>& Hg,
    int eFlags,
    EHyperElasticSpdCorrection eSpdCorrection)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.fem.ToElementElasticity");

    using ScalarType = typename TDerivedx::Scalar;
    using IndexType  = typename TDerivedE::Scalar;

    // Check inputs
    if (x.size() != nNodes * Dims)
    {
        std::string const what = fmt::format(
            "Generalized coordinate vector must have dimensions |#nodes|*kDims={}, but got "
            "x.size()={}",
            nNodes * Dims,
            x.size());
        throw std::invalid_argument(what);
    }

    IndexType const nQuadPts        = wg.size();
    auto constexpr kNodesPerElement = TElement::kNodes;
    auto constexpr kDofsPerElement  = kNodesPerElement * Dims;
    namespace mini                  = math::linalg::mini;
    using mini::FromEigen;
    using mini::ToEigen;

    if (eFlags & EElementElasticityComputationFlags::Potential)
    {
        Ug.resize(nQuadPts);
        Ug.setZero();
    }
    if (eFlags & EElementElasticityComputationFlags::Gradient)
    {
        Gg.resize(kDofsPerElement, nQuadPts);
        Gg.setZero();
    }
    if (eFlags & EElementElasticityComputationFlags::Hessian)
    {
        Hg.resize(kDofsPerElement, kDofsPerElement * nQuadPts);
        Hg.setZero();
    }

    THyperElasticEnergy Psi{};
    if (eFlags == EElementElasticityComputationFlags::Potential)
    {
        tbb::parallel_for(IndexType{0}, nQuadPts, [&](auto g) {
            auto const e               = eg(g);
            auto const nodes           = E.col(e);
            auto const xe              = x.reshaped(Dims, nNodes)(Eigen::placeholders::all, nodes);
            auto const GPeg            = GNeg.template block<kNodesPerElement, Dims>(0, g * Dims);
            Matrix<Dims, Dims> const F = xe * GPeg;
            auto vecF                  = FromEigen(F);
            auto psiF                  = Psi.eval(vecF, mug(g), lambdag(g));
            Ug(g) += wg(g) * psiF;
        });
    }
    else if (
        eFlags == (EElementElasticityComputationFlags::Potential |
                   EElementElasticityComputationFlags::Gradient))
    {
        tbb::parallel_for(IndexType{0}, nQuadPts, [&](auto g) {
            auto const e               = eg(g);
            auto const nodes           = E.col(e);
            auto const xe              = x.reshaped(Dims, nNodes)(Eigen::placeholders::all, nodes);
            auto const GPeg            = GNeg.template block<kNodesPerElement, Dims>(0, g * Dims);
            Matrix<Dims, Dims> const F = xe * GPeg;
            auto vecF                  = FromEigen(F);
            mini::SVector<ScalarType, Dims * Dims> gradPsiF;
            auto const psiF  = Psi.evalWithGrad(vecF, mug(g), lambdag(g), gradPsiF);
            auto const GP    = FromEigen(GPeg);
            auto const GPsix = GradientWrtDofs<TElement, Dims>(gradPsiF, GP);
            Ug(g) += wg(g) * psiF;
            Gg.col(g) += wg(g) * ToEigen(GPsix);
        });
    }
    else if (
        eFlags == (EElementElasticityComputationFlags::Potential |
                   EElementElasticityComputationFlags::Hessian))
    {
        tbb::parallel_for(IndexType{0}, nQuadPts, [&](auto g) {
            auto const e               = eg(g);
            auto const nodes           = E.col(e);
            auto const xe              = x.reshaped(Dims, nNodes)(Eigen::placeholders::all, nodes);
            auto const gradPhi         = GNeg.template block<kNodesPerElement, Dims>(0, g * Dims);
            Matrix<Dims, Dims> const F = xe * gradPhi;
            auto vecF                  = FromEigen(F);
            auto psiF                  = Psi.eval(vecF, mug(g), lambdag(g));
            auto const hessPsiF        = Psi.hessian(vecF, mug(g), lambdag(g));
            auto const GP              = FromEigen(gradPhi);
            auto HPsix                 = HessianWrtDofs<TElement, Dims>(hessPsiF, GP);
            auto heg = Hg.template block<kDofsPerElement, kDofsPerElement>(0, g * kDofsPerElement);
            Ug(g) += wg(g) * psiF;
            heg += wg(g) * ToEigen(HPsix);
        });
    }
    else if (
        eFlags == (EElementElasticityComputationFlags::Potential |
                   EElementElasticityComputationFlags::Gradient |
                   EElementElasticityComputationFlags::Hessian))
    {
        tbb::parallel_for(IndexType{0}, nQuadPts, [&](auto g) {
            auto const e               = eg(g);
            auto const nodes           = E.col(e);
            auto const xe              = x.reshaped(Dims, nNodes)(Eigen::placeholders::all, nodes);
            auto const GPeg            = GNeg.template block<kNodesPerElement, Dims>(0, g * Dims);
            Matrix<Dims, Dims> const F = xe * GPeg;
            auto vecF                  = FromEigen(F);
            mini::SVector<ScalarType, Dims * Dims> gradPsiF;
            mini::SMatrix<ScalarType, Dims * Dims, Dims * Dims> hessPsiF;
            auto psiF = Psi.evalWithGradAndHessian(vecF, mug(g), lambdag(g), gradPsiF, hessPsiF);
            auto const GP = FromEigen(GPeg);
            auto GPsix    = GradientWrtDofs<TElement, Dims>(gradPsiF, GP);
            auto HPsix    = HessianWrtDofs<TElement, Dims>(hessPsiF, GP);
            auto heg = Hg.template block<kDofsPerElement, kDofsPerElement>(0, g * kDofsPerElement);
            Ug(g) += wg(g) * psiF;
            Gg.col(g) += wg(g) * ToEigen(GPsix);
            heg += wg(g) * ToEigen(HPsix);
        });
    }
    else if (eFlags == EElementElasticityComputationFlags::Gradient)
    {
        tbb::parallel_for(IndexType{0}, nQuadPts, [&](auto g) {
            auto const e               = eg(g);
            auto const nodes           = E.col(e);
            auto const xe              = x.reshaped(Dims, nNodes)(Eigen::placeholders::all, nodes);
            auto const GPeg            = GNeg.template block<kNodesPerElement, Dims>(0, g * Dims);
            Matrix<Dims, Dims> const F = xe * GPeg;
            auto vecF                  = FromEigen(F);
            auto const gradPsiF        = Psi.grad(vecF, mug(g), lambdag(g));
            auto const GP              = FromEigen(GPeg);
            auto const GPsix           = GradientWrtDofs<TElement, Dims>(gradPsiF, GP);
            Gg.col(g) += wg(g) * ToEigen(GPsix);
        });
    }
    else if (
        eFlags == (EElementElasticityComputationFlags::Gradient |
                   EElementElasticityComputationFlags::Hessian))
    {
        tbb::parallel_for(IndexType{0}, nQuadPts, [&](auto g) {
            auto const e               = eg(g);
            auto const nodes           = E.col(e);
            auto const xe              = x.reshaped(Dims, nNodes)(Eigen::placeholders::all, nodes);
            auto const GPeg            = GNeg.template block<kNodesPerElement, Dims>(0, g * Dims);
            Matrix<Dims, Dims> const F = xe * GPeg;
            auto vecF                  = FromEigen(F);
            mini::SVector<ScalarType, Dims * Dims> gradPsiF;
            mini::SMatrix<ScalarType, Dims * Dims, Dims * Dims> hessPsiF;
            Psi.evalWithGradAndHessian(vecF, mug(g), lambdag(g), gradPsiF, hessPsiF);
            auto const GP = FromEigen(GPeg);
            auto GPsix    = GradientWrtDofs<TElement, Dims>(gradPsiF, GP);
            auto HPsix    = HessianWrtDofs<TElement, Dims>(hessPsiF, GP);
            auto heg = Hg.template block<kDofsPerElement, kDofsPerElement>(0, g * kDofsPerElement);
            Gg.col(g) += wg(g) * ToEigen(GPsix);
            heg += wg(g) * ToEigen(HPsix);
        });
    }
    else /* if (eFlags == EElementElasticityComputationFlags::Hessian) */
    {
        tbb::parallel_for(IndexType{0}, nQuadPts, [&](auto g) {
            auto const e               = eg(g);
            auto const nodes           = E.col(e);
            auto const xe              = x.reshaped(Dims, nNodes)(Eigen::placeholders::all, nodes);
            auto const GPeg            = GNeg.template block<kNodesPerElement, Dims>(0, g * Dims);
            Matrix<Dims, Dims> const F = xe * GPeg;
            auto vecF                  = FromEigen(F);
            auto hessPsiF              = Psi.hessian(vecF, mug(g), lambdag(g));
            auto const GP              = FromEigen(GPeg);
            auto HPsix                 = HessianWrtDofs<TElement, Dims>(hessPsiF, GP);
            auto heg = Hg.template block<kDofsPerElement, kDofsPerElement>(0, g * kDofsPerElement);
            heg += wg(g) * ToEigen(HPsix);
        });
    }
    // SPD correction
    if (eFlags & EElementElasticityComputationFlags::Hessian)
    {
        using ElementHessianMatrixType = Matrix<kDofsPerElement, kDofsPerElement>;
        switch (eSpdCorrection)
        {
            case EHyperElasticSpdCorrection::None: break;
            case EHyperElasticSpdCorrection::Absolute: {
                tbb::parallel_for(typename TDerivedE::Scalar{0}, nQuadPts, [&](auto g) {
                    auto heg =
                        Hg.template block<kDofsPerElement, kDofsPerElement>(0, g * kDofsPerElement);
                    Eigen::SelfAdjointEigenSolver<ElementHessianMatrixType> eig(
                        heg,
                        Eigen::ComputeEigenvectors);
                    auto D = eig.eigenvalues();
                    auto V = eig.eigenvectors();
                    for (auto i = 0; i < D.size(); ++i)
                    {
                        if (D(i) >= 0)
                            break;
                        D(i) = -D(i);
                    }
                    heg = V * D.asDiagonal() * V.transpose();
                });
                break;
            }
            case EHyperElasticSpdCorrection::Projection: {
                tbb::parallel_for(typename TDerivedE::Scalar{0}, nQuadPts, [&](auto g) {
                    auto heg =
                        Hg.template block<kDofsPerElement, kDofsPerElement>(0, g * kDofsPerElement);
                    Eigen::SelfAdjointEigenSolver<ElementHessianMatrixType> eig(
                        heg,
                        Eigen::ComputeEigenvectors);
                    auto D = eig.eigenvalues();
                    auto V = eig.eigenvectors();
                    for (auto i = 0; i < D.size(); ++i)
                    {
                        if (D(i) >= 0)
                            break;
                        D(i) = 0;
                    }
                    heg = V * D.asDiagonal() * V.transpose();
                });
                break;
            }
            default:
                throw std::invalid_argument(
                    "Unknown SPD correction type for hyper elastic hessian");
        }
    }
}

template <
    CElement TElement,
    int Dims,
    class TDerivedE,
    class TDerivedeg,
    class TDerivedHg,
    class TDerivedIn,
    class TDerivedOut>
inline void GemmHyperElastic(
    Eigen::DenseBase<TDerivedE> const& E,
    typename TDerivedE::Scalar nNodes,
    Eigen::DenseBase<TDerivedeg> const& eg,
    Eigen::MatrixBase<TDerivedHg> const& Hg,
    Eigen::MatrixBase<TDerivedIn> const& X,
    Eigen::DenseBase<TDerivedOut>& Y)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.fem.GemmHyperElastic");
    auto const numberOfDofs = Dims * nNodes;
    if (X.rows() != numberOfDofs or Y.rows() != numberOfDofs or X.cols() != Y.cols())
    {
        std::string const what = fmt::format(
            "Expected inputs and outputs to have rows |#nodes*kDims|={} and same number of "
            "columns, but got dimensions "
            "X,Y=({},{}), ({},{})",
            numberOfDofs,
            X.rows(),
            X.cols(),
            Y.rows(),
            Y.cols());
        throw std::invalid_argument(what);
    }

    auto constexpr kDofsPerElement = Dims * TElement::kNodes;
    auto const nQuadPts            = eg.size();
    for (auto c = 0; c < X.cols(); ++c)
    {
        for (auto g = 0; g < nQuadPts; ++g)
        {
            auto const e     = eg(g);
            auto const nodes = E.col(e);
            auto const heg =
                Hg.template block<kDofsPerElement, kDofsPerElement>(0, g * kDofsPerElement);
            auto const xe = X.col(c).reshaped(Dims, nNodes)(Eigen::placeholders::all, nodes);
            auto ye       = Y.col(c).reshaped(Dims, nNodes)(Eigen::placeholders::all, nodes);
            ye.reshaped() += heg * xe.reshaped();
        }
    }
}

template <
    CElement TElement,
    int Dims,
    Eigen::StorageOptions Options,
    class TDerivedE,
    class TDerivedeg>
auto ElasticHessianSparsity(
    Eigen::DenseBase<TDerivedE> const& E,
    typename TDerivedE::Scalar nNodes,
    Eigen::DenseBase<TDerivedeg> const& eg)
    -> math::linalg::SparsityPattern<typename TDerivedE::Scalar, Options>
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.fem.ElasticHessianSparsity");
    using IndexType                     = typename TDerivedE::Scalar;
    auto const numberOfQuadraturePoints = eg.size();
    auto const kNodesPerElement         = TElement::kNodes;
    auto const kDofsPerElement          = kNodesPerElement * Dims;
    auto const nDofs                    = Dims * nNodes;
    std::vector<IndexType> nonZeroRowIndices{};
    std::vector<IndexType> nonZeroColIndices{};
    nonZeroRowIndices.reserve(
        static_cast<std::size_t>(kDofsPerElement * kDofsPerElement * numberOfQuadraturePoints));
    nonZeroColIndices.reserve(
        static_cast<std::size_t>(kDofsPerElement * kDofsPerElement * numberOfQuadraturePoints));
    // Insert non-zero indices in the storage order of our Hg matrix of element hessians at
    // quadrature points
    for (auto g = 0; g < numberOfQuadraturePoints; ++g)
    {
        auto const e     = eg(g);
        auto const nodes = E.col(e);
        for (auto j = 0; j < kNodesPerElement; ++j)
        {
            for (auto dj = 0; dj < Dims; ++dj)
            {
                for (auto i = 0; i < kNodesPerElement; ++i)
                {
                    for (auto di = 0; di < Dims; ++di)
                    {
                        nonZeroRowIndices.push_back(Dims * nodes(i) + di);
                        nonZeroColIndices.push_back(Dims * nodes(j) + dj);
                    }
                }
            }
        }
    }
    math::linalg::SparsityPattern<IndexType, Options> GH;
    GH.Compute(nDofs, nDofs, nonZeroRowIndices, nonZeroColIndices);
    return GH;
}

template <
    CElement TElement,
    int Dims,
    Eigen::StorageOptions Options,
    class TDerivedE,
    class TDerivedeg,
    class TDerivedHg>
auto HyperElasticHessian(
    Eigen::DenseBase<TDerivedE> const& E,
    typename TDerivedE::Scalar nNodes,
    Eigen::DenseBase<TDerivedeg> const& eg,
    Eigen::MatrixBase<TDerivedHg> const& Hg)
    -> Eigen::SparseMatrix<typename TDerivedHg::Scalar, Options, typename TDerivedE::Scalar>
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.fem.HyperElasticHessian");
    using ScalarType       = typename TDerivedHg::Scalar;
    using IndexType        = typename TDerivedE::Scalar;
    using SparseMatrixType = Eigen::SparseMatrix<ScalarType, Options, IndexType>;
    using Triplet          = Eigen::Triplet<ScalarType, IndexType>;

    auto constexpr kNodesPerElement = TElement::kNodes;
    auto constexpr kDofsPerElement  = kNodesPerElement * Dims;
    auto const nQuadPts             = eg.size();

    std::vector<Triplet> triplets{};
    triplets.reserve(static_cast<std::size_t>(kDofsPerElement * kDofsPerElement * nQuadPts));

    auto const numberOfDofs = Dims * nNodes;
    SparseMatrixType H(numberOfDofs, numberOfDofs);

    for (auto g = 0; g < nQuadPts; ++g)
    {
        auto const e     = eg(g);
        auto const nodes = E.col(e);
        auto const heg =
            Hg.template block<kDofsPerElement, kDofsPerElement>(0, g * kDofsPerElement);
        for (auto j = 0; j < kNodesPerElement; ++j)
        {
            for (auto dj = 0; dj < Dims; ++dj)
            {
                for (auto i = 0; i < kNodesPerElement; ++i)
                {
                    for (auto di = 0; di < Dims; ++di)
                    {
                        auto const ni = static_cast<IndexType>(Dims * nodes(i) + di);
                        auto const nj = static_cast<IndexType>(Dims * nodes(j) + dj);
                        triplets.emplace_back(ni, nj, heg(Dims * i + di, Dims * j + dj));
                    }
                }
            }
        }
    }
    H.setFromTriplets(triplets.begin(), triplets.end());
    return H;
}

template <
    CElement TElement,
    int Dims,
    class TDerivedE,
    class TDerivedHg,
    Eigen::StorageOptions Options,
    class TDerivedH>
void ToHyperElasticHessian(
    Eigen::DenseBase<TDerivedE> const& E,
    typename TDerivedE::Scalar nNodes,
    Eigen::DenseBase<TDerivedHg> const& Hg,
    math::linalg::SparsityPattern<typename TDerivedE::Scalar, Options> const& sparsity,
    Eigen::SparseCompressedBase<TDerivedH>& H)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.fem.ToHyperElasticHessian");
    using SpanType = std::span<Scalar const>;
    using SizeType = typename SpanType::size_type;
    sparsity.To(SpanType(Hg.data(), static_cast<SizeType>(Hg.size())), H);
}

template <
    CElement TElement,
    int Dims,
    Eigen::StorageOptions Options,
    class TDerivedE,
    class TDerivedHg>
auto HyperElasticHessian(
    Eigen::DenseBase<TDerivedE> const& E,
    typename TDerivedE::Scalar nNodes,
    Eigen::DenseBase<TDerivedHg> const& Hg,
    math::linalg::SparsityPattern<typename TDerivedE::Scalar, Options> const& sparsity)
    -> Eigen::SparseMatrix<typename TDerivedHg::Scalar, Options, typename TDerivedE::Scalar>
{
    using SpanType = std::span<Scalar const>;
    using SizeType = typename SpanType::size_type;
    return sparsity.ToMatrix(SpanType(Hg.data(), static_cast<SizeType>(Hg.size())));
}

template <
    CElement TElement,
    int Dims,
    class TDerivedE,
    class TDerivedeg,
    class TDerivedGg,
    class TDerivedOut>
inline void ToHyperElasticGradient(
    Eigen::DenseBase<TDerivedE> const& E,
    typename TDerivedE::Scalar nNodes,
    Eigen::DenseBase<TDerivedeg> const& eg,
    Eigen::MatrixBase<TDerivedGg> const& Gg,
    Eigen::PlainObjectBase<TDerivedOut>& G)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.fem.ToHyperElasticGradient");
    G.resize(Dims * nNodes, 1);
    G.setZero();
    auto constexpr kNodesPerElement = TElement::kNodes;
    auto const nQuadPts             = eg.size();
    auto unvecG                     = G.reshaped(Dims, nNodes);
    for (auto g = 0; g < nQuadPts; ++g)
    {
        auto const e     = eg(g);
        auto const nodes = E.col(e);
        auto const geg   = Gg.col(g).reshaped(Dims, kNodesPerElement);
        auto gi          = unvecG(Eigen::placeholders::all, nodes);
        gi += geg;
    }
}

} // namespace fem
} // namespace pbat

#endif // PBAT_FEM_HYPERELASTICPOTENTIAL_H