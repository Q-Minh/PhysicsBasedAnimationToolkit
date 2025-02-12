/**
 * @file LaplacianMatrix.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief SymmetricLaplacianMatrix API and implementation.
 * @date 2025-02-11
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_FEM_LAPLACIAN_MATRIX_H
#define PBAT_FEM_LAPLACIAN_MATRIX_H

#include "Concepts.h"

#include <exception>
#include <fmt/core.h>
#include <pbat/Aliases.h>
#include <pbat/common/Eigen.h>
#include <pbat/profiling/Profiling.h>
#include <tbb/parallel_for.h>

namespace pbat {
namespace fem {

/**
 * @brief A matrix-free representation of the symmetric part of the Laplacian \f$\Delta u\f$ of a
 * finite element discretized function \f$ u(X) \f$ under Galerkin projection.
 *
 * The precise definition of the Laplacian matrix's symmetric part is given by
 * \f$ \mathbf{L}_{ij} = \int_\Omega -\nabla \phi_i \cdot \nabla \phi_j d\Omega \f$.
 *
 * @todo Explain the Laplacian matrix and its construction and link to my higher-level FEM crash
 * course doc.
 *
 * This matrix-free Laplacian requires the following inputs:
 * - A finite element `mesh` satisfying concept CMesh
 * - A vector of element indices `eg`, associating each quadrature point with an element
 * - A vector of quadrature weights `wg`
 * - A matrix `GNegg` of element shape function gradients at quadrature points (see
 * ShapeFunctionGradients())
 * - An integer `dims` specifying the dimensionality of the image of the FEM function space
 *
 * @note
 * The user-provided quadrature rule is injected into the Laplacian operator, allowing for
 * arbitrary quadrature rules to be used. Since Laplacians of higher-dimensional functions
 * are only one-dimensional Laplacian 'kroneckered' with the identity matrix, the Laplacian
 * matrix is actually \f$ L \otimes I_{d \times d} \f$ where \f$ d \f$ is the function's
 * dimensionality, but we need not store the duplicate entries.
 *
 * @tparam TMesh Type satisfying concept CMesh
 */
template <CMesh TMesh>
struct SymmetricLaplacianMatrix
{
  public:
    using SelfType    = SymmetricLaplacianMatrix<TMesh>; ///< Self type
    using MeshType    = TMesh;                           ///< Mesh type
    using ElementType = typename TMesh::ElementType;     ///< Element type

    static int constexpr kOrder =
        2 * (ElementType::kOrder - 1); ///< Polynomial order of the Laplacian matrix

    /**
     * @brief Construct a new symmetric Laplacian operator
     *
     * @param mesh Finite element mesh
     * @param eg Element indices associating each quadrature point with an element
     * @param wg Quadrature weights
     * @param GNegg Element shape function gradients at quadrature points
     * @param dims Dimensionality of the image of the FEM function space
     * @pre `eg.size() == wg.size()` and `GNegg.rows() == mesh.E.rows()` and `dims >= 1`
     * 
     */
    SymmetricLaplacianMatrix(
        MeshType const& mesh,
        Eigen::Ref<IndexVectorX const> const& eg,
        Eigen::Ref<VectorX const> const& wg,
        Eigen::Ref<MatrixX const> const& GNegg,
        int dims = 1);

    SelfType& operator=(SelfType const&) = delete;

    /**
     * @brief Applies this matrix as a linear operator on x, adding result to y.
     *
     * @tparam TDerivedIn Input matrix type
     * @tparam TDerivedOut Output matrix type
     * @param x Input matrix
     * @param y Output matrix
     * @pre x.rows() == InputDimensions() and y.rows() == OutputDimensions() and y.cols() ==
     * x.cols()
     */
    template <class TDerivedIn, class TDerivedOut>
    void Apply(Eigen::MatrixBase<TDerivedIn> const& x, Eigen::DenseBase<TDerivedOut>& y) const;

    /**
     * @brief Transforms this matrix-free matrix representation into sparse compressed format.
     * @return Sparse compressed column matrix representation of this Laplacian matrix
     */
    CSCMatrix ToMatrix() const;

    /**
     * @brief Number of columns
     *
     * @return Number of columns
     */
    Index InputDimensions() const { return dims * mesh.X.cols(); }
    /**
     * @brief Number of rows
     *
     * @return Number of rows
     */
    Index OutputDimensions() const { return InputDimensions(); }

    /**
     * @brief Compute and store the element laplacians
     */
    void ComputeElementLaplacians();
    /**
     * @brief Check if the state of this Laplacian matrix is valid
     */
    void CheckValidState() const;

    MeshType const& mesh; ///< The finite element mesh
    Eigen::Ref<IndexVectorX const>
        eg; ///< `|# quad.pts.|x1` array of elements associated with quadrature points
    Eigen::Ref<VectorX const> wg; ///< `|# quad.pts.|x1` array of quadrature weights
    Eigen::Ref<MatrixX const>
        GNeg;       ///< `|# element nodes|x|# dims * # quad.pts. * # elements|`
                    ///< matrix of element shape function gradients at quadrature points
    MatrixX deltag; ///< `|# element nodes| x |# element nodes * # quad.pts.|` matrix of element
                    ///< laplacians at quadrature points
    int dims; ///< Dimensionality of image of FEM function space, i.e. this Laplacian matrix is
              ///< actually \f$ L \otimes I_{d} \f$. Must have `dims >= 1`.
};

template <CMesh TMesh>
inline SymmetricLaplacianMatrix<TMesh>::SymmetricLaplacianMatrix(
    MeshType const& mesh,
    Eigen::Ref<IndexVectorX const> const& eg,
    Eigen::Ref<VectorX const> const& wg,
    Eigen::Ref<MatrixX const> const& GNeg,
    int dims)
    : mesh(mesh), eg(eg), wg(wg), GNeg(GNeg), deltag(), dims(dims)
{
    ComputeElementLaplacians();
}

template <CMesh TMesh>
inline CSCMatrix SymmetricLaplacianMatrix<TMesh>::ToMatrix() const
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.fem.SymmetricLaplacianMatrix.ToMatrix");
    CheckValidState();
    CSCMatrix L(OutputDimensions(), InputDimensions());
    using SparseIndex = typename CSCMatrix::StorageIndex;
    using Triplet     = Eigen::Triplet<Scalar, SparseIndex>;

    std::vector<Triplet> triplets{};
    triplets.reserve(static_cast<std::size_t>(deltag.size() * dims));
    auto const numberOfQuadraturePoints = wg.size();
    for (auto g = 0; g < numberOfQuadraturePoints; ++g)
    {
        auto const e                    = eg(g);
        auto const nodes                = mesh.E.col(e);
        auto constexpr kNodesPerElement = ElementType::kNodes;
        auto const Leg = deltag.block(0, g * kNodesPerElement, kNodesPerElement, kNodesPerElement);
        for (auto j = 0; j < Leg.cols(); ++j)
        {
            for (auto i = 0; i < Leg.rows(); ++i)
            {
                for (auto d = 0; d < dims; ++d)
                {
                    auto const ni = static_cast<SparseIndex>(dims * nodes(i) + d);
                    auto const nj = static_cast<SparseIndex>(dims * nodes(j) + d);
                    triplets.push_back(Triplet(ni, nj, Leg(i, j)));
                }
            }
        }
    }
    L.setFromTriplets(triplets.begin(), triplets.end());
    return L;
}

template <CMesh TMesh>
inline void SymmetricLaplacianMatrix<TMesh>::ComputeElementLaplacians()
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.fem.SymmetricLaplacianMatrix.ComputeElementLaplacians");
    CheckValidState();
    // Compute element laplacians
    auto constexpr kNodesPerElement     = ElementType::kNodes;
    auto constexpr kDims                = MeshType::kDims;
    auto const numberOfQuadraturePoints = wg.size();
    deltag.setZero(kNodesPerElement, kNodesPerElement * numberOfQuadraturePoints);
    tbb::parallel_for(Index{0}, Index{numberOfQuadraturePoints}, [&](Index g) {
        auto Leg = deltag.block<kNodesPerElement, kNodesPerElement>(0, g * kNodesPerElement);
        // Use multivariable integration by parts (i.e. Green's identity), and retain only the
        // symmetric part, i.e.
        // Lij = -\int_{\Omega} \nabla \phi_i(X) \cdot \nabla \phi_j(X) \partial \Omega.
        auto const GP = GNeg.block<kNodesPerElement, kDims>(0, g * kDims);
        Leg -= wg(g) * GP * GP.transpose();
    });
}

template <CMesh TMesh>
inline void SymmetricLaplacianMatrix<TMesh>::CheckValidState() const
{
    auto const numberOfQuadraturePoints = wg.size();
    auto constexpr kExpectedGNegRows    = ElementType::kNodes;
    auto const expectedGNegCols         = MeshType::kDims * numberOfQuadraturePoints;
    bool const bShapeFunctionGradientsHaveCorrectDimensions =
        (GNeg.rows() == kExpectedGNegRows) and (GNeg.cols() == expectedGNegCols);
    if (not bShapeFunctionGradientsHaveCorrectDimensions)
    {
        std::string const what = fmt::format(
            "Expected shape function gradients at element quadrature points of dimensions "
            "|#nodes-per-element|={} x |#mesh-dims * #quad.pts.|={} for polynomial but got {}x{} "
            "instead",
            kExpectedGNegRows,
            expectedGNegCols,
            GNeg.rows(),
            GNeg.cols());
        throw std::invalid_argument(what);
    }
    if (dims < 1)
    {
        std::string const what =
            fmt::format("Expected output dimensionality >= 1, got {} instead", dims);
        throw std::invalid_argument(what);
    }
}

template <CMesh TMesh>
template <class TDerivedIn, class TDerivedOut>
inline void SymmetricLaplacianMatrix<TMesh>::Apply(
    Eigen::MatrixBase<TDerivedIn> const& x,
    Eigen::DenseBase<TDerivedOut>& y) const
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.fem.SymmetricLaplacianMatrix.Apply");
    CheckValidState();
    auto const numberOfDofs = InputDimensions();
    bool const bAreInputOutputValid =
        (x.rows() != numberOfDofs) or (y.rows() != numberOfDofs) or (y.cols() != x.cols());
    if (bAreInputOutputValid)
    {
        std::string const what = fmt::format(
            "Expected input x and output y with matching dimensions and {} rows, but got {}x{} "
            "input and {}x{} output",
            numberOfDofs,
            x.rows(),
            x.cols(),
            y.rows(),
            y.cols());
        throw std::invalid_argument(what);
    }

    auto const numberOfQuadraturePoints = wg.size();
    for (auto c = 0; c < x.cols(); ++c)
    {
        for (auto g = 0; g < numberOfQuadraturePoints; ++g)
        {
            auto const e                    = eg(g);
            auto const nodes                = mesh.E.col(e);
            auto constexpr kNodesPerElement = ElementType::kNodes;
            auto const Leg =
                deltag.block(0, g * kNodesPerElement, kNodesPerElement, kNodesPerElement);
            auto ye = y.col(c).reshaped(dims, y.size() / dims)(Eigen::placeholders::all, nodes);
            auto const xe =
                x.col(c).reshaped(dims, x.size() / dims)(Eigen::placeholders::all, nodes);
            ye += xe * Leg /*.transpose() technically, but Laplacian matrix is symmetric*/;
        }
    }
}

} // namespace fem
} // namespace pbat

#endif // PBAT_FEM_LAPLACIAN_MATRIX_H
