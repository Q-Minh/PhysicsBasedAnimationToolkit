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

template <CMesh TMesh>
struct SymmetricLaplacianMatrix
{
  public:
    using SelfType    = SymmetricLaplacianMatrix<TMesh>;
    using MeshType    = TMesh;
    using ElementType = typename TMesh::ElementType;

    static int constexpr kOrder = 2 * (ElementType::kOrder - 1);

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
     * @tparam TDerivedIn
     * @tparam TDerivedOut
     * @param x
     * @param y
     */
    template <class TDerivedIn, class TDerivedOut>
    void Apply(Eigen::MatrixBase<TDerivedIn> const& x, Eigen::DenseBase<TDerivedOut>& y) const;

    /**
     * @brief Transforms this matrix-free matrix representation into sparse compressed format.
     * @return
     */
    CSCMatrix ToMatrix() const;

    Index InputDimensions() const { return dims * mesh.X.cols(); }
    Index OutputDimensions() const { return InputDimensions(); }

    void ComputeElementLaplacians();

    void CheckValidState() const;

    MeshType const& mesh; ///< The finite element mesh
    Eigen::Ref<IndexVectorX const>
        eg; ///< |#quad.pts.|x1 array of elements associated with quadrature points
    Eigen::Ref<VectorX const> wg; ///< |#quad.pts.|x1 array of quadrature weights
    Eigen::Ref<MatrixX const>
        GNeg;       ///< |#element nodes|x|#dims * #quad.pts. * #elements|
                    ///< matrix of element shape function gradients at quadrature points
    MatrixX deltag; ///< |#element nodes| x |#element nodes * #quad.pts.| matrix of element
                    ///< laplacians at quadrature points
    int dims; ///< Dimensionality of image of FEM function space, i.e. this Laplacian matrix is
              ///< actually L \kronecker I_{dims \times dims}. Should be >= 1.
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
        auto const e = eg(g);
        auto Leg     = deltag.block<kNodesPerElement, kNodesPerElement>(0, g * kNodesPerElement);
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
