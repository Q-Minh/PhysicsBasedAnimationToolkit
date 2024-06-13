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

template <CMesh TMesh, int QuadratureOrder>
struct SymmetricLaplacianMatrix
{
  public:
    using SelfType           = SymmetricLaplacianMatrix<TMesh, QuadratureOrder>;
    using MeshType           = TMesh;
    using ElementType        = typename TMesh::ElementType;
    using QuadratureRuleType = typename ElementType::template QuadratureType<QuadratureOrder>;

    static int constexpr kOrder           = 2 * (ElementType::kOrder - 1);
    static int constexpr kQuadratureOrder = QuadratureOrder;

    SymmetricLaplacianMatrix(
        MeshType const& mesh,
        Eigen::Ref<MatrixX const> const& detJe,
        Eigen::Ref<MatrixX const> const& GNe,
        int dims = 1);

    SelfType& operator=(SelfType const&) = delete;

    /**
     * @brief Applies this mass matrix as a linear operator on x, adding result to y.
     *
     * @tparam TDerivedIn
     * @tparam TDerivedOut
     * @param x
     * @param y
     */
    template <class TDerivedIn, class TDerivedOut>
    void Apply(Eigen::MatrixBase<TDerivedIn> const& x, Eigen::DenseBase<TDerivedOut>& y) const;

    /**
     * @brief Transforms this matrix-free mass matrix representation into sparse compressed format.
     * @return
     */
    CSCMatrix ToMatrix() const;

    Index InputDimensions() const { return dims * mesh.X.cols(); }
    Index OutputDimensions() const { return InputDimensions(); }

    void ComputeElementLaplacians();

    void CheckValidState() const;

    MeshType const& mesh;            ///< The finite element mesh
    Eigen::Ref<MatrixX const> detJe; ///< |#quad.pts.|x|#elements| affine element jacobian
                                     ///< determinants at quadrature points
    Eigen::Ref<MatrixX const>
        GNe;        ///< |ElementType::kNodes|x|kDims * QuadratureRuleType::kPoints * #elements|
                    ///< matrix of element shape function gradients at quadrature points
    MatrixX deltaE; ///< |ElementType::kNodes| x |ElementType::kNodes * #elements| matrix element
                    ///< laplacians
    int dims; ///< Dimensionality of image of FEM function space, i.e. this Laplacian matrix is
              ///< actually L \kronecker I_{dims \times dims}. Should be >= 1.
};

template <CMesh TMesh, int QuadratureOrder>
inline SymmetricLaplacianMatrix<TMesh, QuadratureOrder>::SymmetricLaplacianMatrix(
    MeshType const& mesh,
    Eigen::Ref<MatrixX const> const& detJe,
    Eigen::Ref<MatrixX const> const& GNe,
    int dims)
    : mesh(mesh), detJe(detJe), GNe(GNe), deltaE(), dims(dims)
{
    ComputeElementLaplacians();
}

template <CMesh TMesh, int QuadratureOrder>
inline CSCMatrix SymmetricLaplacianMatrix<TMesh, QuadratureOrder>::ToMatrix() const
{
    PBA_PROFILE_SCOPE;
    CheckValidState();
    CSCMatrix L(OutputDimensions(), InputDimensions());
    using SparseIndex = typename CSCMatrix::StorageIndex;
    using Triplet     = Eigen::Triplet<Scalar, SparseIndex>;

    std::vector<Triplet> triplets{};
    triplets.reserve(static_cast<std::size_t>(deltaE.size() * dims));
    auto const numberOfElements = mesh.E.cols();
    for (auto e = 0; e < numberOfElements; ++e)
    {
        auto const nodes                = mesh.E.col(e);
        auto constexpr kNodesPerElement = ElementType::kNodes;
        auto const Le = deltaE.block(0, e * kNodesPerElement, kNodesPerElement, kNodesPerElement);
        for (auto j = 0; j < Le.cols(); ++j)
        {
            for (auto i = 0; i < Le.rows(); ++i)
            {
                for (auto d = 0; d < dims; ++d)
                {
                    auto const ni = static_cast<SparseIndex>(dims * nodes(i) + d);
                    auto const nj = static_cast<SparseIndex>(dims * nodes(j) + d);
                    triplets.push_back(Triplet(ni, nj, Le(i, j)));
                }
            }
        }
    }
    L.setFromTriplets(triplets.begin(), triplets.end());
    return L;
}

template <CMesh TMesh, int QuadratureOrder>
inline void SymmetricLaplacianMatrix<TMesh, QuadratureOrder>::ComputeElementLaplacians()
{
    PBA_PROFILE_SCOPE;
    CheckValidState();
    // Compute element laplacians
    auto const wg                   = common::ToEigen(QuadratureRuleType::weights);
    auto constexpr kNodesPerElement = ElementType::kNodes;
    auto const numberOfElements     = mesh.E.cols();
    deltaE.setZero(kNodesPerElement, kNodesPerElement * numberOfElements);
    tbb::parallel_for(Index{0}, Index{numberOfElements}, [&](Index e) {
        auto Le = deltaE.block(0, e * kNodesPerElement, kNodesPerElement, kNodesPerElement);
        for (auto g = 0; g < QuadratureRuleType::kPoints; ++g)
        {
            // Use multivariable integration by parts (i.e. Green's identity), and retain only the
            // symmetric part, i.e.
            // Lij = -\int_{\Omega} \nabla \phi_i(X) \cdot \nabla \phi_j(X) \partial \Omega.
            // Matrix<kNodesPerElement, MeshType::kDims> const GP =
            //    ShapeFunctionGradients<ElementType>(Xg.col(g), Ve);
            auto const kStride = MeshType::kDims * QuadratureRuleType::kPoints;
            auto const GP =
                GNe.block<kNodesPerElement, MeshType::kDims>(0, e * kStride + g * MeshType::kDims);
            Le -= (wg(g) * detJe(g, e)) * (GP * GP.transpose());
        }
    });
}

template <CMesh TMesh, int QuadratureOrder>
inline void SymmetricLaplacianMatrix<TMesh, QuadratureOrder>::CheckValidState() const
{
    auto const numberOfElements       = mesh.E.cols();
    auto constexpr kExpectedDetJeRows = QuadratureRuleType::kPoints;
    auto const expectedDetJeCols      = numberOfElements;
    bool const bDeterminantsHaveCorrectDimensions =
        (detJe.rows() == kExpectedDetJeRows) and (detJe.cols() == expectedDetJeCols);
    if (not bDeterminantsHaveCorrectDimensions)
    {
        std::string const what = fmt::format(
            "Expected determinants at element quadrature points of dimensions #quad.pts.={} x "
            "#elements={} for polynomial "
            "quadrature order={}, but got {}x{} instead.",
            kExpectedDetJeRows,
            expectedDetJeCols,
            QuadratureOrder,
            detJe.rows(),
            detJe.cols());
        throw std::invalid_argument(what);
    }
    auto constexpr kExpectedGNeRows = ElementType::kNodes;
    auto const expectedGNeCols = MeshType::kDims * QuadratureRuleType::kPoints * numberOfElements;
    bool const bShapeFunctionGradientsHaveCorrectDimensions =
        (GNe.rows() == kExpectedGNeRows) and (GNe.cols() == expectedGNeCols);
    if (not bShapeFunctionGradientsHaveCorrectDimensions)
    {
        std::string const what = fmt::format(
            "Expected shape function gradients at element quadrature points of dimensions "
            "|#nodes-per-element|={} x |#mesh-dims * #quad.pts. * #elemens|={} for polynomiail "
            "quadrature order={}, but got {}x{} instead",
            kExpectedGNeRows,
            expectedGNeCols,
            QuadratureOrder,
            GNe.rows(),
            GNe.cols());
        throw std::invalid_argument(what);
    }
    if (dims < 1)
    {
        std::string const what =
            fmt::format("Expected output dimensionality >= 1, got {} instead", dims);
        throw std::invalid_argument(what);
    }
}

template <CMesh TMesh, int QuadratureOrder>
template <class TDerivedIn, class TDerivedOut>
inline void SymmetricLaplacianMatrix<TMesh, QuadratureOrder>::Apply(
    Eigen::MatrixBase<TDerivedIn> const& x,
    Eigen::DenseBase<TDerivedOut>& y) const
{
    PBA_PROFILE_SCOPE;
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

    auto const numberOfElements = mesh.E.cols();
    for (auto c = 0; c < x.cols(); ++c)
    {
        for (auto e = 0; e < numberOfElements; ++e)
        {
            auto const nodes                = mesh.E.col(e);
            auto constexpr kNodesPerElement = ElementType::kNodes;
            auto const Le =
                deltaE.block(0, e * kNodesPerElement, kNodesPerElement, kNodesPerElement);
            auto ye       = y.col(c).reshaped(dims, y.size() / dims)(Eigen::all, nodes);
            auto const xe = x.col(c).reshaped(dims, x.size() / dims)(Eigen::all, nodes);
            ye += xe * Le /*.transpose() technically, but Laplacian matrix is symmetric*/;
        }
    }
}

} // namespace fem
} // namespace pbat

#endif // PBAT_FEM_LAPLACIAN_MATRIX_H
