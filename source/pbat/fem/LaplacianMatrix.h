#ifndef PBAT_FEM_LAPLACIAN_MATRIX_H
#define PBAT_FEM_LAPLACIAN_MATRIX_H

#include "Concepts.h"
#include "Jacobian.h"
#include "pbat/aliases.h"
#include "pbat/common/Eigen.h"
#include "pbat/profiling/Profiling.h"
#include "pbat/fem/ShapeFunctionGradients.h"

#include <exception>
#include <format>
#include <tbb/parallel_for.h>

namespace pbat {
namespace fem {

template <CMesh TMesh>
struct SymmetricLaplacianMatrix
{
  public:
    using SelfType              = SymmetricLaplacianMatrix<TMesh>;
    using MeshType              = TMesh;
    using ElementType           = typename TMesh::ElementType;
    static int constexpr kOrder = 2 * (ElementType::kOrder - 1);
    static int constexpr kDims  = 1;

    template <int ShapeFunctionOrder>
    struct OrderSelector
    {
        // The symmetric part of the Laplacian matrix's element i,j is
        // -\int_{\Omega} \nabla \phi_i(X) \cdot \nabla \phi_j(X) \partial \Omega.
        static auto constexpr kOrder = 2 * (ShapeFunctionOrder - 1);
    };

    template <>
    struct OrderSelector<1>
    {
        // For linear basis functions, the Laplacian vanishes. The integrand is order 0, but
        // there is no order 0 quadrature rule, so we default to an order 1 quadrature rule, which
        // will simply pick 1 point (and the point will actually not matter in computations).
        static auto constexpr kOrder = 1;
    };

    using QuadratureRuleType =
        ElementType::template QuadratureType<OrderSelector<ElementType::kOrder>::kOrder>;

    SymmetricLaplacianMatrix(MeshType const& mesh);

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

    Index InputDimensions() const { return mesh.X.cols(); }
    Index OutputDimensions() const { return InputDimensions(); }

    void ComputeElementLaplacians();

    MeshType const& mesh; ///< The finite element mesh
    MatrixX deltaE; ///< |ElementType::kNodes| x |ElementType::kNodes * #elements| matrix element
                    ///< laplacians
};

template <CMesh TMesh>
inline SymmetricLaplacianMatrix<TMesh>::SymmetricLaplacianMatrix(MeshType const& mesh) : mesh(mesh)
{
    PBA_PROFILE_NAMED_SCOPE("Construct fem::SymmetricLaplacianMatrix");
    ComputeElementLaplacians();
}

template <CMesh TMesh>
inline CSCMatrix SymmetricLaplacianMatrix<TMesh>::ToMatrix() const
{
    PBA_PROFILE_SCOPE;
    CSCMatrix L(OutputDimensions(), InputDimensions());
    using SparseIndex = typename CSCMatrix::StorageIndex;
    using Triplet     = Eigen::Triplet<Scalar, SparseIndex>;

    std::vector<Triplet> triplets{};
    triplets.reserve(static_cast<std::size_t>(deltaE.size()));
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
                auto const ni = static_cast<SparseIndex>(nodes(i));
                auto const nj = static_cast<SparseIndex>(nodes(j));
                triplets.push_back(Triplet(ni, nj, Le(i, j)));
            }
        }
    }
    L.setFromTriplets(triplets.begin(), triplets.end());
    return L;
}

template <CMesh TMesh>
inline void SymmetricLaplacianMatrix<TMesh>::ComputeElementLaplacians()
{
    PBA_PROFILE_SCOPE;
    using AffineElementType = typename ElementType::AffineBaseType;

    auto const Xg = common::ToEigen(QuadratureRuleType::points)
                        .reshaped(QuadratureRuleType::kDims + 1, QuadratureRuleType::kPoints)
                        .bottomRows<QuadratureRuleType::kDims>();
    auto const wg                   = common::ToEigen(QuadratureRuleType::weights);
    auto constexpr kNodesPerElement = ElementType::kNodes;
    MatrixX const detJe             = DeterminantOfJacobian<QuadratureRuleType::kOrder>(mesh);
    MatrixX const GNe               = ShapeFunctionGradients<QuadratureRuleType::kOrder>(mesh);
    auto const numberOfElements     = mesh.E.cols();
    deltaE.setZero(kNodesPerElement, kNodesPerElement * numberOfElements);
    tbb::parallel_for(Index{0}, Index{numberOfElements}, [&](Index e) {
        auto const nodes    = mesh.E.col(e);
        auto const vertices = nodes(ElementType::Vertices);
        auto const Ve       = mesh.X(Eigen::all, vertices);
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
            Le -= (wg(g) * detJe(g, e)) * GP * GP.transpose();
        }
    });
}

template <CMesh TMesh>
template <class TDerivedIn, class TDerivedOut>
inline void SymmetricLaplacianMatrix<TMesh>::Apply(
    Eigen::MatrixBase<TDerivedIn> const& x,
    Eigen::DenseBase<TDerivedOut>& y) const
{
    PBA_PROFILE_SCOPE;
    auto const numberOfDofs = InputDimensions();
    if ((x.rows() != numberOfDofs) or (y.rows() != numberOfDofs) or (y.cols() != x.cols()))
    {
        std::string const what = std::format(
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
            auto ye       = y.col(c)(nodes);
            auto const xe = x.col(c)(nodes);
            ye += Le * xe;
        }
    }
}

} // namespace fem
} // namespace pbat

#endif // PBAT_FEM_LAPLACIAN_MATRIX_H
