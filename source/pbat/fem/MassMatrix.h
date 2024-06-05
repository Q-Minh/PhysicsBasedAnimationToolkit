#ifndef PBAT_FEM_MASS_MATRIX_H
#define PBAT_FEM_MASS_MATRIX_H

#include "Concepts.h"
#include "ShapeFunctions.h"
#include "pbat/aliases.h"
#include "pbat/common/Eigen.h"
#include "pbat/profiling/Profiling.h"

#include <array>
#include <exception>
#include <format>
#include <tbb/parallel_for.h>

namespace pbat {
namespace fem {

template <CMesh TMesh, int Dims, int QuadratureOrder>
struct MassMatrix
{
  public:
    using SelfType                        = MassMatrix<TMesh, Dims, QuadratureOrder>;
    using MeshType                        = TMesh;
    using ElementType                     = typename TMesh::ElementType;
    using QuadratureRuleType              = ElementType::template QuadratureType<QuadratureOrder>;
    static int constexpr kDims            = Dims;
    static int constexpr kOrder           = 2 * ElementType::kOrder;
    static int constexpr kQuadratureOrder = QuadratureOrder;

    /**
     * @brief
     * @param mesh
     * @param detJe |#quad.pts.|x|#elements| affine element jacobian determinants at quadrature
     * points
     * @param rho Uniform mass density
     */
    MassMatrix(MeshType const& mesh, Eigen::Ref<MatrixX const> const& detJe, Scalar rho = 1.);

    /**
     * @brief
     * @tparam TDerived
     * @param mesh
     * @param detJe |#quad.pts.|x|#elements| affine element jacobian determinants at quadrature
     * points
     * @param rho |#elements| x 1 piecewise constant mass density
     */
    template <class TDerived>
    MassMatrix(
        MeshType const& mesh,
        Eigen::Ref<MatrixX const> const& detJe,
        Eigen::DenseBase<TDerived> const& rho);

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

    Index InputDimensions() const { return kDims * mesh.X.cols(); }
    Index OutputDimensions() const { return InputDimensions(); }

    /**
     * @brief
     * @tparam TDerived
     * @param rho |#elements| x 1 piecewise constant mass density
     */
    template <class TDerived>
    void ComputeElementMassMatrices(Eigen::DenseBase<TDerived> const& rho);

    void CheckValidState();

    MeshType const& mesh;            ///< The finite element mesh
    Eigen::Ref<MatrixX const> detJe; ///< |# element quadrature points| x |# elements| matrix of
                                     ///< jacobian determinants at element quadrature points
    MatrixX Me; ///< |#element nodes|x|#element nodes * #elements| element mass matrices
                ///< for 1-dimensional problems. For d-dimensional problems, these mass matrices
                ///< should be Kroneckered with the d-dimensional identity matrix.
};

template <CMesh TMesh, int Dims, int QuadratureOrder>
inline MassMatrix<TMesh, Dims, QuadratureOrder>::MassMatrix(
    MeshType const& mesh,
    Eigen::Ref<MatrixX const> const& detJe,
    Scalar rho)
    : MassMatrix<TMesh, Dims, QuadratureOrder>(mesh, detJe, VectorX::Constant(mesh.E.cols(), rho))
{
}

template <CMesh TMesh, int Dims, int QuadratureOrder>
template <class TDerived>
inline MassMatrix<TMesh, Dims, QuadratureOrder>::MassMatrix(
    MeshType const& mesh,
    Eigen::Ref<MatrixX const> const& detJe,
    Eigen::DenseBase<TDerived> const& rho)
    : mesh(mesh), detJe(detJe), Me()
{
    ComputeElementMassMatrices(rho);
}

template <CMesh TMesh, int Dims, int QuadratureOrder>
template <class TDerivedIn, class TDerivedOut>
inline void MassMatrix<TMesh, Dims, QuadratureOrder>::Apply(
    Eigen::MatrixBase<TDerivedIn> const& x,
    Eigen::DenseBase<TDerivedOut>& y) const
{
    PBA_PROFILE_SCOPE;
    auto const numberOfDofs = InputDimensions();
    if (x.rows() != numberOfDofs or y.rows() != numberOfDofs or x.cols() != y.cols())
    {
        std::string const what = std::format(
            "Expected inputs and outputs to have rows |#nodes*kDims|={} and same number of "
            "columns, but got dimensions "
            "x,y=({},{}), ({},{})",
            numberOfDofs,
            x.rows(),
            x.cols(),
            y.rows(),
            y.cols());
        throw std::invalid_argument(what);
    }

    auto const numberOfElements = mesh.E.cols();
    // NOTE: Could parallelize over columns, if there are many.
    for (auto c = 0; c < y.cols(); ++c)
    {
        for (auto e = 0; e < numberOfElements; ++e)
        {
            auto const nodes = mesh.E.col(e).array();
            auto const me =
                Me.block<ElementType::kNodes, ElementType::kNodes>(0, e * ElementType::kNodes);
            auto ye       = y.col(c).reshaped(kDims, y.size() / kDims)(Eigen::all, nodes);
            auto const xe = x.col(c).reshaped(kDims, x.size() / kDims)(Eigen::all, nodes);
            ye += xe * me /*.transpose() technically, but mass matrix is symmetric*/;
        }
    }
}

template <CMesh TMesh, int Dims, int QuadratureOrder>
inline CSCMatrix MassMatrix<TMesh, Dims, QuadratureOrder>::ToMatrix() const
{
    PBA_PROFILE_SCOPE;
    using SparseIndex = typename CSCMatrix::StorageIndex;
    using Triplet     = Eigen::Triplet<Scalar, SparseIndex>;

    std::vector<Triplet> triplets{};
    triplets.reserve(static_cast<std::size_t>(Me.size() * kDims * kDims));
    auto const numberOfElements = mesh.E.cols();
    for (auto e = 0; e < numberOfElements; ++e)
    {
        auto const nodes = mesh.E.col(e);
        auto const me =
            Me.block<ElementType::kNodes, ElementType::kNodes>(0, e * ElementType::kNodes);
        for (auto j = 0; j < me.cols(); ++j)
        {
            for (auto i = 0; i < me.rows(); ++i)
            {
                for (auto d = 0; d < kDims; ++d)
                {
                    auto const ni = static_cast<SparseIndex>(kDims * nodes(i) + d);
                    auto const nj = static_cast<SparseIndex>(kDims * nodes(j) + d);
                    triplets.push_back(Triplet(ni, nj, me(i, j)));
                }
            }
        }
    }

    CSCMatrix Mmat(OutputDimensions(), InputDimensions());
    Mmat.setFromTriplets(triplets.begin(), triplets.end());
    return Mmat;
}

template <CMesh TMesh, int Dims, int QuadratureOrder>
inline void MassMatrix<TMesh, Dims, QuadratureOrder>::CheckValidState()
{
    auto const numberOfElements       = mesh.E.cols();
    auto constexpr kExpectedDetJeRows = QuadratureRuleType::kPoints;
    auto const expectedDetJeCols      = numberOfElements;
    bool const bDeterminantsHaveCorrectDimensions =
        (detJe.rows() == kExpectedDetJeRows) and (detJe.cols() == expectedDetJeCols);
    if (not bDeterminantsHaveCorrectDimensions)
    {
        std::string const what = std::format(
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
}

template <CMesh TMesh, int Dims, int QuadratureOrder>
template <class TDerived>
inline void MassMatrix<TMesh, Dims, QuadratureOrder>::ComputeElementMassMatrices(
    Eigen::DenseBase<TDerived> const& rho)
{
    PBA_PROFILE_SCOPE;
    // Check inputs before proceeding
    CheckValidState();
    auto const numberOfElements       = mesh.E.cols();
    auto constexpr kNodesPerElement   = ElementType::kNodes;
    auto constexpr kQuadPtsPerElement = QuadratureRuleType::kPoints;
    bool const bRhoDimensionsAreCorrect =
        (rho.size() == numberOfElements) and ((rho.rows() == 1) or (rho.cols() == 1));
    if (not bRhoDimensionsAreCorrect)
    {
        std::string const what = std::format(
            "Expected element-piecewise mass density rho of dimensions {}x1, but dimensions were "
            "{}x{}",
            numberOfElements,
            rho.rows(),
            rho.cols());
        throw std::invalid_argument(what);
    }
    // Precompute element shape function outer products
    auto const N = ShapeFunctions<ElementType, kQuadratureOrder>();
    std::array<Matrix<kNodesPerElement, kNodesPerElement>, kQuadPtsPerElement> NgOuterNg{};
    auto const wg = common::ToEigen(QuadratureRuleType::weights);
    for (auto g = 0; g < kQuadPtsPerElement; ++g)
    {
        NgOuterNg[static_cast<std::size_t>(g)] = wg(g) * (N.col(g) * N.col(g).transpose());
    }
    // Compute element mass matrices
    Me.setZero(kNodesPerElement, kNodesPerElement * numberOfElements);
    tbb::parallel_for(Index{0}, Index{numberOfElements}, [&](Index e) {
        auto me = Me.block<kNodesPerElement, kNodesPerElement>(0, e * kNodesPerElement);
        for (auto g = 0; g < kQuadPtsPerElement; ++g)
        {
            me += (rho(e) * detJe(g, e)) * NgOuterNg[static_cast<std::size_t>(g)];
        }
    });
}

} // namespace fem
} // namespace pbat

#endif // PBAT_FEM_MASS_MATRIX_H