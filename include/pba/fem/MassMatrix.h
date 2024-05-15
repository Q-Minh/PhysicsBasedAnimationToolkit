#ifndef PBA_CORE_FEM_MASS_MATRIX_H
#define PBA_CORE_FEM_MASS_MATRIX_H

#include "Concepts.h"
#include "Jacobian.h"
#include "pba/aliases.h"
#include "pba/common/Eigen.h"

#include <functional>
#include <tbb/parallel_for.h>

namespace pba {
namespace fem {

template <CMesh TMesh, int Dims>
struct MassMatrix
{
  public:
    using MeshType              = TMesh;
    using ElementType           = typename TMesh::ElementType;
    using QuadratureRuleType    = ElementType::template QuadratureType<2 * ElementType::kOrder>;
    static int constexpr kDims  = Dims;
    static int constexpr kOrder = 2 * ElementType::kOrder;

    MassMatrix(MeshType const& mesh, Scalar rho = 1.);

    template <class TDerived>
    MassMatrix(MeshType const& mesh, Eigen::DenseBase<TDerived> const& rho);

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
    SparseMatrix ToMatrix() const;

    void ComputeElementMassMatrices();

    std::reference_wrapper<MeshType const> mesh; ///< The finite element mesh
    VectorX rho; ///< |#elements| x 1 piecewise constant mass density
    MatrixX Me;  ///< |ElementType::Nodes|x|ElementType::Nodes * |#elements| element mass matrices
                 ///< for 1-dimensional problems. For d-dimensional problems, these mass matrices
                 ///< should be Kroneckered with the d-dimensional identity matrix.
};

template <CMesh TMesh, int Dims>
inline MassMatrix<TMesh, Dims>::MassMatrix(MeshType const& mesh, Scalar rho)
    : MassMatrix<TMesh, Dims>(mesh, VectorX::Constant(rho, mesh.E.cols()))
{
}

template <CMesh TMesh, int Dims>
template <class TDerived>
inline MassMatrix<TMesh, Dims>::MassMatrix(
    MeshType const& mesh,
    Eigen::DenseBase<TDerived> const& rho)
    : mesh(mesh), rho(rho), Me(ElementType::kNodes, ElementType::kNodes * mesh.E.cols())
{
    ComputeElementMassMatrices();
}

template <CMesh TMesh, int Dims>
template <class TDerivedIn, class TDerivedOut>
inline void MassMatrix<TMesh, Dims>::Apply(
    Eigen::MatrixBase<TDerivedIn> const& x,
    Eigen::DenseBase<TDerivedOut>& y) const
{
    auto const numberOfNodes    = mesh.X.cols();
    auto const numberOfElements = mesh.E.cols();
    assert((Me.cols() / ElementType::kNodes) == numberOfElements);
    auto const n = kDims * numberOfNodes;
    assert(x.rows() == n);
    assert(y.rows() == n);
    assert(y.cols() == x.cols());

    // NOTE: Could parallelize over columns, if there are many.
    for (auto col = 0; col < y.cols(); ++col)
    {
        for (auto e = 0; e < numberOfElements; ++e)
        {
            auto const nodes = mesh.E.col(e).array();
            auto const me =
                Me.block(0, e * ElementType::kNodes, ElementType::kNodes, ElementType::kNodes);
            for (auto d = 0; d < kDims; ++d)
                y(kDims * nodes + d, col) += me * x(kDims * nodes + d, col);
        }
    }
}

template <CMesh TMesh, int Dims>
inline SparseMatrix MassMatrix<TMesh, Dims>::ToMatrix() const
{
    auto const numberOfNodes    = mesh.X.cols();
    auto const numberOfElements = mesh.E.cols();
    assert((Me.cols() / ElementType::kNodes) == numberOfElements);
    auto const n      = kDims * numberOfNodes;
    using SparseIndex = typename SparseMatrix::StorageIndex;
    using Triplet     = Eigen::Triplet<Scalar, SparseIndex>;
    std::vector<Triplet> triplets{};
    triplets.reserve(Me.size() * kDims * kDims);
    for (auto e = 0; e < numberOfElements; ++e)
    {
        auto const nodes = mesh.E.col(e);
        auto const me =
            Me.block(0, e * ElementType::kNodes, ElementType::kNodes, ElementType::kNodes);
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
    SparseMatrix M(n, n);
    M.setFromTriplets(triplets.begin(), triplets.end());
    return M;
}

template <CMesh TMesh, int Dims>
inline void MassMatrix<TMesh, Dims>::ComputeElementMassMatrices()
{
    using AffineElementType = typename ElementType::AffineBaseType;

    Me.setZero();
    auto const Xg = common::ToEigen(QuadratureRuleType::points)
                        .reshaped(QuadratureRuleType::kDims + 1, QuadratureRuleType::kPoints)
                        .bottomRows(QuadratureRuleType::kDims);
    auto const numberOfElements = mesh.E.cols();
    tbb::parallel_for(0, numberOfElements, [&](std::size_t e) {
        auto const nodes                = mesh.E.col(e);
        auto const vertices             = nodes(ElementType::Vertices);
        auto constexpr kRowsJ           = MeshType::kDims;
        auto constexpr kColsJ           = AffineElementType::kNodes;
        Matrix<kRowsJ, kColsJ> const Ve = mesh.X(Eigen::all, vertices);
        auto me = Me.block(0, e * ElementType::kNodes, ElementType::kNodes, ElementType::kNodes);
        Scalar detJ{};
        if constexpr (AffineElementType::bHasConstantJacobian)
            detJ = DeterminantOfJacobian(Jacobian<AffineElementType>({}, Ve));

        auto const wg = common::ToEigen(QuadratureRuleType::weights);
        for (auto g = 0; g < QuadratureRuleType::kPoints; ++g)
        {
            if constexpr (!AffineElementType::bHasConstantJacobian)
                detJ = DeterminantOfJacobian(Jacobian<AffineElementType>(Xg.col(g), Ve));

            Vector<ElementType::kNodes> const Ng = ElementType::N(Xg.col(g));
            for (auto j = 0; j < me.cols(); ++j)
            {
                for (auto i = 0; i < me.rows(); ++i)
                {
                    me(i, j) += wg(g) * rho(e) * Ng(i) * Ng(j) * detJ;
                }
            }
        }
    });
}

} // namespace fem
} // namespace pba

#endif // PBA_CORE_FEM_MASS_MATRIX_H