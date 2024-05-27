#ifndef PBA_CORE_FEM_MASS_MATRIX_H
#define PBA_CORE_FEM_MASS_MATRIX_H

#include "Concepts.h"
#include "Jacobian.h"
#include "pba/aliases.h"
#include "pba/common/Eigen.h"
#include "pba/common/Profiling.h"

#include <exception>
#include <format>
#include <tbb/parallel_for.h>

namespace pba {
namespace fem {

template <CMesh TMesh, int Dims>
struct MassMatrix
{
  public:
    using SelfType              = MassMatrix<TMesh, Dims>;
    using MeshType              = TMesh;
    using ElementType           = typename TMesh::ElementType;
    using QuadratureRuleType    = ElementType::template QuadratureType<2 * ElementType::kOrder>;
    static int constexpr kDims  = Dims;
    static int constexpr kOrder = 2 * ElementType::kOrder;

    MassMatrix(MeshType const& mesh, Scalar rho = 1.);

    template <class TDerived>
    MassMatrix(MeshType const& mesh, Eigen::DenseBase<TDerived> const& rho);

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

    void ComputeElementMassMatrices();
    void PrecomputeJacobianDeterminants();

    MeshType const& mesh; ///< The finite element mesh
    VectorX rho;          ///< |#elements| x 1 piecewise constant mass density
    MatrixX Me;    ///< |ElementType::Nodes|x|ElementType::Nodes * |#elements| element mass matrices
                   ///< for 1-dimensional problems. For d-dimensional problems, these mass matrices
                   ///< should be Kroneckered with the d-dimensional identity matrix.
    MatrixX detJe; ///< |# element quadrature points| x |# elements| matrix of jacobian determinants
                   ///< at element quadrature points
};

template <CMesh TMesh, int Dims>
inline MassMatrix<TMesh, Dims>::MassMatrix(MeshType const& mesh, Scalar rho)
    : MassMatrix<TMesh, Dims>(mesh, VectorX::Constant(mesh.E.cols(), rho))
{
}

template <CMesh TMesh, int Dims>
template <class TDerived>
inline MassMatrix<TMesh, Dims>::MassMatrix(
    MeshType const& mesh,
    Eigen::DenseBase<TDerived> const& rho)
    : mesh(mesh), rho(rho), Me()
{
    PBA_PROFILE_NAMED_SCOPE("Construct fem::MassMatrix");
    if (rho.size() != mesh.E.cols())
    {
        std::string const what =
            std::format("Expected element-piecewise mass density rho, but size was {}", rho.size());
        throw std::invalid_argument(what);
    }
    PrecomputeJacobianDeterminants();
    ComputeElementMassMatrices();
}

template <CMesh TMesh, int Dims>
template <class TDerivedIn, class TDerivedOut>
inline void MassMatrix<TMesh, Dims>::Apply(
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

template <CMesh TMesh, int Dims>
inline CSCMatrix MassMatrix<TMesh, Dims>::ToMatrix() const
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

    auto const n = InputDimensions();
    CSCMatrix Mmat(n, n);
    Mmat.setFromTriplets(triplets.begin(), triplets.end());
    return Mmat;
}

template <CMesh TMesh, int Dims>
inline void MassMatrix<TMesh, Dims>::ComputeElementMassMatrices()
{
    PBA_PROFILE_SCOPE;
    using AffineElementType = typename ElementType::AffineBaseType;

    auto const Xg = common::ToEigen(QuadratureRuleType::points)
                        .reshaped(QuadratureRuleType::kDims + 1, QuadratureRuleType::kPoints)
                        .bottomRows<QuadratureRuleType::kDims>();
    auto const wg                   = common::ToEigen(QuadratureRuleType::weights);
    auto const numberOfElements     = mesh.E.cols();
    auto constexpr kNodesPerElement = ElementType::kNodes;
    Me.setZero(kNodesPerElement, kNodesPerElement * numberOfElements);
    tbb::parallel_for(Index{0}, Index{numberOfElements}, [&](Index e) {
        auto const nodes    = mesh.E.col(e);
        auto const vertices = nodes(ElementType::Vertices);
        auto me = Me.block<ElementType::kNodes, ElementType::kNodes>(0, e * ElementType::kNodes);
        for (auto g = 0; g < QuadratureRuleType::kPoints; ++g)
        {
            Scalar const detJ                    = detJe(g, e);
            Vector<ElementType::kNodes> const Ng = ElementType::N(Xg.col(g));
            me += (wg(g) * rho(e) * detJ) * (Ng * Ng.transpose());
        }
    });
}

template <CMesh TMesh, int Dims>
inline void MassMatrix<TMesh, Dims>::PrecomputeJacobianDeterminants()
{
    detJe = DeterminantOfJacobian<kOrder>(mesh);
}

} // namespace fem
} // namespace pba

#endif // PBA_CORE_FEM_MASS_MATRIX_H