#ifndef PBA_CORE_FEM_MASS_MATRIX_H
#define PBA_CORE_FEM_MASS_MATRIX_H

#include "Concepts.h"
#include "Jacobian.h"
#include "pba/aliases.h"
#include "pba/common/Eigen.h"

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
    SparseMatrix ToMatrix() const;

    Index InputDimensions() const { return kDims * mesh.X.cols(); }
    Index OutputDimensions() const { return InputDimensions(); }

    void ComputeElementMassMatrices();

    MeshType const& mesh; ///< The finite element mesh
    VectorX rho;          ///< |#elements| x 1 piecewise constant mass density
    MatrixX Me; ///< |ElementType::Nodes|x|ElementType::Nodes * |#elements| element mass matrices
                ///< for 1-dimensional problems. For d-dimensional problems, these mass matrices
                ///< should be Kroneckered with the d-dimensional identity matrix.
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
    : mesh(mesh), rho(rho), Me(ElementType::kNodes, ElementType::kNodes * mesh.E.cols())
{
    if (rho.size() != mesh.E.cols())
    {
        std::string const what =
            std::format("Expected element-piecewise mass density rho, but size was {}", rho.size());
        throw std::invalid_argument(what);
    }
    ComputeElementMassMatrices();
}

template <CMesh TMesh, int Dims>
template <class TDerivedIn, class TDerivedOut>
inline void MassMatrix<TMesh, Dims>::Apply(
    Eigen::MatrixBase<TDerivedIn> const& x,
    Eigen::DenseBase<TDerivedOut>& y) const
{
    if ((x.rows() != InputDimensions()) or (y.rows() != InputDimensions()) or
        (y.cols() != x.cols()))
    {
        std::string const what = std::format(
            "Expected input x and output y with matching dimensions, but got {}x{} input and {}x{} "
            "output",
            x.rows(),
            x.cols(),
            y.rows(),
            y.cols());
        throw std::invalid_argument(what);
    }

    auto const numberOfElements = mesh.E.cols();
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
    auto const numberOfElements = mesh.E.cols();
    using SparseIndex           = typename SparseMatrix::StorageIndex;
    using Triplet               = Eigen::Triplet<Scalar, SparseIndex>;
    std::vector<Triplet> triplets{};
    triplets.reserve(static_cast<std::size_t>(Me.size() * kDims * kDims));
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
    auto const n = InputDimensions();
    SparseMatrix Mmat(n, n);
    Mmat.setFromTriplets(triplets.begin(), triplets.end());
    return Mmat;
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
    tbb::parallel_for(Index{0}, Index{numberOfElements}, [&](Index e) {
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
            me += (wg(g) * rho(e) * detJ) * (Ng * Ng.transpose());
        }
    });
}

} // namespace fem
} // namespace pba

#endif // PBA_CORE_FEM_MASS_MATRIX_H