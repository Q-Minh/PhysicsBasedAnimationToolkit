#ifndef PBA_CORE_FEM_MASS_MATRIX_H
#define PBA_CORE_FEM_MASS_MATRIX_H

#include "Concepts.h"
#include "Jacobian.h"
#include "pba/common/Eigen.h"
#include "pba/math/SymmetricQuadratureRules.h"

#include <Eigen/SVD>
#include <cmath>
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

    template <class Derived>
    MassMatrix(MeshType const& mesh, Eigen::DenseBase<Derived> const& rho);

    void ComputeElementMassMatrices(MeshType const& mesh);

    VectorX rho; ///< |#elements| x 1 piecewise constant mass density
    MatrixX Me;  ///< |ElementType::Nodes|x|ElementType::Nodes * |#elements| element mass matrices
};

template <CMesh TMesh, int Dims>
inline MassMatrix<TMesh, Dims>::MassMatrix(MeshType const& mesh, Scalar rho)
    : rho(VectorX::Constant(rho, mesh.E.cols())),
      Me(ElementType::kNodes, ElementType::kNodes * mesh.E.cols())
{
    ComputeElementMassMatrices(mesh);
}

template <CMesh TMesh, int Dims>
template <class Derived>
inline MassMatrix<TMesh, Dims>::MassMatrix(
    MeshType const& mesh,
    Eigen::DenseBase<Derived> const& rho)
    : rho(rho), Me(ElementType::kNodes, ElementType::kNodes * mesh.E.cols())
{
    ComputeElementMassMatrices(mesh);
}

template <CMesh TMesh, int Dims>
inline void MassMatrix<TMesh, Dims>::ComputeElementMassMatrices(MeshType const& mesh)
{
    using AffineElement = typename ElementType::AffineBaseType;

    Me.setZero();
    auto const Xg = common::ToEigen(QuadratureRuleType::points)
                        .reshaped(QuadratureRuleType::kDims + 1, QuadratureRuleType::kPoints)
                        .bottomRows(QuadratureRuleType::kDims);
    auto const numberOfElements = mesh.E.cols();
    tbb::parallel_for(0, numberOfElements, [&](std::size_t e) {
        auto const nodes    = mesh.E.col(e);
        auto const vertices = nodes(ElementType::Vertices);
        auto const Ve       = mesh.X(Eigen::all, vertices);
        Scalar const detJ   = DeterminantOfJacobian(Jacobian(
            Xg.col(0) /*Any point will do, since jacobian of affine map is constant*/,
            Ve));
        auto const wg       = detJ * common::ToEigen(QuadratureRuleType::weights);
        auto me = Me.block(0, e * ElementType::kNodes, ElementType::kNodes, ElementType::kNodes);
        for (auto g = 0; g < QuadratureRuleType::kPoints; ++g)
        {
            Vector<ElementType::kNodes> const Ng = ElementType::N(Xg.col(g));
            for (auto j = 0; j < ElementType::kNodes; ++j)
            {
                for (auto i = 0; i < ElementType::kNodes; ++i)
                {
                    me(i, j) += wg(g) * rho(e) * Ng(i) * Ng(j);
                }
            }
        }
    });
}

} // namespace fem
} // namespace pba

#endif // PBA_CORE_FEM_MASS_MATRIX_H