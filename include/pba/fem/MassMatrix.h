#ifndef PBA_CORE_FEM_MASS_MATRIX_H
#define PBA_CORE_FEM_MASS_MATRIX_H

#include "Concepts.h"
#include "pba/common/Eigen.h"
#include "pba/math/SymmetricQuadratureRules.h"

namespace pba {
namespace fem {

template <CMesh TMesh, int Dims>
struct MassMatrix
{
  public:
    using MeshType    = TMesh;
    using ElementType = typename TMesh::ElementType;
    using QuadratureRuleType =
        math::SymmetricPolynomialQuadratureRule<ElementType::kDims, 2 * ElementType::kOrder>;
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
    ComputeElementMassMatrices();
}

template <CMesh TMesh, int Dims>
template <class Derived>
inline MassMatrix<TMesh, Dims>::MassMatrix(
    MeshType const& mesh,
    Eigen::DenseBase<Derived> const& rho)
    : rho(rho), Me(ElementType::kNodes, ElementType::kNodes * mesh.E.cols())
{
    ComputeElementMassMatrices();
}

template <CMesh TMesh, int Dims>
inline void MassMatrix<TMesh, Dims>::ComputeElementMassMatrices(MeshType const& mesh)
{
    Me.setZero();
    // WARNING: We currently do not have quadrature rules for non-simplex elements, i.e. hexahedra
    // and quadrilaterals. The QuadratureRuleType type member should be declared in the element's
    // type, taking only the polynomial order as an argument. This way, the mass matrix can
    // integrate any element using its associated quadrature rule as:
    //
    // using QuadratureRuleType = ElementType::QuadratureRuleType<kOrder>;
    //
    auto const Xg = common::ToEigen(QuadratureRuleType::points)
                        .reshaped(QuadratureRuleType::kDims + 1, QuadratureRuleType::kPoints)
                        .bottomRows(QuadratureRuleType::kDims);
    auto const numberOfElements = mesh.E.cols();
    for (auto e = 0; e < numberOfElements; ++e)
    {
        auto me = Me.block(0, e * ElementType::kNodes, ElementType::kNodes, ElementType::kNodes);
        for (auto g = 0; g < QuadratureRuleType::kPoints; ++g)
        {
            Vector<ElementType::kNodes> const Ng = ElementType::N(Xg.col(g));
            for (auto j = 0; j < ElementType::kNodes; ++j)
            {
                for (auto i = 0; i < ElementType::kNodes; ++i)
                {
                    me(i, j) += rho(e) * Ng(i) * Ng(j);
                }
            }
        }
    }
}

} // namespace fem
} // namespace pba

#endif // PBA_CORE_FEM_MASS_MATRIX_H