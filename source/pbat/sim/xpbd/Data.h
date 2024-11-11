#ifndef PBAT_SIM_XPBD_DATA_H
#define PBAT_SIM_XPBD_DATA_H

#include "Enums.h"
#include "PhysicsBasedAnimationToolkitExport.h"
#include "pbat/Aliases.h"

#include <array>

namespace pbat {
namespace sim {
namespace xpbd {

PBAT_API struct Data
{
  public:
    Data&
    WithVolumeMesh(Eigen::Ref<MatrixX const> const& V, Eigen::Ref<IndexMatrixX const> const& E);
    Data& WithSurfaceMesh(
        Eigen::Ref<IndexVectorX const> const& V,
        Eigen::Ref<IndexMatrixX const> const& F);
    Data& WithBodies(Eigen::Ref<IndexVectorX const> const& BV);
    Data& WithVelocity(Eigen::Ref<MatrixX const> const& v);
    Data& WithAcceleration(Eigen::Ref<MatrixX const> const& aext);
    Data& WithMassInverse(Eigen::Ref<VectorX const> const& minv);
    Data& WithElasticMaterial(Eigen::Ref<MatrixX const> const& lame);
    Data& WithFrictionCoefficients(Scalar muS, Scalar muD);
    Data& WithCompliance(Eigen::Ref<VectorX> const& alpha, EConstraint constraint);
    Data& WithPartitions(std::vector<std::vector<Index>> const& partitions);
    Data& WithDirichletConstrainedVertices(IndexVectorX const& dbc);
    Data& Construct(bool bValidate = true);

  public:
    IndexVectorX V; ///< |#collision vertices| array of indices into columns of x
    IndexMatrixX F; ///< 3x|#triangles| array of collision triangles (on the boundary of T),
                    ///< indexing into columns of x
    IndexMatrixX T; ///< Tetrahedra

    IndexVectorX BV; ///< |#particles| array of body indices

    MatrixX x;    ///< Vertex positions
    MatrixX v;    ///< Vertex velocities
    MatrixX aext; ///< Vertex external accelerations
    VectorX minv; ///< Vertex mass inverses

    MatrixX xt; ///< Vertex positions at time t

    MatrixX lame;     ///< 2x|#quad.pts.| Lame coefficients
    MatrixX DmInv;    ///< 3x3x|#elements| array of material shape matrix inverses
    VectorX gammaSNH; ///< 1. + mu/lambda, where mu,lambda are Lame coefficients

    Scalar alphaC{0.}; ///< Collision compliance
    Scalar muS{0.3};   ///< Static friction coefficient
    Scalar muD{0.2};   ///< Dynamic friction coefficient

    std::array<VectorX, static_cast<int>(EConstraint::NumberOfConstraintTypes)>
        alpha; ///< Compliance
               ///< alpha[0] -> Stable Neo-Hookean constraint compliance
               ///< alpha[1] -> Collision penalty constraint compliance
    std::array<VectorX, static_cast<int>(EConstraint::NumberOfConstraintTypes)>
        lambda; ///< "Lagrange" multipliers:
                ///< lambda[0] -> Stable Neo-Hookean constraint multipliers
                ///< lambda[1] -> Collision penalty constraint multipliers

    IndexVectorX dbc; ///< Dirichlet constrained vertices

    std::vector<std::vector<Index>>
        partitions; ///< partitions[c] gives the c^{th} group of constraints which can all be
                    ///< projected independently in parallel
};

} // namespace xpbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_XPBD_DATA_H