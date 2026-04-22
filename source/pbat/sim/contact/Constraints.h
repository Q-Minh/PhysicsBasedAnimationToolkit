/**
 * @file Constraints.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Contact constraint types for mesh-based contact mechanics
 * @version 0.1
 * @date 2026-02-18
 * @copyright Copyright (c) 2026
 */

#ifndef PBAT_SIM_CONTACT_CONSTRAINTS_H
#define PBAT_SIM_CONTACT_CONSTRAINTS_H

#include "pbat/HostDevice.h"
#include "pbat/common/Concepts.h"
#include "pbat/geometry/MeshDistance.h"
#include "pbat/math/linalg/mini/Mini.h"

#include <cmath>

namespace pbat::sim::contact {

/**
 * @brief Formulation type for the contact constraint \f$ c(x) \f$.
 *
 * Options control the quantity (and its derivatives) evaluated by the constraint
 * - Constraint evaluates \f$ c(x) \f$ and its derivatives
 * - Penalty evaluates \f$ 0.5 \mu c(x)^2 \f$ and its derivatives
 * - AugmentedLagrangian evaluates \f$ 0.5 \mu c(x)^2 - \lambda c(x) \f$ and its derivatives
 * - InteriorPoint evaluates (the constraint part of) the merit function
 * \f$ -\mu log(s) + \lambda (c - s) \f$.
 * For the gradient, it computes
 * \f$ (\frac{\lambda c - \mu}{s} - \lambda) \nabla c(x) \f$.
 * For the hessian, it computes
 * \f$ \frac{\lambda}{s} \nabla c(x) \nabla c(x)^T - \lambda \nabla^2 c(x) \f$.
 */
enum class EConstraintFormulation { Constraint, Penalty, AugmentedLagrangian, InteriorPoint };

/**
 * @brief Which quantities to compute when evaluating a constraint.
 * Acts as a bitmask and can be converted to int for masking.
 */
enum EConstraintComputationFlags : int { Value = 1 << 0, Gradient = 1 << 1, Hessian = 1 << 2 };

/**
 * @brief The constraint hessian approximation type to use when evaluating the constraint Hessian.
 */
enum class EConstraintHessianApproximation { Full, GaussNewton };

/**
 * @brief Options for constraint evaluation, including computation flags and Hessian approximation.
 */
struct ConstraintComputationOptions
{
    EConstraintComputationFlags eComputeFlags{
        EConstraintComputationFlags::Value}; ///< Which quantities to compute
    EConstraintHessianApproximation eHessianApproximation{
        EConstraintHessianApproximation::Full}; ///< Which Hessian approximation to use
};

/**
 * @brief Generic mesh contact constraint struct template parameterized by a distance computation
 * type.
 * @tparam TDistance Mesh distance computation type (e.g. PointPointDistance, PointEdgeDistance,
 * etc.)
 */
template <class TDistance>
struct MeshPairConstraint
{
    using DistanceType = TDistance;                      ///< Underlying distance computation type
    using ScalarType   = typename TDistance::ScalarType; ///< Floating point scalar type
    static constexpr int kStencil =
        DistanceType::kStencil; ///< Number of vertices involved in the constraint
    static constexpr int kDofs =
        DistanceType::kDofs; ///< Degrees of freedom involved in the constraint

    ScalarType lambda; ///< Lagrange multiplier for the constraint
    ScalarType mu;     ///< Penalty or complementarity relaxation for the constraint
    ScalarType s;      ///< Slack variable for interior point formulation
    ScalarType c;      ///< Constraint value
    math::linalg::mini::SVector<ScalarType, kDofs>
        g; ///< Gradient of the constraint with respect to the involved vertices
    math::linalg::mini::SMatrix<ScalarType, kDofs, kDofs>
        H; ///< Hessian of the constraint with respect to the involved vertices

    /**
     * @brief Compute the constraint value, gradient, and Hessian based on the provided options
     * @tparam TMatrixx Matrix type satisfying CMatrix concept
     * @param x `kDofs x 1` stacked positions of the involved vertices
     * @param opts Options for constraint computation, including flags and Hessian approximation
     */
    template <math::linalg::mini::CMatrix TMatrixx>
    PBAT_HOST_DEVICE void Compute(TMatrixx const& x, ConstraintComputationOptions const& opts);

    /**
     * @brief Evaluate the constraint function
     * @tparam TMatrixx Matrix type satisfying CMatrix concept
     * @param x `kDofs x 1`
     * @param eFormulation Formulation type to determine which Hessian to compute
     * @return Constraint value
     */
    template <math::linalg::mini::CMatrix TMatrixx>
    PBAT_HOST_DEVICE auto Eval(TMatrixx const& x, EConstraintFormulation eFormulation) const
        -> ScalarType;

    /**
     * @brief Compute the gradient of the constraint
     * @tparam TMatrixx Matrix type satisfying CMatrix concept
     * @param x `kDofs x 1`
     * @param eFormulation Formulation type to determine which Hessian to compute
     * @return `kDofs x 1` gradient vector
     */
    template <math::linalg::mini::CMatrix TMatrixx>
    PBAT_HOST_DEVICE auto Gradient(TMatrixx const& x, EConstraintFormulation eFormulation) const
        -> math::linalg::mini::SVector<ScalarType, kDofs>;

    /**
     * @brief Compute the Hessian of the constraint
     * @tparam TMatrixx Matrix type satisfying CMatrix concept
     * @param x `kDofs x 1`
     * @param eFormulation Formulation type to determine which Hessian to compute
     * @return `kDofs x kDofs` Hessian matrix
     */
    template <math::linalg::mini::CMatrix TMatrixx>
    PBAT_HOST_DEVICE auto Hessian(TMatrixx const& x, EConstraintFormulation eFormulation) const
        -> math::linalg::mini::SMatrix<ScalarType, kDofs, kDofs>;
};

/**
 * @brief Point-point contact constraint for mesh contact mechanics
 *
 * Stores constraint data (Lagrange multiplier, penalty parameter, gradient, Hessian)
 * and provides methods to evaluate the constraint and its derivatives.
 * Uses geometry::PointPointDistance for distance computation.
 *
 * @tparam TScalar Floating point scalar type
 */
template <common::CFloatingPoint TScalar>
struct MeshPointPointConstraint : public MeshPairConstraint<geometry::PointPointDistance<TScalar>>
{
    using BaseType =
        MeshPairConstraint<geometry::PointPointDistance<TScalar>>; ///< Base type for common members
    using SelfType = MeshPointPointConstraint<TScalar>;            ///< Self type
};

/**
 * @brief Point-edge contact constraint for mesh contact mechanics
 *
 * Stores constraint data (Lagrange multiplier, penalty parameter, gradient, Hessian)
 * and provides methods to evaluate the constraint and its derivatives.
 * Uses geometry::PointEdgeDistance for distance computation.
 *
 * @tparam TScalar Floating point scalar type
 */
template <common::CFloatingPoint TScalar>
struct MeshPointEdgeConstraint : public MeshPairConstraint<geometry::PointEdgeDistance<TScalar>>
{
    using BaseType =
        MeshPairConstraint<geometry::PointEdgeDistance<TScalar>>; ///< Base type for common members
    using SelfType = MeshPointEdgeConstraint<TScalar>;            ///< Self type
};

/**
 * @brief Point-triangle contact constraint for mesh contact mechanics
 *
 * Stores constraint data (Lagrange multiplier, penalty parameter, gradient, Hessian)
 * and provides methods to evaluate the constraint and its derivatives.
 * Uses geometry::PointTriangleDistance for signed distance computation.
 *
 * @tparam TScalar Floating point scalar type
 */
template <common::CFloatingPoint TScalar>
struct MeshPointTriangleConstraint
    : public MeshPairConstraint<geometry::PointTriangleDistance<TScalar>>
{
    using BaseType =
        MeshPairConstraint<geometry::PointTriangleDistance<TScalar>>; ///< Base type for common
                                                                      ///< members
    using SelfType = MeshPointTriangleConstraint<TScalar>;            ///< Self type
};

/**
 * @brief Edge-edge contact constraint for mesh contact mechanics
 *
 * Stores constraint data (Lagrange multiplier, penalty parameter, gradient, Hessian)
 * and provides methods to evaluate the constraint and its derivatives.
 * Uses geometry::EdgeEdgeDistance for signed distance computation with mollified norm.
 *
 * @tparam TScalar Floating point scalar type
 */
template <common::CFloatingPoint TScalar>
struct MeshEdgeEdgeConstraint : public MeshPairConstraint<geometry::EdgeEdgeDistance<TScalar>>
{
    using BaseType =
        MeshPairConstraint<geometry::EdgeEdgeDistance<TScalar>>; ///< Base type for common members
    using SelfType = MeshEdgeEdgeConstraint<TScalar>;            ///< Self type
};

template <class TDistance>
template <math::linalg::mini::CMatrix TMatrixx>
inline PBAT_HOST_DEVICE void
MeshPairConstraint<TDistance>::Compute(TMatrixx const& x, ConstraintComputationOptions const& opts)
{
    int mask = static_cast<int>(opts.eComputeFlags);
    if (mask & static_cast<int>(EConstraintComputationFlags::Value))
        c = DistanceType{}.Eval(x);
    if (mask & static_cast<int>(EConstraintComputationFlags::Gradient))
        g = DistanceType{}.Gradient(x);
    if (mask & static_cast<int>(EConstraintComputationFlags::Hessian))
    {
        switch (opts.eHessianApproximation)
        {
            case EConstraintHessianApproximation::Full: H = DistanceType{}.Hessian(x); break;
            case EConstraintHessianApproximation::GaussNewton: H = g * g.Transpose(); break;
            default: H = DistanceType{}.Hessian(x); break;
        }
    }
}

template <class TDistance>
template <math::linalg::mini::CMatrix TMatrixx>
inline PBAT_HOST_DEVICE auto
MeshPairConstraint<TDistance>::Eval(TMatrixx const& x, EConstraintFormulation eFormulation) const
    -> ScalarType
{
    switch (eFormulation)
    {
        case EConstraintFormulation::Constraint: return c;
        case EConstraintFormulation::Penalty: return ScalarType(0.5) * mu * c * c;
        case EConstraintFormulation::AugmentedLagrangian:
            return (ScalarType(0.5) * mu * c - lambda) * c;
        case EConstraintFormulation::InteriorPoint: return -mu * std::log(s) + lambda * (c - s);
        default: return c;
    }
}

template <class TDistance>
template <math::linalg::mini::CMatrix TMatrixx>
inline PBAT_HOST_DEVICE auto MeshPairConstraint<TDistance>::Gradient(
    TMatrixx const& x,
    EConstraintFormulation eFormulation) const -> math::linalg::mini::SVector<ScalarType, kDofs>
{
    switch (eFormulation)
    {
        case EConstraintFormulation::Constraint: return g;
        case EConstraintFormulation::Penalty: return mu * c * g;
        case EConstraintFormulation::AugmentedLagrangian: return (mu * c - lambda) * g;
        case EConstraintFormulation::InteriorPoint: return ((lambda * c - mu) / s - lambda) * g;
        default: return g;
    }
}

template <class TDistance>
template <math::linalg::mini::CMatrix TMatrixx>
inline PBAT_HOST_DEVICE auto
MeshPairConstraint<TDistance>::Hessian(TMatrixx const& x, EConstraintFormulation eFormulation) const
    -> math::linalg::mini::SMatrix<ScalarType, kDofs, kDofs>
{
    switch (eFormulation)
    {
        case EConstraintFormulation::Constraint: return H;
        case EConstraintFormulation::Penalty: return mu * (g * g.Transpose() + c * H);
        case EConstraintFormulation::AugmentedLagrangian:
            return mu * g * g.Transpose() + (mu * c - lambda) * H;
        case EConstraintFormulation::InteriorPoint:
            return (lambda / s) * g * g.Transpose() - lambda * H;
        default: return H;
    }
}

} // namespace pbat::sim::contact

#endif // PBAT_SIM_CONTACT_CONSTRAINTS_H
