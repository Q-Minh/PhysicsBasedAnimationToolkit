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
 * @brief Options controlling constraint formulation and Hessian approximation.
 */
struct ConstraintOptions
{
    /**
     * @brief Formulation type for the contact constraint \f$ c(x) \f$.
     *
     * Options control the quantity (and its derivatives) evaluated by the constraint
     * - Constraint evaluates \f$ c(x) \f$ and its derivatives
     * - Penalty evaluates \f$ 0.5 \mu c(x)^2 \f$ and its derivatives
     * - AugmentedLagrangian evaluates \f$ \mu c(x)^2 - \lambda c(x) \f$ and its derivatives
     * - InteriorPoint evaluates (the constraint part of) the merit function
     * \f$ -\mu log(s) + \lambda (c - s) \f$.
     * For the gradient, it computes
     * \f$ (\frac{\lambda c - \mu}{s} - \lambda) \nabla c(x) \f$.
     * For the hessian, it computes
     * \f$ \frac{\lambda}{s} \nabla c(x) \nabla c(x)^T - \lambda \nabla^2 c(x) \f$.
     */
    enum class EFormulation { Constraint, Penalty, AugmentedLagrangian, InteriorPoint };
    /**
     * @brief Hessian approximation type for the contact constraint.
     */
    enum class EHessianApproximation { Full, GaussNewton };

    EFormulation eFormulation{EFormulation::Constraint}; ///< Formulation type (default: Constraint)
    EHessianApproximation eHessianApprox{
        EHessianApproximation::Full}; ///< Hessian approximation type (default: Full)
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
struct MeshPointPointConstraint
{
    using ScalarType = TScalar; ///< Floating point scalar type
    using DistanceType =
        geometry::PointPointDistance<TScalar>; ///< Underlying distance computation type
    static constexpr int kStencilSize = DistanceType::kStencilSize; ///< Number of vertices (2)
    static constexpr int kDofs        = DistanceType::kDofs;        ///< Degrees of freedom (6)

    ScalarType lambda; ///< Lagrange multiplier for the constraint
    ScalarType mu;     ///< Penalty or complementarity relaxation for the constraint
    ScalarType s;      ///< Slack variable for interior point formulation
    ScalarType c;      ///< Constraint value
    math::linalg::mini::SVector<ScalarType, kDofs>
        g; ///< Gradient of the constraint with respect to the involved vertices
    math::linalg::mini::SMatrix<ScalarType, kDofs, kDofs>
        H; ///< Hessian of the constraint with respect to the involved vertices

    /**
     * @brief Evaluate the constraint function
     * @tparam TMatrixx Matrix type satisfying CMatrix concept
     * @param x `6 x 1` stacked positions of the two points
     * @return Constraint value
     */
    template <math::linalg::mini::CMatrix TMatrixx>
    PBAT_HOST_DEVICE TScalar Eval(TMatrixx const& x, ConstraintOptions const& options);

    /**
     * @brief Compute the gradient of the constraint
     * @tparam TMatrixx Matrix type satisfying CMatrix concept
     * @param x `6 x 1` stacked positions of the two points
     * @return `6 x 1` gradient vector
     */
    template <math::linalg::mini::CMatrix TMatrixx>
    PBAT_HOST_DEVICE void Gradient(TMatrixx const& x, ConstraintOptions const& options);

    /**
     * @brief Compute the Hessian of the constraint
     * @tparam TMatrixx Matrix type satisfying CMatrix concept
     * @param x `6 x 1` stacked positions of the two points
     * @return `6 x 6` Hessian matrix
     */
    template <math::linalg::mini::CMatrix TMatrixx>
    PBAT_HOST_DEVICE void Hessian(TMatrixx const& x, ConstraintOptions const& options);
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
struct MeshPointEdgeConstraint
{
    using ScalarType = TScalar; ///< Floating point scalar type
    using DistanceType =
        geometry::PointEdgeDistance<TScalar>; ///< Underlying distance computation type
    static constexpr int kStencilSize = DistanceType::kStencilSize; ///< Number of vertices (3)
    static constexpr int kDofs        = DistanceType::kDofs;        ///< Degrees of freedom (9)

    ScalarType lambda; ///< Lagrange multiplier for the constraint
    ScalarType mu;     ///< Penalty or complementarity relaxation for the constraint
    math::linalg::mini::SVector<ScalarType, kDofs>
        g; ///< Gradient of the constraint with respect to the involved vertices
    math::linalg::mini::SMatrix<ScalarType, kDofs, kDofs>
        H; ///< Hessian of the constraint with respect to the involved vertices

    /**
     * @brief Evaluate the constraint function
     * @tparam TMatrixx Matrix type satisfying CMatrix concept
     * @param x `9 x 1` stacked positions of point (x) and edge endpoints (a, b)
     * @return Constraint value
     */
    template <math::linalg::mini::CMatrix TMatrixx>
    PBAT_HOST_DEVICE TScalar Eval(TMatrixx const& x, ConstraintOptions const& options);

    /**
     * @brief Compute the gradient of the constraint
     * @tparam TMatrixx Matrix type satisfying CMatrix concept
     * @param x `9 x 1` stacked positions of point (x) and edge endpoints (a, b)
     * @return `9 x 1` gradient vector
     */
    template <math::linalg::mini::CMatrix TMatrixx>
    PBAT_HOST_DEVICE void Gradient(TMatrixx const& x, ConstraintOptions const& options);

    /**
     * @brief Compute the Hessian of the constraint
     * @tparam TMatrixx Matrix type satisfying CMatrix concept
     * @param x `9 x 1` stacked positions of point (x) and edge endpoints (a, b)
     * @return `9 x 9` Hessian matrix
     */
    template <math::linalg::mini::CMatrix TMatrixx>
    PBAT_HOST_DEVICE void Hessian(TMatrixx const& x, ConstraintOptions const& options);
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
{
    using ScalarType = TScalar; ///< Floating point scalar type
    using DistanceType =
        geometry::PointTriangleDistance<TScalar>; ///< Underlying distance computation type
    static constexpr int kStencilSize = DistanceType::kStencilSize; ///< Number of vertices (4)
    static constexpr int kDofs        = DistanceType::kDofs;        ///< Degrees of freedom (12)

    ScalarType lambda; ///< Lagrange multiplier for the constraint
    ScalarType mu;     ///< Penalty or complementarity relaxation for the constraint
    math::linalg::mini::SVector<ScalarType, kDofs>
        g; ///< Gradient of the constraint with respect to the involved vertices
    math::linalg::mini::SMatrix<ScalarType, kDofs, kDofs>
        H; ///< Hessian of the constraint with respect to the involved vertices

    /**
     * @brief Evaluate the constraint function
     * @tparam TMatrixx Matrix type satisfying CMatrix concept
     * @param x `12 x 1` stacked positions of point (x) and triangle vertices (a, b, c)
     * @return Constraint value (signed distance)
     */
    template <math::linalg::mini::CMatrix TMatrixx>
    PBAT_HOST_DEVICE TScalar Eval(TMatrixx const& x, ConstraintOptions const& options);

    /**
     * @brief Compute the gradient of the constraint
     * @tparam TMatrixx Matrix type satisfying CMatrix concept
     * @param x `12 x 1` stacked positions of point (x) and triangle vertices (a, b, c)
     * @return `12 x 1` gradient vector
     */
    template <math::linalg::mini::CMatrix TMatrixx>
    PBAT_HOST_DEVICE void Gradient(TMatrixx const& x, ConstraintOptions const& options);

    /**
     * @brief Compute the Hessian of the constraint
     * @tparam TMatrixx Matrix type satisfying CMatrix concept
     * @param x `12 x 1` stacked positions of point (x) and triangle vertices (a, b, c)
     * @return `12 x 12` Hessian matrix
     */
    template <math::linalg::mini::CMatrix TMatrixx>
    PBAT_HOST_DEVICE void Hessian(TMatrixx const& x, ConstraintOptions const& options);
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
struct MeshEdgeEdgeConstraint
{
    using ScalarType = TScalar; ///< Floating point scalar type
    using DistanceType =
        geometry::EdgeEdgeDistance<TScalar>; ///< Underlying distance computation type
    static constexpr int kStencilSize = DistanceType::kStencilSize; ///< Number of vertices (4)
    static constexpr int kDofs        = DistanceType::kDofs;        ///< Degrees of freedom (12)

    ScalarType lambda; ///< Lagrange multiplier for the constraint
    ScalarType mu;     ///< Penalty or complementarity relaxation for the constraint
    math::linalg::mini::SVector<ScalarType, kDofs>
        g; ///< Gradient of the constraint with respect to the involved vertices
    math::linalg::mini::SMatrix<ScalarType, kDofs, kDofs>
        H; ///< Hessian of the constraint with respect to the involved vertices

    /**
     * @brief Evaluate the constraint function
     * @tparam TMatrixx Matrix type satisfying CMatrix concept
     * @param x `12 x 1` stacked positions of edge 1 endpoints (a, b) and edge 2 endpoints (c, d)
     * @param eps Mollification parameter for numerical stability
     * @return Constraint value (signed distance)
     */
    template <math::linalg::mini::CMatrix TMatrixx>
    PBAT_HOST_DEVICE TScalar Eval(TMatrixx const& x, TScalar eps, ConstraintOptions const& options);

    /**
     * @brief Compute the gradient of the constraint
     * @tparam TMatrixx Matrix type satisfying CMatrix concept
     * @param x `12 x 1` stacked positions of edge 1 endpoints (a, b) and edge 2 endpoints (c, d)
     * @param eps Mollification parameter for numerical stability
     * @return `12 x 1` gradient vector
     */
    template <math::linalg::mini::CMatrix TMatrixx>
    PBAT_HOST_DEVICE void
    Gradient(TMatrixx const& x, TScalar eps, ConstraintOptions const& options);

    /**
     * @brief Compute the Hessian of the constraint
     * @tparam TMatrixx Matrix type satisfying CMatrix concept
     * @param x `12 x 1` stacked positions of edge 1 endpoints (a, b) and edge 2 endpoints (c, d)
     * @param eps Mollification parameter for numerical stability
     * @return `12 x 12` Hessian matrix
     */
    template <math::linalg::mini::CMatrix TMatrixx>
    PBAT_HOST_DEVICE void Hessian(TMatrixx const& x, TScalar eps, ConstraintOptions const& options);
};

template <common::CFloatingPoint TScalar>
template <math::linalg::mini::CMatrix TMatrixx>
inline PBAT_HOST_DEVICE TScalar
MeshPointPointConstraint<TScalar>::Eval(TMatrixx const& x, ConstraintOptions const& /*options*/)
{
    return DistanceType{}.Eval(x);
}

template <common::CFloatingPoint TScalar>
template <math::linalg::mini::CMatrix TMatrixx>
inline PBAT_HOST_DEVICE void
MeshPointPointConstraint<TScalar>::Gradient(TMatrixx const& x, ConstraintOptions const& /*options*/)
{
    g = DistanceType{}.Gradient(x);
}

template <common::CFloatingPoint TScalar>
template <math::linalg::mini::CMatrix TMatrixx>
inline PBAT_HOST_DEVICE void
MeshPointPointConstraint<TScalar>::Hessian(TMatrixx const& x, ConstraintOptions const& /*options*/)
{
    H = DistanceType{}.Hessian(x);
}

template <common::CFloatingPoint TScalar>
template <math::linalg::mini::CMatrix TMatrixx>
inline PBAT_HOST_DEVICE TScalar
MeshPointEdgeConstraint<TScalar>::Eval(TMatrixx const& x, ConstraintOptions const& /*options*/)
{
    return DistanceType{}.Eval(x);
}

template <common::CFloatingPoint TScalar>
template <math::linalg::mini::CMatrix TMatrixx>
inline PBAT_HOST_DEVICE void
MeshPointEdgeConstraint<TScalar>::Gradient(TMatrixx const& x, ConstraintOptions const& /*options*/)
{
    g = DistanceType{}.Gradient(x);
}

template <common::CFloatingPoint TScalar>
template <math::linalg::mini::CMatrix TMatrixx>
inline PBAT_HOST_DEVICE void
MeshPointEdgeConstraint<TScalar>::Hessian(TMatrixx const& x, ConstraintOptions const& /*options*/)
{
    H = DistanceType{}.Hessian(x);
}

template <common::CFloatingPoint TScalar>
template <math::linalg::mini::CMatrix TMatrixx>
inline PBAT_HOST_DEVICE TScalar
MeshPointTriangleConstraint<TScalar>::Eval(TMatrixx const& x, ConstraintOptions const& /*options*/)
{
    return DistanceType{}.Eval(x);
}

template <common::CFloatingPoint TScalar>
template <math::linalg::mini::CMatrix TMatrixx>
inline PBAT_HOST_DEVICE void MeshPointTriangleConstraint<TScalar>::Gradient(
    TMatrixx const& x,
    ConstraintOptions const& /*options*/)
{
    g = DistanceType{}.Gradient(x);
}

template <common::CFloatingPoint TScalar>
template <math::linalg::mini::CMatrix TMatrixx>
inline PBAT_HOST_DEVICE void MeshPointTriangleConstraint<TScalar>::Hessian(
    TMatrixx const& x,
    ConstraintOptions const& /*options*/)
{
    H = DistanceType{}.Hessian(x);
}

template <common::CFloatingPoint TScalar>
template <math::linalg::mini::CMatrix TMatrixx>
inline PBAT_HOST_DEVICE TScalar MeshEdgeEdgeConstraint<TScalar>::Eval(
    TMatrixx const& x,
    TScalar eps,
    ConstraintOptions const& /*options*/)
{
    return DistanceType{}.Eval(x, eps);
}

template <common::CFloatingPoint TScalar>
template <math::linalg::mini::CMatrix TMatrixx>
inline PBAT_HOST_DEVICE void MeshEdgeEdgeConstraint<TScalar>::Gradient(
    TMatrixx const& x,
    TScalar eps,
    ConstraintOptions const& /*options*/)
{
    g = DistanceType{}.Gradient(x, eps);
}

template <common::CFloatingPoint TScalar>
template <math::linalg::mini::CMatrix TMatrixx>
inline PBAT_HOST_DEVICE void MeshEdgeEdgeConstraint<TScalar>::Hessian(
    TMatrixx const& x,
    TScalar eps,
    ConstraintOptions const& /*options*/)
{
    H = DistanceType{}.Hessian(x, eps);
}

} // namespace pbat::sim::contact

#endif // PBAT_SIM_CONTACT_CONSTRAINTS_H
