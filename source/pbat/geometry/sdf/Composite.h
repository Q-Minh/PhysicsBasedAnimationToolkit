/**
 * @file Composite.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief This file defines an SDF composition
 * @version 0.1
 * @date 2025-09-16
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef PBAT_GEOMETRY_SDF_COMPOSITE_H
#define PBAT_GEOMETRY_SDF_COMPOSITE_H

#include "BinaryNode.h"
#include "Primitive.h"
#include "Transform.h"
#include "TypeDefs.h"
#include "UnaryNode.h"
#include "pbat/HostDevice.h"

#include <algorithm>
#include <array>
#include <span>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace pbat::geometry::sdf {

/**
 * @brief Variant type representing a node in the composite SDF forest.
 */
template <common::CArithmetic TScalar>
using Node = std::variant<
    /* primitives */
    Sphere<TScalar>,
    Box<TScalar>,
    BoxFrame<TScalar>,
    Torus<TScalar>,
    CappedTorus<TScalar>,
    Link<TScalar>,
    InfiniteCylinder<TScalar>,
    Cone<TScalar>,
    InfiniteCone<TScalar>,
    Plane<TScalar>,
    HexagonalPrism<TScalar>,
    Capsule<TScalar>,
    VerticalCapsule<TScalar>,
    CappedCylinder<TScalar>,
    VerticalCappedCylinder<TScalar>,
    RoundedCylinder<TScalar>,
    VerticalCappedCone<TScalar>,
    CutHollowSphere<TScalar>,
    VerticalRoundCone<TScalar>,
    Octahedron<TScalar>,
    Pyramid<TScalar>,
    Triangle<TScalar>,
    Quadrilateral<TScalar>,
    /* unary nodes */
    Scale<TScalar>,
    Elongate<TScalar>,
    Round<TScalar>,
    Onion<TScalar>,
    Symmetrize<TScalar>,
    Repeat<TScalar>,
    Bump<TScalar>,
    Twist<TScalar>,
    Bend<TScalar>,
    /* binary nodes */
    Union<TScalar>,
    Difference<TScalar>,
    Intersection<TScalar>,
    ExclusiveOr<TScalar>,
    SmoothUnion<TScalar>,
    SmoothDifference<TScalar>,
    SmoothIntersection<TScalar>>;

/**
 * @brief Composite signed distance function represented as a forest of SDFs.
 */
template <common::CArithmetic TScalar>
struct Composite
{
    static constexpr int kMaxRoots = 16;

    using ScalarType = TScalar; ///< Scalar type

    /**
     * @brief Status of the composite SDF
     */
    enum class EStatus {
        Valid,         ///< The composite SDF can be evaluated
        InvalidForest, ///< The SDF forest topology is invalid (wrong children/ancestor relations)
        UnexpectedNodeType ///< An unexpected node type was encountered
    };

    /**
     * @brief Construct a composite SDF from nodes, transforms, children, ancestors and roots
     * @param nodes Span of nodes in the composite SDF
     * @param transforms Span of transforms associated to each node
     * @param children Span of pairs of children indices for each node, such that c* < 0 if no child
     * @param roots Span of root indices of the composite SDF
     */
    PBAT_HOST_DEVICE
    Composite(
        std::span<Node<ScalarType> const> nodes,
        std::span<Transform<ScalarType> const> transforms,
        std::span<std::pair<int, int> const> children,
        std::span<int> roots);
    /**
     * @brief Evaluate the signed distance function at a given point
     * @param p Point at which to evaluate the SDF
     * @return Signed distance to the composite shape
     */
    PBAT_HOST_DEVICE
    ScalarType Eval(Vec3<ScalarType> const& p) const;
    /**
     * @brief Evaluate the signed distance function rooted in node n at a given point p
     * @param n Index of the root node
     * @param p Point at which to evaluate the SDF
     * @return Signed distance to the composite shape rooted in node n
     */
    PBAT_HOST_DEVICE
    ScalarType Eval(int n, Vec3<ScalarType> const& p) const;
    /**
     * @brief Evaluate the (numerical) gradient of the signed distance function at a given point
     * using a central difference scheme
     *
     * We use the tetrahedron technique described in https://iquilezles.org/articles/normalsSDF/
     *
     * @param p Point at which to evaluate the gradient
     * @param h Finite difference step size
     * @return Numerical gradient of the signed distance function at point p
     */
    PBAT_HOST_DEVICE
    Vec3<ScalarType> Grad(Vec3<ScalarType> const& p, ScalarType h) const;
    /**
     * @brief Get the status of the composite SDF
     * @return Status of the composite SDF
     */
    PBAT_HOST_DEVICE
    EStatus Status() const { return mStatus; }

    std::span<Node<ScalarType> const> mNodes; ///< `|# nodes|` nodes in the composite SDF
    std::span<Transform<ScalarType> const>
        mTransforms; ///< `|# nodes|` transforms associated to each node
    std::span<std::pair<int, int> const>
        mChildren; ///< `|# nodes|` children pairs (ci, cj) of each
                   ///< node, such that c* < 0 if no child. A binary node n has 2 children stored in
                   ///< (ci, cj), a unary node n has 1 child stored in ci, and a primitive has no
                   ///< children.
    std::span<int> mRoots; ///< `|# roots|` root indices of the composite SDF
    EStatus mStatus;       ///< Status of the composite SDF
};

/**
 * @brief Find the roots and parents of a composite SDF forest given its children
 * @param children `|# nodes|` children pairs (ci, cj) of each node, such that c* < 0 if no child
 * @return (roots, parents) where roots is a vector of root node indices in the forest and
 * parents is a vector of parent node indices, such that parents[n] = -1 if n is a root
 */
auto FindRootsAndParents(std::span<std::pair<int, int> const> children)
    -> std::pair<std::vector<int>, std::vector<int>>;

template <common::CArithmetic TScalar>
inline Composite<TScalar>::Composite(
    std::span<Node<ScalarType> const> nodes,
    std::span<Transform<ScalarType> const> transforms,
    std::span<std::pair<int, int> const> children,
    std::span<int> roots)
    : mNodes(nodes),
      mTransforms(transforms),
      mChildren(children),
      mRoots(roots),
      mStatus(EStatus::Valid)
{
    auto const nNodes = static_cast<int>(mNodes.size());
    for (auto n = 0; n < nNodes; ++n)
    {
        if (mStatus != EStatus::Valid)
            break;
        std::visit(
            [&](auto const& node) {
                using NodeType = std::remove_cvref_t<decltype(node)>;
                if constexpr (std::is_base_of_v<Primitive, NodeType>)
                {
                    if (mChildren[n].first >= 0 || mChildren[n].second >= 0)
                        mStatus = EStatus::InvalidForest;
                }
                else if constexpr (std::is_base_of_v<UnaryNode, NodeType>)
                {
                    if (mChildren[n].first < 0)
                        mStatus = EStatus::InvalidForest;
                    if (mChildren[n].second >= 0)
                        mStatus = EStatus::InvalidForest;
                }
                else if constexpr (std::is_base_of_v<BinaryNode, NodeType>)
                {
                    if (mChildren[n].first < 0 || mChildren[n].second < 0)
                        mStatus = EStatus::InvalidForest;
                }
            },
            mNodes[n]);
    }
}

template <common::CArithmetic TScalar>
inline TScalar Composite<TScalar>::Eval(Vec3<ScalarType> const& p) const
{
    using namespace std;
    TScalar sd = numeric_limits<TScalar>::max();
    for (auto root : mRoots)
        sd = min(sd, Eval(root, p));
    return sd;
}

template <common::CArithmetic TScalar>
inline TScalar Composite<TScalar>::Eval(int n, Vec3<ScalarType> const& p) const
{
    ScalarType sd;
    Vec3<ScalarType> q = mTransforms[n] / p;
    std::visit(
        [&](auto const& node) {
            using NodeType = std::remove_cvref_t<decltype(node)>;
            if constexpr (std::is_base_of_v<Primitive, NodeType>)
            {
                sd = node.Eval(q);
            }
            else if constexpr (std::is_base_of_v<UnaryNode, NodeType>)
            {
                sd = node.Eval(q, [&](Vec3<ScalarType> const& x) {
                    auto c = mChildren[n].first;
                    return Eval(c, x);
                });
            }
            else if constexpr (std::is_base_of_v<BinaryNode, NodeType>)
            {
                auto ci = mChildren[n].first;
                auto cj = mChildren[n].second;
                node.Eval(Eval(ci, q), Eval(cj, q));
            }
            else
            {
                mStatus = EStatus::UnexpectedNodeType;
            }
        },
        mNodes[n]);
    return sd;
}

template <common::CArithmetic TScalar>
inline Vec3<TScalar> Composite<TScalar>::Grad(Vec3<ScalarType> const& p, ScalarType h) const
{
    ScalarType constexpr one{1};
    ScalarType constexpr none{-1};
    Vec3<ScalarType> g;
    Vec3<ScalarType> k{one, none, none};
    g = k * Eval(p + h * k);
    k = Vec3<ScalarType>{none, none, one};
    g += k * Eval(p + h * k);
    k = Vec3<ScalarType>{none, one, none};
    g += k * Eval(p + h * k);
    k = Vec3<ScalarType>{one, one, one};
    g += k * Eval(p + h * k);
    g *= ScalarType(0.25);
    return g;
}

} // namespace pbat::geometry::sdf

#endif // PBAT_GEOMETRY_SDF_COMPOSITE_H
