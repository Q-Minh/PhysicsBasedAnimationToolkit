#ifndef PBAT_GEOMETRY_SDF_FOREST_H
#define PBAT_GEOMETRY_SDF_FOREST_H

#include "Composite.h"
#include "Transform.h"
#include "pbat/common/Concepts.h"
#include "pbat/io/Archive.h"

#include <array>
#include <fmt/format.h>
#include <string_view>
#include <utility>
#include <vector>

namespace pbat::geometry::sdf {

/**
 * @brief CPU storage for a forest (of SDFs).
 */
template <common::CArithmetic TScalar>
struct Forest
{
    using ScalarType = TScalar;          ///< Scalar type
    std::vector<Node<ScalarType>> nodes; ///< `|# nodes|` nodes in the forest
    std::vector<Transform<ScalarType>>
        transforms;         ///< `|# nodes|` transforms associated to each node
    std::vector<int> roots; ///< `|# roots|` indices of the root nodes in the forest
    std::vector<std::pair<int, int>> children; ///< `|# nodes|` list of pairs of children indices
                                               ///< for each node, such that c* < 0 if no child
    /**
     * @brief Serialize the forest to an archive
     * @param archive Archive to serialize to
     */
    void Serialize(io::Archive& archive) const;
    /**
     * @brief Deserialize the forest from an archive
     * @param archive Archive to deserialize from
     */
    void Deserialize(io::Archive& archive);
};

namespace detail::forest {

template <class TMatrix>
void SerializeMiniMatrix(std::string_view key, TMatrix const& A, io::Archive& archive)
{
    using ScalarType = typename TMatrix::ScalarType;
    std::vector<ScalarType> data(TMatrix::kRows * TMatrix::kCols);
    std::memcpy(data.data(), A.Data(), sizeof(ScalarType) * data.size());
    archive.WriteData(std::string{key}, data);
}

template <class TMatrix>
void DeserializeMiniMatrix(std::string_view key, TMatrix& A, io::Archive const& archive)
{
    using ScalarType             = typename TMatrix::ScalarType;
    std::vector<ScalarType> data = archive.ReadData<std::vector<ScalarType>>(std::string{key});
    std::memcpy(A.Data(), data.data(), sizeof(ScalarType) * data.size());
}

} // namespace detail::forest

template <common::CArithmetic TScalar>
void Forest<TScalar>::Serialize(io::Archive& archive) const
{
    io::Archive group      = archive["pbat.geometry.sdf.Forest"];
    io::Archive nodesGroup = group["nodes"];
    struct Visitor
    {
        Visitor(io::Archive const& _group, std::size_t _i) : group(_group), i(_i) {}
        io::Archive group;
        std::size_t i;
        /**
         * Primitives
         */
        void operator()(Sphere<TScalar> const& prim)
        {
            io::Archive sphereGroup = group.GetOrCreateGroup("pbat.geometry.sdf.Sphere");
            sphereGroup.WriteMetaData("radius", prim.R);
        }
        void operator()(Box<TScalar> const& prim)
        {
            io::Archive boxGroup = group.GetOrCreateGroup("pbat.geometry.sdf.Box");
            detail::forest::SerializeMiniMatrix("he", prim.he, boxGroup);
        }
        void operator()(BoxFrame<TScalar> const& prim)
        {
            io::Archive boxFrameGroup = group.GetOrCreateGroup("pbat.geometry.sdf.BoxFrame");
            detail::forest::SerializeMiniMatrix("he", prim.he, boxFrameGroup);
            boxFrameGroup.WriteMetaData("t", prim.t);
        }
        void operator()(Torus<TScalar> const& prim)
        {
            io::Archive torusGroup = group.GetOrCreateGroup("pbat.geometry.sdf.Torus");
            detail::forest::SerializeMiniMatrix("t", prim.t, torusGroup);
        }
        void operator()(CappedTorus<TScalar> const& prim)
        {
            io::Archive cappedTorusGroup = group.GetOrCreateGroup("pbat.geometry.sdf.CappedTorus");
            detail::forest::SerializeMiniMatrix("sc", prim.sc, cappedTorusGroup);
            cappedTorusGroup.WriteMetaData("ra", prim.ra);
            cappedTorusGroup.WriteMetaData("rb", prim.rb);
        }
        void operator()(Link<TScalar> const& prim)
        {
            io::Archive linkGroup = group.GetOrCreateGroup("pbat.geometry.sdf.Link");
            detail::forest::SerializeMiniMatrix("t", prim.t, linkGroup);
            linkGroup.WriteMetaData("le", prim.le);
        }
        void operator()(InfiniteCylinder<TScalar> const& prim)
        {
            io::Archive infiniteCylinderGroup =
                group.GetOrCreateGroup("pbat.geometry.sdf.InfiniteCylinder");
            detail::forest::SerializeMiniMatrix("c", prim.c, infiniteCylinderGroup);
        }
        void operator()(Cone<TScalar> const& prim)
        {
            io::Archive coneGroup = group.GetOrCreateGroup("pbat.geometry.sdf.Cone");
            detail::forest::SerializeMiniMatrix("sc", prim.sc, coneGroup);
            coneGroup.WriteMetaData("h", prim.h);
        }
        void operator()(InfiniteCone<TScalar> const& prim)
        {
            io::Archive infiniteConeGroup =
                group.GetOrCreateGroup("pbat.geometry.sdf.InfiniteCone");
            detail::forest::SerializeMiniMatrix("sc", prim.sc, infiniteConeGroup);
        }
        void operator()([[maybe_unused]] Plane<TScalar> const& prim)
        {
            [[maybe_unused]] io::Archive planeGroup =
                group.GetOrCreateGroup("pbat.geometry.sdf.Plane");
        }
        void operator()(HexagonalPrism<TScalar> const& prim)
        {
            io::Archive hexagonalPrismGroup =
                group.GetOrCreateGroup("pbat.geometry.sdf.HexagonalPrism");
            detail::forest::SerializeMiniMatrix("h", prim.h, hexagonalPrismGroup);
        }
        void operator()(Capsule<TScalar> const& prim)
        {
            io::Archive capsuleGroup = group.GetOrCreateGroup("pbat.geometry.sdf.Capsule");
            detail::forest::SerializeMiniMatrix("a", prim.a, capsuleGroup);
            detail::forest::SerializeMiniMatrix("b", prim.b, capsuleGroup);
            capsuleGroup.WriteMetaData("r", prim.r);
        }
        void operator()(VerticalCapsule<TScalar> const& prim)
        {
            io::Archive verticalCapsuleGroup =
                group.GetOrCreateGroup("pbat.geometry.sdf.VerticalCapsule");
            verticalCapsuleGroup.WriteMetaData("h", prim.h);
            verticalCapsuleGroup.WriteMetaData("r", prim.r);
        }
        void operator()(CappedCylinder<TScalar> const& prim)
        {
            io::Archive cappedCylinderGroup =
                group.GetOrCreateGroup("pbat.geometry.sdf.CappedCylinder");
            detail::forest::SerializeMiniMatrix("a", prim.a, cappedCylinderGroup);
            detail::forest::SerializeMiniMatrix("b", prim.b, cappedCylinderGroup);
            cappedCylinderGroup.WriteMetaData("r", prim.r);
        }
        void operator()(VerticalCappedCylinder<TScalar> const& prim)
        {
            io::Archive verticalCappedCylinderGroup =
                group.GetOrCreateGroup("pbat.geometry.sdf.VerticalCappedCylinder");
            verticalCappedCylinderGroup.WriteMetaData("h", prim.h);
            verticalCappedCylinderGroup.WriteMetaData("r", prim.r);
        }
        void operator()(RoundedCylinder<TScalar> const& prim)
        {
            io::Archive roundedCylinderGroup =
                group.GetOrCreateGroup("pbat.geometry.sdf.RoundedCylinder");
            roundedCylinderGroup.WriteMetaData("h", prim.h);
            roundedCylinderGroup.WriteMetaData("ra", prim.ra);
            roundedCylinderGroup.WriteMetaData("rb", prim.rb);
        }
        void operator()(VerticalCappedCone<TScalar> const& prim)
        {
            io::Archive verticalCappedConeGroup =
                group.GetOrCreateGroup("pbat.geometry.sdf.VerticalCappedCone");
            verticalCappedConeGroup.WriteMetaData("h", prim.h);
            verticalCappedConeGroup.WriteMetaData("r1", prim.r1);
            verticalCappedConeGroup.WriteMetaData("r2", prim.r2);
        }
        void operator()(CutHollowSphere<TScalar> const& prim)
        {
            io::Archive cutHollowSphereGroup =
                group.GetOrCreateGroup("pbat.geometry.sdf.CutHollowSphere");
            cutHollowSphereGroup.WriteMetaData("r", prim.r);
            cutHollowSphereGroup.WriteMetaData("h", prim.h);
            cutHollowSphereGroup.WriteMetaData("t", prim.t);
        }
        void operator()(VerticalRoundCone<TScalar> const& prim)
        {
            io::Archive verticalRoundConeGroup =
                group.GetOrCreateGroup("pbat.geometry.sdf.VerticalRoundCone");
            verticalRoundConeGroup.WriteMetaData("h", prim.h);
            verticalRoundConeGroup.WriteMetaData("r1", prim.r1);
            verticalRoundConeGroup.WriteMetaData("r2", prim.r2);
        }
        void operator()(Octahedron<TScalar> const& prim)
        {
            io::Archive octahedronGroup = group.GetOrCreateGroup("pbat.geometry.sdf.Octahedron");
            octahedronGroup.WriteMetaData("s", prim.s);
        }
        void operator()(Pyramid<TScalar> const& prim)
        {
            io::Archive pyramidGroup = group.GetOrCreateGroup("pbat.geometry.sdf.Pyramid");
            pyramidGroup.WriteMetaData("h", prim.h);
        }
        void operator()(Triangle<TScalar> const& prim)
        {
            io::Archive triangleGroup = group.GetOrCreateGroup("pbat.geometry.sdf.Triangle");
            detail::forest::SerializeMiniMatrix("a", prim.a, triangleGroup);
            detail::forest::SerializeMiniMatrix("b", prim.b, triangleGroup);
            detail::forest::SerializeMiniMatrix("c", prim.c, triangleGroup);
        }
        void operator()(Quadrilateral<TScalar> const& prim)
        {
            io::Archive quadrilateralGroup =
                group.GetOrCreateGroup("pbat.geometry.sdf.Quadrilateral");
            detail::forest::SerializeMiniMatrix("a", prim.a, quadrilateralGroup);
            detail::forest::SerializeMiniMatrix("b", prim.b, quadrilateralGroup);
            detail::forest::SerializeMiniMatrix("c", prim.c, quadrilateralGroup);
            detail::forest::SerializeMiniMatrix("d", prim.d, quadrilateralGroup);
        }

        /**
         * Unary nodes
         */
        void operator()(Scale<TScalar> const& un)
        {
            io::Archive scaleGroup = group.GetOrCreateGroup("pbat.geometry.sdf.Scale");
            scaleGroup.WriteMetaData("s", un.s);
        }
        void operator()(Elongate<TScalar> const& un)
        {
            io::Archive elongateGroup = group.GetOrCreateGroup("pbat.geometry.sdf.Elongate");
            detail::forest::SerializeMiniMatrix("h", un.h, elongateGroup);
        }
        void operator()(Round<TScalar> const& un)
        {
            io::Archive roundGroup = group.GetOrCreateGroup("pbat.geometry.sdf.Round");
            roundGroup.WriteMetaData("r", un.r);
        }
        void operator()(Onion<TScalar> const& un)
        {
            io::Archive onionGroup = group.GetOrCreateGroup("pbat.geometry.sdf.Onion");
            onionGroup.WriteMetaData("t", un.t);
        }
        void operator()([[maybe_unused]] Symmetrize<TScalar> const& un)
        {
            [[maybe_unused]] io::Archive symmetrizeGroup =
                group.GetOrCreateGroup("pbat.geometry.sdf.Symmetrize");
        }
        void operator()(Repeat<TScalar> const& un)
        {
            io::Archive repeatGroup = group.GetOrCreateGroup("pbat.geometry.sdf.Repeat");
            repeatGroup.WriteMetaData("s", un.s);
            detail::forest::SerializeMiniMatrix("l", un.l, repeatGroup);
        }
        void operator()(Bump<TScalar> const& un)
        {
            io::Archive bumpGroup = group.GetOrCreateGroup("pbat.geometry.sdf.Bump");
            bumpGroup.WriteMetaData("f", un.f);
            bumpGroup.WriteMetaData("g", un.g);
        }
        void operator()(Twist<TScalar> const& un)
        {
            io::Archive twistGroup = group.GetOrCreateGroup("pbat.geometry.sdf.Twist");
            twistGroup.WriteMetaData("k", un.k);
        }
        void operator()(Bend<TScalar> const& un)
        {
            io::Archive bendGroup = group.GetOrCreateGroup("pbat.geometry.sdf.Bend");
            bendGroup.WriteMetaData("k", un.k);
        }

        /**
         * Binary nodes
         */
        void operator()([[maybe_unused]] Union<TScalar> const& bn)
        {
            [[maybe_unused]] io::Archive unionGroup =
                group.GetOrCreateGroup("pbat.geometry.sdf.Union");
        }
        void operator()([[maybe_unused]] Difference<TScalar> const& bn)
        {
            [[maybe_unused]] io::Archive differenceGroup =
                group.GetOrCreateGroup("pbat.geometry.sdf.Difference");
        }
        void operator()([[maybe_unused]] Intersection<TScalar> const& bn)
        {
            [[maybe_unused]] io::Archive intersectionGroup =
                group.GetOrCreateGroup("pbat.geometry.sdf.Intersection");
        }
        void operator()([[maybe_unused]] ExclusiveOr<TScalar> const& bn)
        {
            [[maybe_unused]] io::Archive exclusiveOrGroup =
                group.GetOrCreateGroup("pbat.geometry.sdf.ExclusiveOr");
        }
        void operator()(SmoothUnion<TScalar> const& bn)
        {
            io::Archive smoothUnionGroup = group.GetOrCreateGroup("pbat.geometry.sdf.SmoothUnion");
            smoothUnionGroup.WriteMetaData("k", bn.k);
        }
        void operator()(SmoothDifference<TScalar> const& bn)
        {
            io::Archive smoothDifferenceGroup =
                group.GetOrCreateGroup("pbat.geometry.sdf.SmoothDifference");
            smoothDifferenceGroup.WriteMetaData("k", bn.k);
        }
        void operator()(SmoothIntersection<TScalar> const& bn)
        {
            io::Archive smoothIntersectionGroup =
                group.GetOrCreateGroup("pbat.geometry.sdf.SmoothIntersection");
            smoothIntersectionGroup.WriteMetaData("k", bn.k);
        }
    };
    group.WriteMetaData("nNodes", static_cast<int>(nodes.size()));
    for (std::size_t i = 0; i < nodes.size(); ++i)
    {
        io::Archive nodeGroup = nodesGroup[fmt::format("{}", i)];
        std::visit(Visitor{nodeGroup, i}, nodes[i]);
    }
    io::Archive transformsGroup = group["transforms"];
    for (std::size_t i = 0; i < transforms.size(); ++i)
    {
        Transform<TScalar> const& tr = transforms[i];
        detail::forest::SerializeMiniMatrix(fmt::format("R{}", i), tr.R, transformsGroup);
        detail::forest::SerializeMiniMatrix(fmt::format("t{}", i), tr.t, transformsGroup);
    }
    group.WriteData("roots", roots);
    std::vector<int> lc(children.size()), rc(children.size());
    for (std::size_t i = 0; i < children.size(); ++i)
    {
        lc[i] = children[i].first;
        rc[i] = children[i].second;
    }
    group.WriteData("lc", lc);
    group.WriteData("rc", rc);
}

template <common::CArithmetic TScalar>
void Forest<TScalar>::Deserialize(io::Archive& archive)
{
    io::Archive group  = archive["pbat.geometry.sdf.Forest"];
    std::size_t nNodes = group.ReadMetaData<std::size_t>("nNodes");
    nodes.resize(nNodes);
    io::Archive nodesGroup = group["nodes"];
    struct Visitor
    {
        Visitor(io::Archive const& _group, std::size_t _i) : group(_group), i(_i) {}
        io::Archive group;
        std::size_t i;

        /**
         * Primitives
         */
        void operator()(Sphere<TScalar>& prim) const
        {
            io::Archive sphereGroup = group["pbat.geometry.sdf.Sphere"];
            prim.R                  = sphereGroup.ReadMetaData<TScalar>("radius");
        }
        void operator()(Box<TScalar>& prim) const
        {
            io::Archive boxGroup = group["pbat.geometry.sdf.Box"];
            detail::forest::DeserializeMiniMatrix("he", prim.he, boxGroup);
        }
        void operator()(BoxFrame<TScalar>& prim) const
        {
            io::Archive boxFrameGroup = group["pbat.geometry.sdf.BoxFrame"];
            detail::forest::DeserializeMiniMatrix("he", prim.he, boxFrameGroup);
            prim.t = boxFrameGroup.ReadMetaData<TScalar>("t");
        }
        void operator()(Torus<TScalar>& prim) const
        {
            io::Archive torusGroup = group["pbat.geometry.sdf.Torus"];
            detail::forest::DeserializeMiniMatrix("t", prim.t, torusGroup);
        }
        void operator()(CappedTorus<TScalar>& prim) const
        {
            io::Archive cappedTorusGroup = group["pbat.geometry.sdf.CappedTorus"];
            detail::forest::DeserializeMiniMatrix("sc", prim.sc, cappedTorusGroup);
            prim.ra = cappedTorusGroup.ReadMetaData<TScalar>("ra");
            prim.rb = cappedTorusGroup.ReadMetaData<TScalar>("rb");
        }
        void operator()(Link<TScalar>& prim) const
        {
            io::Archive linkGroup = group["pbat.geometry.sdf.Link"];
            detail::forest::DeserializeMiniMatrix("t", prim.t, linkGroup);
            prim.le = linkGroup.ReadMetaData<TScalar>("le");
        }
        void operator()(InfiniteCylinder<TScalar>& prim) const
        {
            io::Archive infiniteCylinderGroup = group["pbat.geometry.sdf.InfiniteCylinder"];
            detail::forest::DeserializeMiniMatrix("c", prim.c, infiniteCylinderGroup);
        }
        void operator()(Cone<TScalar>& prim) const
        {
            io::Archive coneGroup = group["pbat.geometry.sdf.Cone"];
            detail::forest::DeserializeMiniMatrix("sc", prim.sc, coneGroup);
            prim.h = coneGroup.ReadMetaData<TScalar>("h");
        }
        void operator()(InfiniteCone<TScalar>& prim) const
        {
            io::Archive infiniteConeGroup = group["pbat.geometry.sdf.InfiniteCone"];
            detail::forest::DeserializeMiniMatrix("sc", prim.sc, infiniteConeGroup);
        }
        void operator()([[maybe_unused]] Plane<TScalar>& prim) const
        {
            [[maybe_unused]] io::Archive planeGroup = group["pbat.geometry.sdf.Plane"];
        }
        void operator()(HexagonalPrism<TScalar>& prim) const
        {
            io::Archive hexagonalPrismGroup = group["pbat.geometry.sdf.HexagonalPrism"];
            detail::forest::DeserializeMiniMatrix("h", prim.h, hexagonalPrismGroup);
        }
        void operator()(Capsule<TScalar>& prim) const
        {
            io::Archive capsuleGroup = group["pbat.geometry.sdf.Capsule"];
            detail::forest::DeserializeMiniMatrix("a", prim.a, capsuleGroup);
            detail::forest::DeserializeMiniMatrix("b", prim.b, capsuleGroup);
            prim.r = capsuleGroup.ReadMetaData<TScalar>("r");
        }
        void operator()(VerticalCapsule<TScalar>& prim) const
        {
            io::Archive verticalCapsuleGroup = group["pbat.geometry.sdf.VerticalCapsule"];
            prim.h                           = verticalCapsuleGroup.ReadMetaData<TScalar>("h");
            prim.r                           = verticalCapsuleGroup.ReadMetaData<TScalar>("r");
        }
        void operator()(CappedCylinder<TScalar>& prim) const
        {
            io::Archive cappedCylinderGroup = group["pbat.geometry.sdf.CappedCylinder"];
            detail::forest::DeserializeMiniMatrix("a", prim.a, cappedCylinderGroup);
            detail::forest::DeserializeMiniMatrix("b", prim.b, cappedCylinderGroup);
            prim.r = cappedCylinderGroup.ReadMetaData<TScalar>("r");
        }
        void operator()(VerticalCappedCylinder<TScalar>& prim) const
        {
            io::Archive verticalCappedCylinderGroup =
                group["pbat.geometry.sdf.VerticalCappedCylinder"];
            prim.h = verticalCappedCylinderGroup.ReadMetaData<TScalar>("h");
            prim.r = verticalCappedCylinderGroup.ReadMetaData<TScalar>("r");
        }
        void operator()(RoundedCylinder<TScalar>& prim) const
        {
            io::Archive roundedCylinderGroup = group["pbat.geometry.sdf.RoundedCylinder"];
            prim.h                           = roundedCylinderGroup.ReadMetaData<TScalar>("h");
            prim.ra                          = roundedCylinderGroup.ReadMetaData<TScalar>("ra");
            prim.rb                          = roundedCylinderGroup.ReadMetaData<TScalar>("rb");
        }
        void operator()(VerticalCappedCone<TScalar>& prim) const
        {
            io::Archive verticalCappedConeGroup = group["pbat.geometry.sdf.VerticalCappedCone"];
            prim.h  = verticalCappedConeGroup.ReadMetaData<TScalar>("h");
            prim.r1 = verticalCappedConeGroup.ReadMetaData<TScalar>("r1");
            prim.r2 = verticalCappedConeGroup.ReadMetaData<TScalar>("r2");
        }
        void operator()(CutHollowSphere<TScalar>& prim) const
        {
            io::Archive cutHollowSphereGroup = group["pbat.geometry.sdf.CutHollowSphere"];
            prim.r                           = cutHollowSphereGroup.ReadMetaData<TScalar>("r");
            prim.h                           = cutHollowSphereGroup.ReadMetaData<TScalar>("h");
            prim.t                           = cutHollowSphereGroup.ReadMetaData<TScalar>("t");
        }
        void operator()(VerticalRoundCone<TScalar>& prim) const
        {
            io::Archive verticalRoundConeGroup = group["pbat.geometry.sdf.VerticalRoundCone"];
            prim.h                             = verticalRoundConeGroup.ReadMetaData<TScalar>("h");
            prim.r1                            = verticalRoundConeGroup.ReadMetaData<TScalar>("r1");
            prim.r2                            = verticalRoundConeGroup.ReadMetaData<TScalar>("r2");
        }
        void operator()(Octahedron<TScalar>& prim) const
        {
            io::Archive octahedronGroup = group["pbat.geometry.sdf.Octahedron"];
            prim.s                      = octahedronGroup.ReadMetaData<TScalar>("s");
        }
        void operator()(Pyramid<TScalar>& prim) const
        {
            io::Archive pyramidGroup = group["pbat.geometry.sdf.Pyramid"];
            prim.h                   = pyramidGroup.ReadMetaData<TScalar>("h");
        }
        void operator()(Triangle<TScalar>& prim) const
        {
            io::Archive triangleGroup = group["pbat.geometry.sdf.Triangle"];
            detail::forest::DeserializeMiniMatrix("a", prim.a, triangleGroup);
            detail::forest::DeserializeMiniMatrix("b", prim.b, triangleGroup);
            detail::forest::DeserializeMiniMatrix("c", prim.c, triangleGroup);
        }
        void operator()(Quadrilateral<TScalar>& prim) const
        {
            io::Archive quadrilateralGroup = group["pbat.geometry.sdf.Quadrilateral"];
            detail::forest::DeserializeMiniMatrix("a", prim.a, quadrilateralGroup);
            detail::forest::DeserializeMiniMatrix("b", prim.b, quadrilateralGroup);
            detail::forest::DeserializeMiniMatrix("c", prim.c, quadrilateralGroup);
            detail::forest::DeserializeMiniMatrix("d", prim.d, quadrilateralGroup);
        }

        /**
         * Unary nodes
         */
        void operator()(Scale<TScalar>& un) const
        {
            io::Archive scaleGroup = group["pbat.geometry.sdf.Scale"];
            un.s                   = scaleGroup.ReadMetaData<TScalar>("s");
        }
        void operator()(Elongate<TScalar>& un) const
        {
            io::Archive elongateGroup = group["pbat.geometry.sdf.Elongate"];
            detail::forest::DeserializeMiniMatrix("h", un.h, elongateGroup);
        }
        void operator()(Round<TScalar>& un) const
        {
            io::Archive roundGroup = group["pbat.geometry.sdf.Round"];
            un.r                   = roundGroup.ReadMetaData<TScalar>("r");
        }
        void operator()(Onion<TScalar>& un) const
        {
            io::Archive onionGroup = group["pbat.geometry.sdf.Onion"];
            un.t                   = onionGroup.ReadMetaData<TScalar>("t");
        }
        void operator()([[maybe_unused]] Symmetrize<TScalar>& un) const
        {
            [[maybe_unused]] io::Archive symmetrizeGroup = group["pbat.geometry.sdf.Symmetrize"];
        }
        void operator()(Repeat<TScalar>& un) const
        {
            io::Archive repeatGroup = group["pbat.geometry.sdf.Repeat"];
            un.s                    = repeatGroup.ReadMetaData<TScalar>("s");
            detail::forest::DeserializeMiniMatrix("l", un.l, repeatGroup);
        }
        void operator()(Bump<TScalar>& un) const
        {
            io::Archive bumpGroup = group["pbat.geometry.sdf.Bump"];
            detail::forest::DeserializeMiniMatrix("f", un.f, bumpGroup);
            detail::forest::DeserializeMiniMatrix("g", un.g, bumpGroup);
        }
        void operator()(Twist<TScalar>& un) const
        {
            io::Archive twistGroup = group["pbat.geometry.sdf.Twist"];
            un.k                   = twistGroup.ReadMetaData<TScalar>("k");
        }
        void operator()(Bend<TScalar>& un) const
        {
            io::Archive bendGroup = group["pbat.geometry.sdf.Bend"];
            un.k                  = bendGroup.ReadMetaData<TScalar>("k");
        }

        /**
         * Binary nodes
         */
        void operator()([[maybe_unused]] Union<TScalar>& bn) const
        {
            [[maybe_unused]] io::Archive unionGroup = group["pbat.geometry.sdf.Union"];
        }
        void operator()([[maybe_unused]] Difference<TScalar>& bn) const
        {
            [[maybe_unused]] io::Archive differenceGroup = group["pbat.geometry.sdf.Difference"];
        }
        void operator()([[maybe_unused]] Intersection<TScalar>& bn) const
        {
            [[maybe_unused]] io::Archive intersectionGroup =
                group["pbat.geometry.sdf.Intersection"];
        }
        void operator()([[maybe_unused]] ExclusiveOr<TScalar>& bn) const
        {
            [[maybe_unused]] io::Archive exclusiveOrGroup = group["pbat.geometry.sdf.ExclusiveOr"];
        }
        void operator()(SmoothUnion<TScalar>& bn) const
        {
            io::Archive smoothUnionGroup = group["pbat.geometry.sdf.SmoothUnion"];
            bn.k                         = smoothUnionGroup.ReadMetaData<TScalar>("k");
        }
        void operator()(SmoothDifference<TScalar>& bn) const
        {
            io::Archive smoothDifferenceGroup = group["pbat.geometry.sdf.SmoothDifference"];
            bn.k                              = smoothDifferenceGroup.ReadMetaData<TScalar>("k");
        }
        void operator()(SmoothIntersection<TScalar>& bn) const
        {
            io::Archive smoothIntersectionGroup = group["pbat.geometry.sdf.SmoothIntersection"];
            bn.k = smoothIntersectionGroup.ReadMetaData<TScalar>("k");
        }
    };
    for (std::size_t i = 0; i < nNodes; ++i)
    {
        io::Archive nodeGroup = nodesGroup[fmt::format("{}", i)];
        if (nodeGroup.HasGroup("pbat.geometry.sdf.Sphere"))
            nodes[i] = Sphere<TScalar>{};
        else if (nodeGroup.HasGroup("pbat.geometry.sdf.Box"))
            nodes[i] = Box<TScalar>{};
        else if (nodeGroup.HasGroup("pbat.geometry.sdf.BoxFrame"))
            nodes[i] = BoxFrame<TScalar>{};
        else if (nodeGroup.HasGroup("pbat.geometry.sdf.Torus"))
            nodes[i] = Torus<TScalar>{};
        else if (nodeGroup.HasGroup("pbat.geometry.sdf.CappedTorus"))
            nodes[i] = CappedTorus<TScalar>{};
        else if (nodeGroup.HasGroup("pbat.geometry.sdf.Link"))
            nodes[i] = Link<TScalar>{};
        else if (nodeGroup.HasGroup("pbat.geometry.sdf.InfiniteCylinder"))
            nodes[i] = InfiniteCylinder<TScalar>{};
        else if (nodeGroup.HasGroup("pbat.geometry.sdf.Cone"))
            nodes[i] = Cone<TScalar>{};
        else if (nodeGroup.HasGroup("pbat.geometry.sdf.InfiniteCone"))
            nodes[i] = InfiniteCone<TScalar>{};
        else if (nodeGroup.HasGroup("pbat.geometry.sdf.Plane"))
            nodes[i] = Plane<TScalar>{};
        else if (nodeGroup.HasGroup("pbat.geometry.sdf.HexagonalPrism"))
            nodes[i] = HexagonalPrism<TScalar>{};
        else if (nodeGroup.HasGroup("pbat.geometry.sdf.Capsule"))
            nodes[i] = Capsule<TScalar>{};
        else if (nodeGroup.HasGroup("pbat.geometry.sdf.VerticalCapsule"))
            nodes[i] = VerticalCapsule<TScalar>{};
        else if (nodeGroup.HasGroup("pbat.geometry.sdf.CappedCylinder"))
            nodes[i] = CappedCylinder<TScalar>{};
        else if (nodeGroup.HasGroup("pbat.geometry.sdf.VerticalCappedCylinder"))
            nodes[i] = VerticalCappedCylinder<TScalar>{};
        else if (nodeGroup.HasGroup("pbat.geometry.sdf.RoundedCylinder"))
            nodes[i] = RoundedCylinder<TScalar>{};
        else if (nodeGroup.HasGroup("pbat.geometry.sdf.VerticalCappedCone"))
            nodes[i] = VerticalCappedCone<TScalar>{};
        else if (nodeGroup.HasGroup("pbat.geometry.sdf.CutHollowSphere"))
            nodes[i] = CutHollowSphere<TScalar>{};
        else if (nodeGroup.HasGroup("pbat.geometry.sdf.VerticalRoundCone"))
            nodes[i] = VerticalRoundCone<TScalar>{};
        else if (nodeGroup.HasGroup("pbat.geometry.sdf.Octahedron"))
            nodes[i] = Octahedron<TScalar>{};
        else if (nodeGroup.HasGroup("pbat.geometry.sdf.Pyramid"))
            nodes[i] = Pyramid<TScalar>{};
        else if (nodeGroup.HasGroup("pbat.geometry.sdf.Triangle"))
            nodes[i] = Triangle<TScalar>{};
        else if (nodeGroup.HasGroup("pbat.geometry.sdf.Quadrilateral"))
            nodes[i] = Quadrilateral<TScalar>{};
        else if (nodeGroup.HasGroup("pbat.geometry.sdf.Scale"))
            nodes[i] = Scale<TScalar>{};
        else if (nodeGroup.HasGroup("pbat.geometry.sdf.Elongate"))
            nodes[i] = Elongate<TScalar>{};
        else if (nodeGroup.HasGroup("pbat.geometry.sdf.Round"))
            nodes[i] = Round<TScalar>{};
        else if (nodeGroup.HasGroup("pbat.geometry.sdf.Onion"))
            nodes[i] = Onion<TScalar>{};
        else if (nodeGroup.HasGroup("pbat.geometry.sdf.Symmetrize"))
            nodes[i] = Symmetrize<TScalar>{};
        else if (nodeGroup.HasGroup("pbat.geometry.sdf.Repeat"))
            nodes[i] = Repeat<TScalar>{};
        else if (nodeGroup.HasGroup("pbat.geometry.sdf.Bump"))
            nodes[i] = Bump<TScalar>{};
        else if (nodeGroup.HasGroup("pbat.geometry.sdf.Twist"))
            nodes[i] = Twist<TScalar>{};
        else if (nodeGroup.HasGroup("pbat.geometry.sdf.Bend"))
            nodes[i] = Bend<TScalar>{};
        else if (nodeGroup.HasGroup("pbat.geometry.sdf.Union"))
            nodes[i] = Union<TScalar>{};
        else if (nodeGroup.HasGroup("pbat.geometry.sdf.Difference"))
            nodes[i] = Difference<TScalar>{};
        else if (nodeGroup.HasGroup("pbat.geometry.sdf.Intersection"))
            nodes[i] = Intersection<TScalar>{};
        else if (nodeGroup.HasGroup("pbat.geometry.sdf.ExclusiveOr"))
            nodes[i] = ExclusiveOr<TScalar>{};
        else if (nodeGroup.HasGroup("pbat.geometry.sdf.SmoothUnion"))
            nodes[i] = SmoothUnion<TScalar>{};
        else if (nodeGroup.HasGroup("pbat.geometry.sdf.SmoothDifference"))
            nodes[i] = SmoothDifference<TScalar>{};
        else if (nodeGroup.HasGroup("pbat.geometry.sdf.SmoothIntersection"))
            nodes[i] = SmoothIntersection<TScalar>{};
        else
            throw std::runtime_error(
                fmt::format("Forest::Deserialize: Unknown node type for node {}", i));
        std::visit(Visitor{nodeGroup, i}, nodes[i]);
    }
    transforms.resize(nNodes);
    io::Archive transformsGroup = group["transforms"];
    for (std::size_t i = 0; i < nNodes; ++i)
    {
        detail::forest::DeserializeMiniMatrix(
            fmt::format("R{}", i),
            transforms[i].R,
            transformsGroup);
        detail::forest::DeserializeMiniMatrix(
            fmt::format("t{}", i),
            transforms[i].t,
            transformsGroup);
    }
    roots   = group.ReadData<std::vector<int>>("roots");
    auto lc = group.ReadData<std::vector<int>>("lc");
    auto rc = group.ReadData<std::vector<int>>("rc");
    children.resize(nNodes);
    for (std::size_t i = 0; i < nNodes; ++i)
        children[i] = {lc[i], rc[i]};
}

} // namespace pbat::geometry::sdf

#endif // PBAT_GEOMETRY_SDF_FOREST_H
