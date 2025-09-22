#include "Forest.h"

#include "pbat/io/Archive.h"

#include <cmath>
#include <filesystem>

namespace pbat::geometry::sdf {
} // namespace pbat::geometry::sdf

#include <doctest/doctest.h>

TEST_CASE("[geometry][sdf] Forest")
{
    using namespace pbat::geometry::sdf;
    using ScalarType = double;

    SUBCASE("Serialize and deserialize empty Forest")
    {
        // Create an empty forest
        Forest<ScalarType> originalForest;

        // Create temporary file for testing
        std::filesystem::path testFile =
            std::filesystem::temp_directory_path() / "test_forest_empty.h5";

        // Serialize the forest
        {
            pbat::io::Archive archive(testFile, HighFive::File::Truncate);
            originalForest.Serialize(archive);
        }

        // Deserialize the forest
        Forest<ScalarType> deserializedForest;
        {
            pbat::io::Archive archive(testFile, HighFive::File::ReadOnly);
            deserializedForest.Deserialize(archive);
        }

        // Check that they match
        CHECK(deserializedForest.nodes.size() == originalForest.nodes.size());
        CHECK(deserializedForest.transforms.size() == originalForest.transforms.size());
        CHECK(deserializedForest.roots.size() == originalForest.roots.size());
        CHECK(deserializedForest.children.size() == originalForest.children.size());

        // Clean up
        std::filesystem::remove(testFile);
    }
    SUBCASE("Serialize and deserialize Forest with various primitives")
    {
        // Create a forest with various SDF primitives
        Forest<ScalarType> originalForest;

        // Add some primitive nodes
        originalForest.nodes.push_back(Sphere<ScalarType>{2.5}); // radius = 2.5
        originalForest.nodes.push_back(
            Box<ScalarType>{Vec3<ScalarType>{1.0, 2.0, 0.5}}); // half extents
        originalForest.nodes.push_back(
            BoxFrame<ScalarType>{Vec3<ScalarType>{1.5, 1.0, 2.0}, 0.1}); // half extents, thickness
        originalForest.nodes.push_back(
            Capsule<ScalarType>{
                Vec3<ScalarType>{0.0, 0.0, 0.0}, // point a
                Vec3<ScalarType>{1.0, 2.0, 3.0}, // point b
                0.5                              // radius
            });
        originalForest.nodes.push_back(VerticalCapsule<ScalarType>{1.0, 0.3}); // height, radius
        originalForest.nodes.push_back(Octahedron<ScalarType>{1.2});           // size
        originalForest.nodes.push_back(
            Triangle<ScalarType>{
                Vec3<ScalarType>{0.0, 0.0, 0.0}, // vertex a
                Vec3<ScalarType>{1.0, 0.0, 0.0}, // vertex b
                Vec3<ScalarType>{0.5, 1.0, 0.0}  // vertex c
            });

        // Add transforms for each node
        for (std::size_t i = 0; i < originalForest.nodes.size(); ++i)
        {
            Transform<ScalarType> transform = Transform<ScalarType>::Identity();
            transform.t(0)                  = static_cast<ScalarType>(i);
            transform.t(1)                  = static_cast<ScalarType>(i * 0.5);
            transform.t(2)                  = static_cast<ScalarType>(i * 2);
            originalForest.transforms.push_back(transform);
        }

        // Set up tree structure
        originalForest.roots = {0, 3}; // Two root nodes
        originalForest.children.resize(originalForest.nodes.size());
        for (std::size_t i = 0; i < originalForest.children.size(); ++i)
        {
            if (i == 0)
            {
                originalForest.children[i] = {1, 2}; // Node 0 has children 1 and 2
            }
            else if (i == 3)
            {
                originalForest.children[i] = {4, 5}; // Node 3 has children 4 and 5
            }
            else
            {
                originalForest.children[i] = {-1, -1}; // Leaf nodes have no children
            }
        }

        // Create temporary file for testing
        std::filesystem::path testFile =
            std::filesystem::temp_directory_path() / "test_forest_full.h5";

        // Serialize the forest
        {
            pbat::io::Archive archive(testFile, HighFive::File::Truncate);
            originalForest.Serialize(archive);
        }

        // Deserialize the forest
        Forest<ScalarType> deserializedForest;
        {
            pbat::io::Archive archive(testFile, HighFive::File::ReadOnly);
            deserializedForest.Deserialize(archive);
        }

        // Check basic structure
        CHECK(deserializedForest.nodes.size() == originalForest.nodes.size());
        CHECK(deserializedForest.transforms.size() == originalForest.transforms.size());
        CHECK(deserializedForest.roots.size() == originalForest.roots.size());
        CHECK(deserializedForest.children.size() == originalForest.children.size());

        // Check roots
        for (std::size_t i = 0; i < originalForest.roots.size(); ++i)
        {
            CHECK(deserializedForest.roots[i] == originalForest.roots[i]);
        }

        // Check children
        for (std::size_t i = 0; i < originalForest.children.size(); ++i)
        {
            CHECK(deserializedForest.children[i].first == originalForest.children[i].first);
            CHECK(deserializedForest.children[i].second == originalForest.children[i].second);
        }

        // Check transforms
        for (std::size_t i = 0; i < originalForest.transforms.size(); ++i)
        {
            for (int j = 0; j < 9; ++j)
            {
                CHECK(
                    std::abs(
                        deserializedForest.transforms[i].R[j] - originalForest.transforms[i].R[j]) <
                    1e-10);
            }
            for (int j = 0; j < 3; ++j)
            {
                CHECK(
                    std::abs(
                        deserializedForest.transforms[i].t[j] - originalForest.transforms[i].t[j]) <
                    1e-10);
            }
        }

        // Check node contents using visitors
        for (std::size_t i = 0; i < originalForest.nodes.size(); ++i)
        {
            std::visit(
                [&](auto&& originalNode) {
                    using NodeType = std::decay_t<decltype(originalNode)>;

                    // Get the deserialized node and check it's the same type
                    auto* deserializedNodePtr = std::get_if<NodeType>(&deserializedForest.nodes[i]);
                    CHECK(deserializedNodePtr != nullptr);

                    if (deserializedNodePtr)
                    {
                        auto& deserializedNode = *deserializedNodePtr;

                        if constexpr (std::is_same_v<NodeType, Sphere<ScalarType>>)
                        {
                            CHECK(std::abs(deserializedNode.R - originalNode.R) < 1e-10);
                        }
                        else if constexpr (std::is_same_v<NodeType, Box<ScalarType>>)
                        {
                            for (int j = 0; j < 3; ++j)
                            {
                                CHECK(
                                    std::abs(deserializedNode.he[j] - originalNode.he[j]) < 1e-10);
                            }
                        }
                        else if constexpr (std::is_same_v<NodeType, BoxFrame<ScalarType>>)
                        {
                            for (int j = 0; j < 3; ++j)
                            {
                                CHECK(
                                    std::abs(deserializedNode.he[j] - originalNode.he[j]) < 1e-10);
                            }
                            CHECK(std::abs(deserializedNode.t - originalNode.t) < 1e-10);
                        }
                        else if constexpr (std::is_same_v<NodeType, Capsule<ScalarType>>)
                        {
                            for (int j = 0; j < 3; ++j)
                            {
                                CHECK(std::abs(deserializedNode.a[j] - originalNode.a[j]) < 1e-10);
                                CHECK(std::abs(deserializedNode.b[j] - originalNode.b[j]) < 1e-10);
                            }
                            CHECK(std::abs(deserializedNode.r - originalNode.r) < 1e-10);
                        }
                        else if constexpr (std::is_same_v<NodeType, VerticalCapsule<ScalarType>>)
                        {
                            CHECK(std::abs(deserializedNode.h - originalNode.h) < 1e-10);
                            CHECK(std::abs(deserializedNode.r - originalNode.r) < 1e-10);
                        }
                        else if constexpr (std::is_same_v<NodeType, Octahedron<ScalarType>>)
                        {
                            CHECK(std::abs(deserializedNode.s - originalNode.s) < 1e-10);
                        }
                        else if constexpr (std::is_same_v<NodeType, Triangle<ScalarType>>)
                        {
                            for (int j = 0; j < 3; ++j)
                            {
                                CHECK(std::abs(deserializedNode.a[j] - originalNode.a[j]) < 1e-10);
                                CHECK(std::abs(deserializedNode.b[j] - originalNode.b[j]) < 1e-10);
                                CHECK(std::abs(deserializedNode.c[j] - originalNode.c[j]) < 1e-10);
                            }
                        }
                    }
                },
                originalForest.nodes[i]);
        }

        // Clean up
        std::filesystem::remove(testFile);
    }
}