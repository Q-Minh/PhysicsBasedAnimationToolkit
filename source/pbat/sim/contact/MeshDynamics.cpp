#include "MeshDynamics.h"

namespace pbat::sim::contact {

} // namespace pbat::sim::contact

#include "pbat/geometry/Device.h"
#include "pbat/geometry/model/Cube.h"

#include <doctest/doctest.h>

TEST_CASE("[sim][contact] MeshDynamics")
{
    using namespace pbat;
    using ScalarType = Scalar;
    using IndexType  = Index;

    // Arrange: Create a simple cube tetrahedral mesh
    auto const [X, T] = geometry::model::Cube();
    // Create MultiMesh from tetrahedral mesh
    auto XCC                 = IndexVectorX::Zero(X.cols());
    Eigen::Index nComponents = 1;
    sim::contact::MultiMesh<IndexType> multiMesh(T.bottomRows<4>(), XCC, nComponents);
    // Create geometry device for acceleration structures
    geometry::DeviceConfig deviceConfig;
    geometry::Device device(deviceConfig);

    // Create MeshDynamics with default parameters
    sim::contact::MeshDynamics<ScalarType, IndexType>::Params params;
    params.mOgcParams.WithRadii(0.01, 0.02)
        .WithDynamicBuildQuality(
            pbat::sim::contact::ogc::EBuildQuality::Low,
            pbat::sim::contact::ogc::EBuildQuality::Low)
        .WithMaxContactEstimates(16, 16, 16)
        .Construct();
    params.WithFrictionalContact(0.5, 1e-3).WithNormalContact(1e5);

    sim::contact::MeshDynamics<ScalarType, IndexType> meshDynamics(params);

    // Act: Set dynamic geometry
    meshDynamics.SetDynamicGeometry(X, std::move(multiMesh));
    CHECK_NOTHROW(meshDynamics.Initialize(device));

    // Assert: Verify initialization succeeded
    // The device should be valid
    CHECK(static_cast<bool>(device));
}