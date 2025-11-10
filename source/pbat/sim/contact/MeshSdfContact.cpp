#include "MeshSdfContact.h"

#include "pbat/profiling/Profiling.h"

#include <tbb/parallel_for.h>

namespace pbat::sim::contact {

MeshSdfContact::MeshSdfContact(Index nTriangles) : mTriangleSdfContacts(nTriangles)
{
    for (auto& contacts : mTriangleSdfContacts)
        contacts.reserve(1); // reserve space for 1 contact point per triangle
}

void MeshSdfContact::Serialize(io::Archive& archive) const
{
    io::Archive group = archive.GetOrCreateGroup("pbat.sim.contact.MeshSdfContact");
    group.WriteData("mTriangleSdfContacts", mTriangleSdfContacts);
}

void MeshSdfContact::Deserialize(io::Archive& archive)
{
    io::Archive group    = archive.GetOrCreateGroup("pbat.sim.contact.MeshSdfContact");
    mTriangleSdfContacts = group.ReadData<std::vector<std::vector<Eigen::Vector<ScalarType, 3>>>>(
        "mTriangleSdfContacts");
}

void MeshSdfContact::TriangleSdfContactDetection(
    Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
    Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
    geometry::sdf::Composite<ScalarType> const& sdf)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MeshSdfContact.TriangleSdfContactDetection");
    for (auto& contacts : mTriangleSdfContacts)
        contacts.clear();
    tbb::parallel_for(Eigen::Index(0), F.cols(), [&](Eigen::Index f) {
        
    });
}

} // namespace pbat::sim::contact
