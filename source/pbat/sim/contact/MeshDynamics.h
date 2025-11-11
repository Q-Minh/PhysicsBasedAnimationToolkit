#ifndef PBAT_SIM_CONTACT_MESHDYNAMICS_H
#define PBAT_SIM_CONTACT_MESHDYNAMICS_H

#include "MultiMesh.h"
#include "OffsetGeometryContact.h"
#include "pbat/Aliases.h"
#include "pbat/geometry/sdf/Composite.h"
#include "pbat/geometry/sdf/Forest.h"

namespace pbat::sim::contact {

class MeshDynamics
{
  public:
    using ScalarType = Scalar;
    using IndexType  = Index;

  private:
    MultiMesh<IndexType> mMeshes; ///< Dynamic geometry
};

} // namespace pbat::sim::contact

#endif // PBAT_SIM_CONTACT_MESHDYNAMICS_H
