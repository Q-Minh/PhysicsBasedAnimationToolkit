#include "MeshVertexTetrahedronDcd.h"

#include <pbat/sim/contact/MeshVertexTetrahedronDcd.h>
#include <pbat/sim/contact/MultibodyTetrahedralMeshSystem.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <vector>

namespace pbat::py::sim::contact {

void BindMeshVertexTetrahedronDcd(pybind11::module& m)
{
    namespace pyb = pybind11;
    using pbat::sim::contact::MeshVertexTetrahedronDcd;
    using IndexType  = MeshVertexTetrahedronDcd::IndexType;
    using ScalarType = MeshVertexTetrahedronDcd::ScalarType;
    pyb::class_<MeshVertexTetrahedronDcd>(m, "MeshVertexTetrahedronDcd")
        .def(
            pyb::init([](Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic>> X,
                         Eigen::Ref<Eigen::Matrix<IndexType, 4, Eigen::Dynamic>> T) {
                return MeshVertexTetrahedronDcd(std::move(X), std::move(T));
            }),
            pyb::arg("X"),
            pyb::arg("T"),
            "Construct a multibody tetrahedral mesh DCD system out of multiple meshes\n\n"
            "Args:\n"
            "    X (numpy.ndarray): `3 x |# verts|` mesh vertex positions\n"
            "    T (numpy.ndarray): `4 x |# tetrahedra|` tetrahedron array\n")
        .def(
            "update_active_set",
            [](MeshVertexTetrahedronDcd& self,
               Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
               Eigen::Ref<Eigen::Matrix<IndexType, 4, Eigen::Dynamic> const> const& T) {
                self.UpdateActiveSet(X, T);
            },
            pyb::arg("X"),
            pyb::arg("T"),
            "Update the active set of vertex-triangle contacts\n\n"
            "Args:\n"
            "    X (numpy.ndarray): `3 x |# verts|` mesh vertex positions\n"
            "    T (numpy.ndarray): `4 x |# tetrahedra|` tetrahedron array\n"
            "Returns:\n"
            "    None: ")
        .def(
            "vertex_triangle_contacts",
            [](MeshVertexTetrahedronDcd const& self) {
                auto const nCollisionVertices = self.MultibodySystem().NumContactVertices();
                Eigen::Matrix<IndexType, 2, Eigen::Dynamic> VFC(
                    2,
                    nCollisionVertices * MeshVertexTetrahedronDcd::kMaxVertexTriangleContacts);
                Eigen::Index k = 0;
                for (auto v = 0; v < nCollisionVertices; ++v)
                {
                    self.ForEachActiveVertexTriangleContact(
                        v,
                        [&](IndexType f, [[maybe_unused]] IndexType c) {
                            VFC(0, k) = v;
                            VFC(1, k) = f;
                            ++k;
                        });
                }
                return VFC.leftCols(k).eval();
            },
            "Get all vertex-triangle contacts\n\n"
            "Args:\n"
            "Returns:\n"
            "    numpy.ndarray: `2 x |# contacting triangles|` array of triangle indices")
        .def_property_readonly(
            "multibody_system",
            &MeshVertexTetrahedronDcd::MultibodySystem,
            "Get the multibody tetrahedral mesh system\n\n"
            "Returns:\n"
            "    MultibodyTetrahedralMeshSystem: the multibody tetrahedral mesh system");
}

} // namespace pbat::py::sim::contact