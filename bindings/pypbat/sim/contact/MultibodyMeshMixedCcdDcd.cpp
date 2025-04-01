#include "MultibodyMeshMixedCcdDcd.h"

#include <pbat/sim/contact/MultibodyMeshMixedCcdDcd.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <tuple>
#include <vector>

namespace pbat::py::sim::contact {

void BindMultibodyMeshMixedCcdDcd([[maybe_unused]] pybind11::module& m)
{
    namespace pyb = pybind11;
    using pbat::sim::contact::MultibodyMeshMixedCcdDcd;
    pyb::class_<MultibodyMeshMixedCcdDcd>(m, "MultibodyMeshMixedCcdDcd")
        .def(
            pyb::init([](Eigen::Ref<Matrix<3, Eigen::Dynamic> const> const& X,
                         Eigen::Ref<IndexVectorX const> const& V,
                         Eigen::Ref<IndexMatrix<2, Eigen::Dynamic> const> const& E,
                         Eigen::Ref<IndexMatrix<3, Eigen::Dynamic> const> const& F,
                         Eigen::Ref<IndexMatrix<4, Eigen::Dynamic> const> const& T,
                         Eigen::Ref<IndexVectorX const> const& VP,
                         Eigen::Ref<IndexVectorX const> const& EP,
                         Eigen::Ref<IndexVectorX const> const& FP,
                         Eigen::Ref<IndexVectorX const> const& TP) {
                return MultibodyMeshMixedCcdDcd(X, V, E, F, T, VP, EP, FP, TP);
            }),
            pyb::arg("X"),
            pyb::arg("V").noconvert(),
            pyb::arg("E").noconvert(),
            pyb::arg("F").noconvert(),
            pyb::arg("T").noconvert(),
            pyb::arg("VP").noconvert(),
            pyb::arg("EP").noconvert(),
            pyb::arg("FP").noconvert(),
            pyb::arg("TP").noconvert(),
            "Construct a multibody mesh CCD system out of multiple meshes\n\n"
            "Args:\n"
            "    X (numpy.ndarray): `3 x |# verts|` mesh vertex positions\n"
            "    V (numpy.ndarray): `|# collision verts|` collision vertex array\n"
            "    E (numpy.ndarray): `2 x |# collision edges|` edge array\n"
            "    F (numpy.ndarray): `3 x |# collision triangles|` triangle array\n"
            "    T (numpy.ndarray): `4 x |# tetrahedra|` tetrahedron array\n"
            "    VP (numpy.ndarray): `|# meshes + 1|` prefix sum of vertex pointers\n"
            "    EP (numpy.ndarray): `|# meshes + 1|` prefix sum of edge pointers\n"
            "    FP (numpy.ndarray): `|# meshes + 1|` prefix sum of triangle pointers\n"
            "    TP (numpy.ndarray): `|# meshes + 1|` prefix sum of tetrahedron pointers\n")
        .def(
            "all_pairs",
            [](MultibodyMeshMixedCcdDcd& self,
               Eigen::Ref<MatrixX const> const& XT,
               Eigen::Ref<MatrixX const> const& X,
               Eigen::Ref<MatrixX const> const& XK) {
#include <pbat/warning/Push.h>
#include <pbat/warning/SignConversion.h>
                // We will store the results in these vectors
                std::vector<Index> VTv;
                std::vector<Index> VTf;
                std::vector<Index> EEei;
                std::vector<Index> EEej;
                VTv.reserve(XT.cols());
                VTf.reserve(XT.cols());
                EEei.reserve(XT.cols());
                EEej.reserve(XT.cols());
                // Detect DCD+CCD contact pairs
                self.UpdateActiveSet(
                    XT,
                    X,
                    XK,
                    [&](MultibodyMeshMixedCcdDcd::VertexTriangleContact c) {
                        VTv.push_back(c.i);
                        VTf.push_back(c.f);
                    },
                    [&](MultibodyMeshMixedCcdDcd::EdgeEdgeContact c) {
                        EEei.push_back(c.ei);
                        EEej.push_back(c.ej);
                    });
#include <pbat/warning/Pop.h>
                return std::make_tuple(VTv, VTf, EEei, EEej);
            },
            pyb::arg("XT"),
            pyb::arg("X"),
            pyb::arg("XK"),
            "Detect all vertex-triangle and edge-edge contact pairs\n\n"
            "Args:\n"
            "    XT (numpy.ndarray): `3 x |# verts|` mesh vertex positions at time t\n"
            "    X (numpy.ndarray): `3 x |# verts|` mesh vertex positions at time t+1\n"
            "    XK (numpy.ndarray): `3 x |# verts|` mesh vertex positions at current time\n"
            "Returns:\n"
            "    Tuple[List[int], List[int], List[int], List[int]]: (v,f,ei,ej) lists of "
            "vertex-triangle (v,f) and edge-edge "
            "(ei,ej) contact pairs")
        .def(
            "dcd_pairs",
            [](MultibodyMeshMixedCcdDcd& self, Eigen::Ref<MatrixX const> const& XK) {
#include <pbat/warning/Push.h>
#include <pbat/warning/SignConversion.h>
                // We will store the results in these vectors
                std::vector<Index> VTv;
                std::vector<Index> VTf;
                VTv.reserve(XK.cols());
                VTf.reserve(XK.cols());
                // Detect DCD contact pairs
                self.UpdateDcdActiveSet(XK, [&](MultibodyMeshMixedCcdDcd::VertexTriangleContact c) {
                    VTv.push_back(c.i);
                    VTf.push_back(c.f);
                });
#include <pbat/warning/Pop.h>
                return std::make_tuple(VTv, VTf);
            },
            pyb::arg("XK"),
            "Detect all vertex-triangle contact pairs using DCD\n\n"
            "Args:\n"
            "    XK (numpy.ndarray): `3 x |# verts|` mesh vertex positions at current time\n"
            "Returns:\n"
            "    Tuple[List[int], List[int]]: (v,f) lists of vertex-triangle contact pairs")
        .def_property_readonly(
            "body_aabbs",
            &MultibodyMeshMixedCcdDcd::GetBodyAabbs,
            "`2*kDims x |# bodies|` array of body AABBs")
        .def(
            "compute_vertex_aabbs",
            [](MultibodyMeshMixedCcdDcd& self, Eigen::Ref<MatrixX const> const& X) {
                self.ComputeVertexAabbs(X);
            },
            pyb::arg("X"),
            "Compute vertex AABBs for mesh vertex BVHs\n\n"
            "Args:\n"
            "    X (numpy.ndarray): `3 x |# verts|` mesh vertex positions")
        .def(
            "compute_vertex_aabbs",
            [](MultibodyMeshMixedCcdDcd& self,
               Eigen::Ref<MatrixX const> const& XT,
               Eigen::Ref<MatrixX const> const& X) { self.ComputeVertexAabbs(XT, X); },
            pyb::arg("XT"),
            pyb::arg("X"),
            "Compute vertex AABBs for mesh vertex BVHs\n\n"
            "Args:\n"
            "    XT (numpy.ndarray): `3 x |# verts|` mesh vertex positions at time t\n"
            "    X (numpy.ndarray): `3 x |# verts|` mesh vertex positions at time t+1")
        .def(
            "compute_edge_aabbs",
            [](MultibodyMeshMixedCcdDcd& self,
               Eigen::Ref<MatrixX const> const& XT,
               Eigen::Ref<MatrixX const> const& X) { self.ComputeEdgeAabbs(XT, X); },
            pyb::arg("XT"),
            pyb::arg("X"),
            "Compute edge AABBs for mesh edge BVHs\n\n"
            "Args:\n"
            "    XT (numpy.ndarray): `3 x |# verts|` mesh vertex positions at time t\n"
            "    X (numpy.ndarray): `3 x |# verts|` mesh vertex positions at time t+1")
        .def(
            "compute_triangle_aabbs",
            [](MultibodyMeshMixedCcdDcd& self, Eigen::Ref<MatrixX const> const& X) {
                self.ComputeTriangleAabbs(X);
            },
            pyb::arg("X"),
            "Compute triangle AABBs for mesh triangle BVHs\n\n"
            "Args:\n"
            "    X (numpy.ndarray): `3 x |# verts|` mesh vertex positions")
        .def(
            "compute_triangle_aabbs",
            [](MultibodyMeshMixedCcdDcd& self,
               Eigen::Ref<MatrixX const> const& XT,
               Eigen::Ref<MatrixX const> const& X) { self.ComputeTriangleAabbs(XT, X); },
            pyb::arg("XT"),
            pyb::arg("X"),
            "Compute triangle AABBs for mesh triangle BVHs\n\n"
            "Args:\n"
            "    XT (numpy.ndarray): `3 x |# verts|` mesh vertex positions at time t\n"
            "    X (numpy.ndarray): `3 x |# verts|` mesh vertex positions at time t+1")
        .def(
            "compute_tetrahedron_aabbs",
            [](MultibodyMeshMixedCcdDcd& self, Eigen::Ref<MatrixX const> const& X) {
                self.ComputeTetrahedronAabbs(X);
            },
            pyb::arg("X"),
            "Compute tetrahedron AABBs for mesh tetrahedron BVHs\n\n"
            "Args:\n"
            "    X (numpy.ndarray): `3 x |# verts|` mesh vertex positions")
        .def(
            "compute_body_aabbs",
            &MultibodyMeshMixedCcdDcd::ComputeBodyAabbs,
            "Compute body AABBs from mesh vertex BVHs")
        .def(
            "update_mesh_vertex_bvhs",
            &MultibodyMeshMixedCcdDcd::UpdateMeshVertexBvhs,
            "Recompute mesh vertex BVHs' bounding volumes")
        .def(
            "update_mesh_edge_bvhs",
            &MultibodyMeshMixedCcdDcd::UpdateMeshEdgeBvhs,
            "Recompute mesh edge BVHs' bounding volumes")
        .def(
            "update_mesh_triangle_bvhs",
            &MultibodyMeshMixedCcdDcd::UpdateMeshTriangleBvhs,
            "Recompute mesh triangle BVHs' bounding volumes")
        .def(
            "update_mesh_tetrahedron_bvhs",
            &MultibodyMeshMixedCcdDcd::UpdateMeshTetrahedronBvhs,
            "Recompute mesh tetrahedron BVHs' bounding volumes")
        .def(
            "recompute_body_bvh",
            &MultibodyMeshMixedCcdDcd::RecomputeBodyBvh,
            "Recompute body BVH tree and internal node bounding volumes");
}

} // namespace pbat::py::sim::contact
