#include "MultibodyMeshMixedCcdDcd.h"

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <pbat/sim/contact/MultibodyMeshMixedCcdDcd.h>
#include <tuple>
#include <vector>

namespace pbat::py::sim::contact {

void BindMultibodyMeshMixedCcdDcd(nanobind::module_& m)
{
    namespace nb = nanobind;
    using pbat::sim::contact::MultibodyMeshMixedCcdDcd;
    nb::class_<MultibodyMeshMixedCcdDcd>(m, "MultibodyMeshMixedCcdDcd")
        .def(
            "__init__",
            [](Eigen::Ref<Matrix<3, Eigen::Dynamic> const> const& X,
               Eigen::Ref<IndexVectorX const> const& V,
               Eigen::Ref<IndexMatrix<2, Eigen::Dynamic> const> const& E,
               Eigen::Ref<IndexMatrix<3, Eigen::Dynamic> const> const& F,
               Eigen::Ref<IndexMatrix<4, Eigen::Dynamic> const> const& T,
               Eigen::Ref<IndexVectorX const> const& VP,
               Eigen::Ref<IndexVectorX const> const& EP,
               Eigen::Ref<IndexVectorX const> const& FP,
               Eigen::Ref<IndexVectorX const> const& TP) {
                return MultibodyMeshMixedCcdDcd(X, V, E, F, T, VP, EP, FP, TP);
            },
            nb::arg("X"),
            nb::arg("V").noconvert(),
            nb::arg("E").noconvert(),
            nb::arg("F").noconvert(),
            nb::arg("T").noconvert(),
            nb::arg("VP").noconvert(),
            nb::arg("EP").noconvert(),
            nb::arg("FP").noconvert(),
            nb::arg("TP").noconvert(),
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
            nb::arg("XT"),
            nb::arg("X"),
            nb::arg("XK"),
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
            nb::arg("XK"),
            "Detect all vertex-triangle contact pairs using DCD\n\n"
            "Args:\n"
            "    XK (numpy.ndarray): `3 x |# verts|` mesh vertex positions at current time\n"
            "Returns:\n"
            "    Tuple[List[int], List[int]]: (v,f) lists of vertex-triangle contact pairs")
        .def_prop_ro_static(
            "vertex_aabbs",
            &MultibodyMeshMixedCcdDcd::GetVertexAabbs,
            "`2*kDims x |# bodies|` array of vertex AABBs")
        .def_prop_ro_static(
            "edge_aabbs",
            &MultibodyMeshMixedCcdDcd::GetEdgeAabbs,
            "`2*kDims x |# bodies|` array of edge AABBs")
        .def_prop_ro_static(
            "triangle_aabbs",
            &MultibodyMeshMixedCcdDcd::GetTriangleAabbs,
            "`2*kDims x |# bodies|` array of triangle AABBs")
        .def_prop_ro_static(
            "tetrahedron_aabbs",
            &MultibodyMeshMixedCcdDcd::GetTetrahedronAabbs,
            "`2*kDims x |# bodies|` array of tetrahedron AABBs")
        .def_prop_ro_static(
            "body_aabbs",
            &MultibodyMeshMixedCcdDcd::GetBodyAabbs,
            "`2*kDims x |# bodies|` array of body AABBs")
        .def(
            "compute_vertex_aabbs",
            [](MultibodyMeshMixedCcdDcd& self, Eigen::Ref<MatrixX const> const& X) {
                self.ComputeVertexAabbs(X);
            },
            nb::arg("X"),
            "Compute vertex AABBs for mesh vertex BVHs\n\n"
            "Args:\n"
            "    X (numpy.ndarray): `3 x |# verts|` mesh vertex positions")
        .def(
            "compute_vertex_aabbs",
            [](MultibodyMeshMixedCcdDcd& self,
               Eigen::Ref<MatrixX const> const& XT,
               Eigen::Ref<MatrixX const> const& X) { self.ComputeVertexAabbs(XT, X); },
            nb::arg("XT"),
            nb::arg("X"),
            "Compute vertex AABBs for mesh vertex BVHs\n\n"
            "Args:\n"
            "    XT (numpy.ndarray): `3 x |# verts|` mesh vertex positions at time t\n"
            "    X (numpy.ndarray): `3 x |# verts|` mesh vertex positions at time t+1")
        .def(
            "compute_edge_aabbs",
            [](MultibodyMeshMixedCcdDcd& self,
               Eigen::Ref<MatrixX const> const& XT,
               Eigen::Ref<MatrixX const> const& X) { self.ComputeEdgeAabbs(XT, X); },
            nb::arg("XT"),
            nb::arg("X"),
            "Compute edge AABBs for mesh edge BVHs\n\n"
            "Args:\n"
            "    XT (numpy.ndarray): `3 x |# verts|` mesh vertex positions at time t\n"
            "    X (numpy.ndarray): `3 x |# verts|` mesh vertex positions at time t+1")
        .def(
            "compute_triangle_aabbs",
            [](MultibodyMeshMixedCcdDcd& self, Eigen::Ref<MatrixX const> const& X) {
                self.ComputeTriangleAabbs(X);
            },
            nb::arg("X"),
            "Compute triangle AABBs for mesh triangle BVHs\n\n"
            "Args:\n"
            "    X (numpy.ndarray): `3 x |# verts|` mesh vertex positions")
        .def(
            "compute_triangle_aabbs",
            [](MultibodyMeshMixedCcdDcd& self,
               Eigen::Ref<MatrixX const> const& XT,
               Eigen::Ref<MatrixX const> const& X) { self.ComputeTriangleAabbs(XT, X); },
            nb::arg("XT"),
            nb::arg("X"),
            "Compute triangle AABBs for mesh triangle BVHs\n\n"
            "Args:\n"
            "    XT (numpy.ndarray): `3 x |# verts|` mesh vertex positions at time t\n"
            "    X (numpy.ndarray): `3 x |# verts|` mesh vertex positions at time t+1")
        .def(
            "compute_tetrahedron_aabbs",
            [](MultibodyMeshMixedCcdDcd& self, Eigen::Ref<MatrixX const> const& X) {
                self.ComputeTetrahedronAabbs(X);
            },
            nb::arg("X"),
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
            "Recompute body BVH tree and internal node bounding volumes")
        .def_prop_ro_static(
            "body_pairs",
            [](MultibodyMeshMixedCcdDcd const& self) {
#include <pbat/warning/Push.h>
#include <pbat/warning/SignConversion.h>
                // We will store the results in these vectors
                auto const nBodies = self.BodyCount();
                std::vector<Index> oi;
                std::vector<Index> oj;
                oj.reserve(nBodies);
                oi.reserve(nBodies);
                // Detect DCD contact pairs
                self.ForEachBodyPair([&](Index i, Index j) {
                    oi.push_back(i);
                    oj.push_back(j);
                });
#include <pbat/warning/Pop.h>
                return std::make_tuple(oi, oj);
            },
            "Get the body pairs for the multibody mesh CCD system\n\n"
            "Returns:\n"
            "    Tuple[List[int], List[int]]: list of body pairs (oi,oj)")
        .def(
            "vertex_body_pairs",
            [](MultibodyMeshMixedCcdDcd& self, Eigen::Ref<MatrixX const> const& XK) {
#include <pbat/warning/Push.h>
#include <pbat/warning/SignConversion.h>
                // We will store the results in these vectors
                auto const nVertices = self.VertexCount();
                std::vector<Index> vi;
                std::vector<Index> oi;
                vi.reserve(nVertices);
                oi.reserve(nVertices);
                // Detect DCD contact pairs
                self.ForEachPenetratingVertex(XK, [&](Index i, Index o) {
                    vi.push_back(i);
                    oi.push_back(o);
                });
#include <pbat/warning/Pop.h>
                return std::make_tuple(vi, oi);
            },
            nb::arg("XK"),
            "Get the vertex-body pairs of DCD query\n\n"
            "Args:\n"
            "    XK (numpy.ndarray): `3 x |# verts|` mesh vertex positions at current time\n"
            "Returns:\n"
            "    Tuple[List[int], List[int]]: list of vertex-body pairs (v,o)");
}

} // namespace pbat::py::sim::contact
