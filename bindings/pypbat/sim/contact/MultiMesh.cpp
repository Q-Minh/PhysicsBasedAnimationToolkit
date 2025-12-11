#include "MultiMesh.h"

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/tuple.h>
#include <pbat/sim/contact/MultiMesh.h>

namespace pbat::py::sim::contact {

void BindMultiMesh(nanobind::module_& m)
{
    namespace nb    = nanobind;
    using IndexType = Index;
    using MultiMesh = pbat::sim::contact::MultiMesh<IndexType>;

    nb::class_<MultiMesh>(m, "MultiMesh")
        .def(nb::init<>())
        .def(
            "__init__",
            [](MultiMesh* self,
               nb::DRef<Eigen::Matrix<IndexType, 4, Eigen::Dynamic> const> const& T,
               nb::DRef<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& XCC,
               Eigen::Index n_components) { new (self) MultiMesh(T, XCC, n_components); },
            nb::arg("T"),
            nb::arg("XCC"),
            nb::arg("n_components") = -1,
            "Construct a MultiMesh from a tetrahedral mesh and connected-component labels.\n\n"
            "Args:\n"
            "    T (numpy.ndarray): `4 x |# tetrahedra|` tetrahedral connectivity.\n"
            "    XCC (numpy.ndarray): `|# nodes|` node connected-component labels.\n"
            "    n_components (int): Optional number of components (-1 infers T.max()+1).\n")
        .def(
            "construct_from_tetrahedral_mesh",
            [](MultiMesh& self,
               nb::DRef<Eigen::Matrix<IndexType, 4, Eigen::Dynamic> const> const& T,
               nb::DRef<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& XCC,
               Eigen::Index n_components) {
                self.ConstructFromTetrahedralMesh(T, XCC, n_components);
            },
            nb::arg("T"),
            nb::arg("XCC"),
            nb::arg("n_components") = -1,
            "Construct a MultiMesh from a tetrahedral mesh and connected-component labels.\n\n"
            "Args:\n"
            "    T (numpy.ndarray): `4 x |# tetrahedra|` tetrahedral connectivity.\n"
            "    XCC (numpy.ndarray): `|# nodes|` node connected-component labels.\n"
            "    n_components (int): Optional number of components (-1 infers T.max()+1).\n")
        .def(
            "construct_from_triangle_mesh",
            [](MultiMesh& self,
               nb::DRef<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
               nb::DRef<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& XCC,
               Eigen::Index n_components) { self.ConstructFromTriangleMesh(F, XCC, n_components); },
            nb::arg("F"),
            nb::arg("XCC"),
            nb::arg("n_components") = -1,
            "Construct a MultiMesh from a triangle mesh and connected-component labels.\n\n"
            "Args:\n"
            "    F (numpy.ndarray): `3 x |# triangles|` triangle connectivity.\n"
            "    XCC (numpy.ndarray): `|# nodes|` node connected-component labels.\n"
            "    n_components (int): Optional number of components (-1 infers XCC.max()+1).\n")
        .def_ro("V", &MultiMesh::V, "`|# vertices| x 1` (surface) vertex indices")
        .def_ro("F", &MultiMesh::F, "`3 x |# faces|` face indices")
        .def_ro("E", &MultiMesh::E, "`2 x |# edges|` edge indices")
        .def_ro("VP", &MultiMesh::VP, "`|# components+1|` vertex prefix")
        .def_ro("FP", &MultiMesh::FP, "`|# components+1|` face prefix")
        .def_ro("EP", &MultiMesh::EP, "`|# components+1|` edge prefix")
        .def_ro("GVHEp", &MultiMesh::GVHEp, "`|# points+1|` point-to-half-edge prefix")
        .def_ro("GVHEadj", &MultiMesh::GVHEadj, "`|# half-edges|` point-to-half-edge adjacency")
        .def_ro("GHEF", &MultiMesh::GHEF, "`2 x |# half-edges|` half-edge to face adjacency")
        .def_ro("EHE", &MultiMesh::EHE, "`2 x |# edges|` edge to half-edge adjacency")
        .def_ro("GXV", &MultiMesh::GXV, "`|# points| x 1` point to vertex mapping");

    m.def(
        "boundary_triangulation",
        [](nb::DRef<Eigen::Matrix<IndexType, 4, Eigen::Dynamic> const> const& T,
           nb::DRef<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& XCC,
           IndexType nComponents) {
            if (nComponents < 0)
                nComponents = XCC.maxCoeff() + 1;
            Eigen::Vector<IndexType, Eigen::Dynamic> V;
            Eigen::Matrix<IndexType, 3, Eigen::Dynamic> F;
            Eigen::Vector<IndexType, Eigen::Dynamic> VP(nComponents + 1);
            Eigen::Vector<IndexType, Eigen::Dynamic> FP(nComponents + 1);
            Eigen::Vector<IndexType, Eigen::Dynamic> GXV;
            pbat::sim::contact::BoundaryTriangulation(T, XCC, V, F, VP, FP, GXV);
            return std::make_tuple(V, F, VP, FP, GXV);
        },
        nb::arg("T"),
        nb::arg("XCC"),
        nb::arg("n_components") = -1,
        "Compute boundary triangulation of a multi-mesh with connected component labeling.\n\n"
        "Args:\n"
        "    T (numpy.ndarray): `4 x |# tets|` tetrahedral connectivity.\n"
        "    XCC (numpy.ndarray): `|# nodes|` node connected component labels.\n"
        "    n_components (int): Number of connected components.\n"
        "Returns:\n"
        "    Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]: (V, "
        "F, VP, FP, GXV) where V is a `|# vertices| x 1` (surface) vertex indices array, F is a `3 "
        "x |# triangles|` triangle indices array, VP is a `|# components + 1| x 1` vertex prefix "
        "array, FP is a `|# components + 1| x 1` face prefix array, and GXV is a `|# points| x 1` "
        "point to vertex mapping.\n");

    m.def(
        "boundary_triangulation_edges",
        [](nb::DRef<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
           nb::DRef<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& XCC,
           IndexType nComponents) {
            if (nComponents < 0)
                nComponents = XCC.maxCoeff() + 1;
            Eigen::Matrix<IndexType, 2, Eigen::Dynamic> E;
            Eigen::Vector<IndexType, Eigen::Dynamic> EP(nComponents + 1);
            Eigen::Vector<IndexType, Eigen::Dynamic> GVHEp;
            Eigen::Vector<IndexType, Eigen::Dynamic> GVHEadj;
            Eigen::Matrix<IndexType, 2, Eigen::Dynamic> GHEF;
            Eigen::Matrix<IndexType, 2, Eigen::Dynamic> EHE;
            pbat::sim::contact::BoundaryTriangulationEdges(
                F,
                XCC,
                E,
                EP,
                GVHEp,
                GVHEadj,
                GHEF,
                EHE);
            return std::make_tuple(E, EP, GVHEp, GVHEadj, GHEF, EHE);
        },
        nb::arg("F"),
        nb::arg("XCC"),
        nb::arg("n_components"),
        "Compute boundary edges and adjacencies for a multi-mesh boundary triangulation.\n\n"
        "Args:\n"
        "    F (numpy.ndarray): `3 x |# faces|` boundary triangles.\n"
        "    XCC (numpy.ndarray): `|# nodes|` node connected component labels.\n"
        "    n_components (int): Number of connected components.\n"
        "Returns:\n"
        "    Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, "
        "numpy.ndarray]:\n"
        "    (E, EP, GVHEp, GVHEadj, GHEF, EHE), where E is a `2 x |# edges|` edge indices array, "
        "EP is a `|# connected components + 1| x 1` edge prefix array, GVHEp is a `|# points + 1| "
        "x 1` point to half-edge prefix array, GVHEadj is a `|# half edges| x 1` point to "
        "half-edge adjacency array, GHEF is a `2 x |# half edges|` half-edge to face adjacency "
        "array, and EHE is a `2 x |# edges|` edge to half-edge adjacency array.\n");
}

} // namespace pbat::py::sim::contact
