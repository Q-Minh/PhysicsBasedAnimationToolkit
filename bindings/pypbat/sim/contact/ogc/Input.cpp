#include "Input.h"

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/optional.h>
#include <pbat/sim/contact/MultiMesh.h>
#include <pbat/sim/contact/ogc/Input.h>

namespace pbat::py::sim::contact::ogc {

template <typename TScalar, typename TIndex>
struct InputStorage
{
    pbat::sim::contact::MultiMesh<TIndex> mDynamicMesh;
    pbat::sim::contact::MultiMesh<TIndex> mStaticMesh;
    Eigen::Matrix<TScalar, 3, Eigen::Dynamic> Xdynamic;
    Eigen::Matrix<TScalar, 3, Eigen::Dynamic> Xstatic;
    pbat::sim::contact::ogc::Input<TScalar, TIndex> ToInput()
    {
        pbat::sim::contact::ogc::Input<TScalar, TIndex> input{};
        input.WithDynamicGeometry(
            Xdynamic,
            mDynamicMesh.V,
            mDynamicMesh.F,
            mDynamicMesh.E,
            mDynamicMesh.VP,
            mDynamicMesh.FP,
            mDynamicMesh.EP,
            mDynamicMesh.GVHEp,
            mDynamicMesh.GVHEadj,
            mDynamicMesh.GHEF,
            mDynamicMesh.EHE);
        if (Xstatic.size() > 0)
        {
            input.WithStaticGeometry(
                Xstatic,
                mStaticMesh.E,
                mStaticMesh.F,
                mStaticMesh.GVHEp,
                mStaticMesh.GVHEadj,
                mStaticMesh.GHEF,
                mStaticMesh.EHE);
        }
        return input;
    }
};

void BindInput(nanobind::module_& m)
{
    namespace nb     = nanobind;
    using ScalarType = Scalar;
    using IndexType  = Index;
    using InputType  = pbat::sim::contact::ogc::Input<ScalarType, IndexType>;

    nb::class_<InputStorage<ScalarType, IndexType>>(m, "InputStorage")
        .def(nb::init<>())
        .def("to_input", &InputStorage<ScalarType, IndexType>::ToInput)
        .def_rw("dynamic_mesh", &InputStorage<ScalarType, IndexType>::mDynamicMesh)
        .def_rw("static_mesh", &InputStorage<ScalarType, IndexType>::mStaticMesh)
        .def_rw("Xdynamic", &InputStorage<ScalarType, IndexType>::Xdynamic)
        .def_rw("Xstatic", &InputStorage<ScalarType, IndexType>::Xstatic);

    nb::class_<InputType>(m, "Input")
        .def(nb::init<>(), "Construct an empty OGC input data structure.")
        .def(
            "with_dynamic_geometry",
            [](InputType& self,
               nb::DRef<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
               nb::DRef<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& V,
               nb::DRef<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
               nb::DRef<Eigen::Matrix<IndexType, 2, Eigen::Dynamic> const> const& E,
               nb::DRef<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& VP,
               nb::DRef<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& FP,
               nb::DRef<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& EP,
               nb::DRef<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& GVHEp,
               nb::DRef<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& GVHEadj,
               nb::DRef<Eigen::Matrix<IndexType, 2, Eigen::Dynamic> const> const& GHEF,
               nb::DRef<Eigen::Matrix<IndexType, 2, Eigen::Dynamic> const> const& EHE) {
                return self.WithDynamicGeometry(X, V, F, E, VP, FP, EP, GVHEp, GVHEadj, GHEF, EHE);
            },
            nb::arg("X").noconvert(),
            nb::arg("V").noconvert(),
            nb::arg("F").noconvert(),
            nb::arg("E").noconvert(),
            nb::arg("VP").noconvert(),
            nb::arg("FP").noconvert(),
            nb::arg("EP").noconvert(),
            nb::arg("GVHEp").noconvert(),
            nb::arg("GVHEadj").noconvert(),
            nb::arg("GHEF").noconvert(),
            nb::arg("EHE").noconvert(),
            nb::rv_policy::reference_internal,
            "Borrow dynamic geometry data.\n\n"
            "Args:\n"
            "    X (numpy.ndarray): `3 x |# points|` dynamic vertex positions.\n"
            "    V (numpy.ndarray): `|# verts|` vertex indices into X.\n"
            "    F (numpy.ndarray): `3 x |# facets|` face indices into X.\n"
            "    E (numpy.ndarray): `2 x |# edges|` edge indices into X.\n"
            "    VP (numpy.ndarray): `|# bodies + 1|` prefix sum of vertex counts per body.\n"
            "    FP (numpy.ndarray): `|# bodies + 1|` prefix sum of face counts per body.\n"
            "    EP (numpy.ndarray): `|# bodies + 1|` prefix sum of edge counts per body.\n"
            "    GVHEp (numpy.ndarray): `|# points + 1|` point to half-edge prefix.\n"
            "    GVHEadj (numpy.ndarray): `|# half edges|` point to half-edge adjacency.\n"
            "    GHEF (numpy.ndarray): `2 x |# half edges|` half-edge to face adjacency.\n"
            "    EHE (numpy.ndarray): `2 x |# edges|` half-edge indices for edges.\n\n"
            "Returns:\n"
            "    Input: Reference to this input object.")
        .def(
            "with_static_geometry",
            [](InputType& self,
               nb::DRef<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& Venv,
               nb::DRef<Eigen::Matrix<IndexType, 2, Eigen::Dynamic> const> const& Eenv,
               nb::DRef<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& Fenv,
               nb::DRef<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& GVHEp,
               nb::DRef<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& GVHEadj,
               nb::DRef<Eigen::Matrix<IndexType, 2, Eigen::Dynamic> const> const& GHEF,
               nb::DRef<Eigen::Matrix<IndexType, 2, Eigen::Dynamic> const> const& EHE) {
                return self.WithStaticGeometry(Venv, Eenv, Fenv, GVHEp, GVHEadj, GHEF, EHE);
            },
            nb::arg("Venv").noconvert(),
            nb::arg("Eenv").noconvert(),
            nb::arg("Fenv").noconvert(),
            nb::arg("GVHEp").noconvert(),
            nb::arg("GVHEadj").noconvert(),
            nb::arg("GHEF").noconvert(),
            nb::arg("EHE").noconvert(),
            nb::rv_policy::reference_internal,
            "Borrow static geometry data.\n\n"
            "Args:\n"
            "    Venv (numpy.ndarray): `3 x |# env. verts|` static vertex positions.\n"
            "    Eenv (numpy.ndarray): `2 x |# env. edges|` static edge indices into Venv.\n"
            "    Fenv (numpy.ndarray): `3 x |# env. faces|` static face indices into Venv.\n"
            "    GVHEp (numpy.ndarray): `|# env. verts + 1|` point to half-edge prefix.\n"
            "    GVHEadj (numpy.ndarray): `|# env. half edges|` point to half-edge adjacency.\n"
            "    GHEF (numpy.ndarray): `2 x |# env. half edges|` half-edge to face adjacency.\n"
            "    EHE (numpy.ndarray): `2 x |# env. edges|` half-edge indices for edges.\n\n"
            "Returns:\n"
            "    Input: Reference to this input object.")
        .def(
            "construct",
            &InputType::Construct,
            nb::rv_policy::reference_internal,
            "Validate construction of the input data.\n\n"
            "Returns:\n"
            "    Input: Reference to this input object.\n\n"
            "Raises:\n"
            "    RuntimeError: If the input data is invalid.")
        .def_prop_ro(
            "has_dynamic_geometry",
            &InputType::HasDynamicGeometry,
            "(bool) Whether dynamic geometry is provided.")
        .def_prop_ro(
            "has_static_geometry",
            &InputType::HasStaticGeometry,
            "(bool) Whether static geometry is provided.")
        .def(
            "dynamic_vertex",
            &InputType::DynamicVertex,
            nb::arg("b"),
            nb::arg("v"),
            "Get (global) dynamic vertex index of body b given its local vertex index v.\n\n"
            "Args:\n"
            "    b (int): Body index.\n"
            "    v (int): (Local) vertex index.\n\n"
            "Returns:\n"
            "    int: Dynamic vertex index.")
        .def(
            "dynamic_edge",
            &InputType::DynamicEdge,
            nb::arg("b"),
            nb::arg("e"),
            "Get (global) dynamic edge index of body b given its local edge index e.\n\n"
            "Args:\n"
            "    b (int): Body index.\n"
            "    e (int): (Local) edge index.\n\n"
            "Returns:\n"
            "    int: Dynamic edge index.")
        .def(
            "dynamic_facet",
            &InputType::DynamicFacet,
            nb::arg("b"),
            nb::arg("f"),
            "Get (global) dynamic facet index of body b given its local facet index f.\n\n"
            "Args:\n"
            "    b (int): Body index.\n"
            "    f (int): (Local) facet index.\n\n"
            "Returns:\n"
            "    int: Dynamic facet index.")
        .def(
            "num_bodies",
            &InputType::NumBodies,
            "Get number of bodies.\n\n"
            "Returns:\n"
            "    int: Number of bodies.")
        .def(
            "num_vertices",
            [](InputType const& self, IndexType b) {
                return (b >= 0) ? self.NumVertices(b) : self.NumVertices();
            },
            nb::arg("b"),
            "Get number of vertices for a dynamic body if `b` is non-negative, otherwise get total "
            "number of vertices.\n\n"
            "Args:\n"
            "    b (int): Body index.\n\n"
            "Returns:\n"
            "    int: Number of vertices for the body.")
        .def(
            "num_edges",
            [](InputType const& self, IndexType b) {
                return (b >= 0) ? self.NumEdges(b) : self.NumEdges();
            },
            nb::arg("b"),
            "Get number of edges for a dynamic body if `b` is non-negative, otherwise get total "
            "number of edges.\n\n"
            "Args:\n"
            "    b (int): Body index.\n\n"
            "Returns:\n"
            "    int: Number of edges for the body.")
        .def(
            "num_facets",
            [](InputType const& self, IndexType b) {
                return (b >= 0) ? self.NumFacets(b) : self.NumFacets();
            },
            nb::arg("b"),
            "Get number of facets for a dynamic body if `b` is non-negative, otherwise get total "
            "number of facets.\n\n"
            "Args:\n"
            "    b (int): Body index.\n\n"
            "Returns:\n"
            "    int: Number of facets for the body.")
        .def(
            "num_static_vertices",
            &InputType::NumStaticVertices,
            "Get number of static vertices.\n\n"
            "Returns:\n"
            "    int: Number of static vertices.")
        .def(
            "num_static_edges",
            &InputType::NumStaticEdges,
            "Get number of static edges.\n\n"
            "Returns:\n"
            "    int: Number of static edges.")
        .def(
            "num_static_facets",
            &InputType::NumStaticFacets,
            "Get number of static facets.\n\n"
            "Returns:\n"
            "    int: Number of static facets.")
        .def_ro("X", &InputType::X, "(numpy.ndarray) 3 x |# points| dynamic points.")
        .def_ro("V", &InputType::V, "(numpy.ndarray) |# verts| dynamic vertex indices into X.")
        .def_ro("F", &InputType::F, "(numpy.ndarray) 3 x |# facets| dynamic facet indices into X.")
        .def_ro("E", &InputType::E, "(numpy.ndarray) 2 x |# edges| dynamic edge indices into X.")
        .def_ro("VP", &InputType::VP, "(numpy.ndarray) |# bodies + 1| dynamic vertex prefix.")
        .def_ro("FP", &InputType::FP, "(numpy.ndarray) |# bodies + 1| dynamic face prefix.")
        .def_ro("EP", &InputType::EP, "(numpy.ndarray) |# bodies + 1| dynamic edge prefix.")
        .def_ro(
            "GVHEp",
            &InputType::GVHEp,
            "(numpy.ndarray) |# points + 1| point to half-edge prefix.")
        .def_ro(
            "GVHEadj",
            &InputType::GVHEadj,
            "(numpy.ndarray) |# half edges| point to half-edge adjacency.")
        .def_ro(
            "GHEF",
            &InputType::GHEF,
            "(numpy.ndarray) 2 x |# half edges| half-edge to face adjacency.")
        .def_ro(
            "EHE",
            &InputType::EHE,
            "(numpy.ndarray) 2 x |# edges| half-edge indices for edges.")
        .def_ro(
            "Venv",
            &InputType::Venv,
            "(numpy.ndarray) 3 x |# env. verts| static environment points.")
        .def_ro(
            "Eenv",
            &InputType::Eenv,
            "(numpy.ndarray) 2 x |# env. edges| static environment edge indices into Venv.")
        .def_ro(
            "Fenv",
            &InputType::Fenv,
            "(numpy.ndarray) 3 x |# env. faces| static environment facet indices into Venv.")
        .def_ro(
            "GVHEenvp",
            &InputType::GVHEenvp,
            "(numpy.ndarray) |# env. verts + 1| point to half-edge prefix.")
        .def_ro(
            "GVHEenvadj",
            &InputType::GVHEenvadj,
            "(numpy.ndarray) |# env. half edges| point to half-edge adjacency.")
        .def_ro(
            "GHEFenv",
            &InputType::GHEFenv,
            "(numpy.ndarray) 2 x |# env. half edges| half-edge to face adjacency.")
        .def_ro(
            "EHEenv",
            &InputType::EHEenv,
            "(numpy.ndarray) 2 x |# env. edges| half-edge indices for edges.");
}

} // namespace pbat::py::sim::contact::ogc