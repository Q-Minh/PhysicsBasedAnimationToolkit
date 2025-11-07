#include "MultiMesh.h"

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/tuple.h>
#include <pbat/sim/contact/MultiMesh.h>

namespace pbat::py::sim::contact {

void BindMultiMesh(nanobind::module_& m)
{
    namespace nb    = nanobind;
    using IndexType = Index;

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
            pbat::sim::contact::BoundaryTriangulation(T, XCC, V, F, VP, FP);
            return std::make_tuple(V, F, VP, FP);
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
        "    Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]: (V, F, VP, FP).\n");

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
        "    (E, EP, GVHEp, GVHEadj, GHEF, EHE).\n");
}

} // namespace pbat::py::sim::contact
