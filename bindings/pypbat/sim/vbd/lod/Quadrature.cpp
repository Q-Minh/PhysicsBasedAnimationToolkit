#include "Quadrature.h"

#include "pypbat/fem/Mesh.h"

#include <pbat/sim/vbd/lod/Mesh.h>
#include <pbat/sim/vbd/lod/Quadrature.h>
#include <pybind11/eigen.h>

namespace pbat {
namespace py {
namespace sim {
namespace vbd {
namespace lod {

void BindQuadrature(pybind11::module& m)
{
    namespace pyb = pybind11;
    using namespace pbat::sim::vbd::lod;

    pyb::enum_<ECageQuadratureStrategy>(m, "CageQuadratureStrategy")
        .value("CageMesh", ECageQuadratureStrategy::CageMesh, "Use the cage mesh's quadrature rule")
        .value(
            "EmbeddedMesh",
            ECageQuadratureStrategy::EmbeddedMesh,
            "Use the embedded mesh's quadrature rule")
        .value(
            "PolynomialSubCellIntegration",
            ECageQuadratureStrategy::PolynomialSubCellIntegration,
            "Use moment fitting to compute the cage mesh quadrature rule given the embedded mesh's "
            "quadrature rule as right-hand side.")
        .export_values();

    pyb::class_<CageQuadratureParameters>(m, "CageQuadratureParameters")
        .def(pyb::init<>())
        .def("with_strategy", &CageQuadratureParameters::WithStrategy, pyb::arg("strategy"))
        .def(
            "with_cage_mesh_pts",
            &CageQuadratureParameters::WithCageMeshPointsOfOrder,
            pyb::arg("order"))
        .def(
            "with_patch_cell_pts",
            &CageQuadratureParameters::WithPatchCellPointsOfOrder,
            pyb::arg("order"))
        .def("with_patch_error", &CageQuadratureParameters::WithPatchError, pyb::arg("err"))
        .def_readwrite("strategy", &CageQuadratureParameters::eStrategy)
        .def_readwrite("cage_order", &CageQuadratureParameters::mCageMeshPointsOfOrder)
        .def_readwrite("patch_order", &CageQuadratureParameters::mPatchCellPointsOfOrder)
        .def_readwrite("patch_error", &CageQuadratureParameters::mPatchTetVolumeError);

    pyb::class_<CageQuadrature>(
        m,
        "CageQuadrature",
        "Quadrature rule for embedded mesh FEM simulation")
        .def(
            pyb::init([](pbat::py::fem::Mesh const& FM,
                         pbat::py::fem::Mesh const& CM,
                         CageQuadratureParameters const& params) {
                VolumeMesh const* FMraw = FM.Raw<VolumeMesh>();
                VolumeMesh const* CMraw = CM.Raw<VolumeMesh>();
                if (FMraw == nullptr or CMraw == nullptr)
                    throw std::invalid_argument(
                        "Requested underlying MeshType that this Mesh does not hold.");
                return CageQuadrature(*FMraw, *CMraw, params);
            }),
            pyb::arg("fine_mesh"),
            pyb::arg("cage_mesh"),
            pyb::arg("params") = CageQuadratureParameters{})
        .def_readwrite("Xg", &CageQuadrature::Xg, "3x|#quad.pts.| array of quadrature points")
        .def_readwrite("wg", &CageQuadrature::wg, "|#quad.pts.| array of quadrature weights")
        .def_readwrite(
            "sg",
            &CageQuadrature::sg,
            "|#quad.pts.| boolean mask indicating quad.pts. outside the embedded domain")
        .def_readwrite(
            "eg",
            &CageQuadrature::eg,
            "|#quad.pts.| array of cage elements containing corresponding to quad.pts.")
        .def_readwrite(
            "Ncg",
            &CageQuadrature::Ncg,
            "4x|#quad.pts.| array of cage element shape functions at quad.pts.")
        .def_readwrite(
            "GNcg",
            &CageQuadrature::GNcg,
            "4x|3*#quad.pts.| array of cage element shape function gradients at quad.pts.")
        .def_readwrite(
            "efg",
            &CageQuadrature::efg,
            "|#quad.pts.| array of fine mesh elements containing corresponding to quad.pts.")
        .def_readwrite(
            "Nfg",
            &CageQuadrature::Nfg,
            "4x|#quad.pts.| array of fine mesh element shape functions at quad.pts.")
        .def_readwrite(
            "GNfg",
            &CageQuadrature::GNfg,
            "4x|3*#quad.pts.| array of fine mesh element shape function gradients at quad.pts.")
        .def_readwrite("GVGp", &CageQuadrature::GVGp, "|#cage verts + 1| prefix")
        .def_readwrite(
            "GVGg",
            &CageQuadrature::GVGg,
            "|#quad.pts.| cage vertex-quad.pt. adjacencies")
        .def_readwrite(
            "GVGilocal",
            &CageQuadrature::GVGilocal,
            "|#quad.pts.| cage vertex local element index");

    pyb::enum_<ESurfaceQuadratureStrategy>(m, "ESurfaceQuadratureStrategy")
        .value(
            "EmbeddedVertexSinglePointQuadrature",
            ESurfaceQuadratureStrategy::EmbeddedVertexSinglePointQuadrature,
            "Distribute the per-face 1-pt quadrature rule onto boundary vertices")
        .export_values();

    pyb::class_<SurfaceQuadrature>(
        m,
        "SurfaceQuadrature",
        "Boundary quadrature for a linear tetrahedral mesh")
        .def(
            pyb::init([](pbat::py::fem::Mesh const& FM,
                         pbat::py::fem::Mesh const& CM,
                         ESurfaceQuadratureStrategy eStrategy) {
                VolumeMesh const* FMraw = FM.Raw<VolumeMesh>();
                VolumeMesh const* CMraw = CM.Raw<VolumeMesh>();
                if (FMraw == nullptr or CMraw == nullptr)
                    throw std::invalid_argument(
                        "Requested underlying MeshType that this Mesh does not hold.");
                return SurfaceQuadrature(*FMraw, *CMraw, eStrategy);
            }),
            pyb::arg("fine_mesh"),
            pyb::arg("cage_mesh"),
            pyb::arg("strategy"))
        .def_readwrite("Xg", &SurfaceQuadrature::Xg)
        .def_readwrite("wg", &SurfaceQuadrature::wg)
        .def_readwrite("eg", &SurfaceQuadrature::eg)
        .def_readwrite("GVGp", &SurfaceQuadrature::GVGp)
        .def_readwrite("GVGg", &SurfaceQuadrature::GVGg)
        .def_readwrite("GVGilocal", &SurfaceQuadrature::GVGilocal);

    pyb::class_<DirichletQuadrature>(
        m,
        "DirichletQuadrature",
        "Quadrature rule for Dirichlet conditions")
        .def(
            pyb::init([](pbat::py::fem::Mesh const& FM,
                         pbat::py::fem::Mesh const& CM,
                         Eigen::Ref<VectorX const> const& m,
                         Eigen::Ref<IndexVectorX const> const& dbcs) {
                VolumeMesh const* FMraw = FM.Raw<VolumeMesh>();
                VolumeMesh const* CMraw = CM.Raw<VolumeMesh>();
                if (FMraw == nullptr or CMraw == nullptr)
                    throw std::invalid_argument(
                        "Requested underlying MeshType that this Mesh does not hold.");
                return DirichletQuadrature(*FMraw, *CMraw, m, dbcs);
            }),
            pyb::arg("fine_mesh"),
            pyb::arg("cage_mesh"),
            pyb::arg("mass"),
            pyb::arg("dbc"))
        .def_readwrite("Xg", &DirichletQuadrature::Xg)
        .def_readwrite("wg", &DirichletQuadrature::wg)
        .def_readwrite("eg", &DirichletQuadrature::eg)
        .def_readwrite("Ncg", &DirichletQuadrature::Ncg)
        .def_readwrite("GVGp", &DirichletQuadrature::GVGp)
        .def_readwrite("GVGg", &DirichletQuadrature::GVGg)
        .def_readwrite("GVGilocal", &DirichletQuadrature::GVGilocal);
}

} // namespace lod
} // namespace vbd
} // namespace sim
} // namespace py
} // namespace pbat