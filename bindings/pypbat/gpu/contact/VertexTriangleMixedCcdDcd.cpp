#include "VertexTriangleMixedCcdDcd.h"

#include "pypbat/gpu/common/Buffer.h"

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/vector.h>
#include <pbat/gpu/contact/VertexTriangleMixedCcdDcd.h>

namespace pbat::py::gpu::contact {

void BindVertexTriangleMixedCcdDcd([[maybe_unused]] nanobind::module_& m)
{
#ifdef PBAT_USE_CUDA
    namespace nb = nanobind;
    using namespace pbat::gpu::contact;

    nb::class_<VertexTriangleMixedCcdDcd>(m, "VertexTriangleMixedCcdDcd")
        .def(
            nb::init<
                Eigen::Ref<GpuIndexVectorX const>,
                Eigen::Ref<GpuIndexVectorX const>,
                Eigen::Ref<GpuIndexMatrixX const>>(),
            nb::arg("B"),
            nb::arg("V"),
            nb::arg("F"),
            "Construct a triangle mesh collision detection system considering only vertex-triangle "
            "contacts\n\n"
            "Args:\n"
            "    B (np.ndarray): |#points| array mapping point indices to body indices\n"
            "    V (np.ndarray): |#collision verts| array of collision vertices (subset of "
            "points)\n"
            "    F (np.ndarray): 3x|#collision triangles| array of collision triangles")
        .def(
            "initialize_active_set",
            &VertexTriangleMixedCcdDcd::InitializeActiveSet,
            nb::arg("xt"),
            nb::arg("xtp1"),
            nb::arg("wmin"),
            nb::arg("wmax"),
            "Computes the initial active set\n\n"
            "Args:\n"
            "    xt (pbat.gpu.common.Buffer): 3x|#points| buffer of point positions at time t\n"
            "    xtp1 (pbat.gpu.common.Buffer): 3x|#points| buffer of predicted point positions at "
            "time t+1\n"
            "    wmin (np.ndarray): World bounding box min\n"
            "    wmax (np.ndarray): World bounding box max")
        .def(
            "update_active_set",
            &VertexTriangleMixedCcdDcd::UpdateActiveSet,
            nb::arg("x"),
            nb::arg("bComputeBoxes") = true,
            "Updates constraints involved with active vertices\n\n"
            "Args:\n"
            "    x (pbat.gpu.common.Buffer): 3x|#points| buffer of current point positions\n"
            "    bComputeBoxes (bool): If true, computes the bounding boxes of (non-swept) "
            "triangles")
        .def(
            "finalize_active_set",
            &VertexTriangleMixedCcdDcd::FinalizeActiveSet,
            nb::arg("x"),
            nb::arg("bComputeBoxes") = true,
            "Removes inactive vertices from the active set\n\n"
            "Args:\n"
            "    x (pbat.gpu.common.Buffer): 3x|#points| buffer of current point positions\n"
            "    bComputeBoxes (bool): If true, computes the bounding boxes of (non-swept) "
            "triangles")
        .def_prop_rw(
            "eps",
            nullptr,
            &VertexTriangleMixedCcdDcd::SetNearestNeighbourFloatingPointTolerance,
            "Floating point tolerance for nearest neighbour search")
        .def_prop_ro(
            "active_set",
            &VertexTriangleMixedCcdDcd::ActiveVertexTriangleConstraints,
            "Returns a 2x|#vertex-triangle constraints| matrix where each column is a "
            "vertex-triangle constraint pair")
        .def_prop_ro(
            "active_vertices",
            &VertexTriangleMixedCcdDcd::ActiveVertices,
            "Returns |#active verts| 1D array of active vertex indices into V")
        .def_prop_ro(
            "active_mask",
            &VertexTriangleMixedCcdDcd::ActiveMask,
            "Returns |#verts| 1D mask for active vertices");
#endif // PBAT_USE_CUDA
}

} // namespace pbat::py::gpu::contact