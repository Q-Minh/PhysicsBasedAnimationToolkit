#include "VertexTriangleMixedCcdDcd.h"

#include "pypbat/gpu/common/Buffer.h"

#include <pbat/gpu/contact/VertexTriangleMixedCcdDcd.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace pbat::py::gpu::contact {

void BindVertexTriangleMixedCcdDcd(pybind11::module& m)
{
    namespace pyb = pybind11;
    using namespace pbat::gpu::contact;

    pyb::class_<VertexTriangleMixedCcdDcd>(m, "VertexTriangleMixedCcdDcd")
        .def(
            pyb::init<
                Eigen::Ref<GpuIndexVectorX const>,
                Eigen::Ref<GpuIndexVectorX const>,
                Eigen::Ref<GpuIndexMatrixX const>>(),
            pyb::arg("B"),
            pyb::arg("V"),
            pyb::arg("F"),
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
            pyb::arg("xt"),
            pyb::arg("xtp1"),
            pyb::arg("wmin"),
            pyb::arg("wmax"),
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
            pyb::arg("x"),
            pyb::arg("bComputeBoxes") = true,
            "Updates constraints involved with active vertices\n\n"
            "Args:\n"
            "    x (pbat.gpu.common.Buffer): 3x|#points| buffer of current point positions\n"
            "    bComputeBoxes (bool): If true, computes the bounding boxes of (non-swept) "
            "triangles")
        .def(
            "finalize_active_set",
            &VertexTriangleMixedCcdDcd::FinalizeActiveSet,
            pyb::arg("x"),
            pyb::arg("bComputeBoxes") = true,
            "Removes inactive vertices from the active set\n\n"
            "Args:\n"
            "    x (pbat.gpu.common.Buffer): 3x|#points| buffer of current point positions\n"
            "    bComputeBoxes (bool): If true, computes the bounding boxes of (non-swept) "
            "triangles")
        .def_property(
            "eps",
            nullptr,
            &VertexTriangleMixedCcdDcd::SetNearestNeighbourFloatingPointTolerance,
            "Floating point tolerance for nearest neighbour search")
        .def_property_readonly(
            "active_set",
            &VertexTriangleMixedCcdDcd::ActiveVertexTriangleConstraints,
            "Returns a 2x|#vertex-triangle constraints| matrix where each column is a "
            "vertex-triangle constraint pair")
        .def_property_readonly(
            "active_vertices",
            &VertexTriangleMixedCcdDcd::ActiveVertices,
            "Returns |#active verts| 1D array of active vertex indices into V")
        .def_property_readonly(
            "active_mask",
            &VertexTriangleMixedCcdDcd::ActiveMask,
            "Returns |#verts| 1D mask for active vertices");
}

} // namespace pbat::py::gpu::contact