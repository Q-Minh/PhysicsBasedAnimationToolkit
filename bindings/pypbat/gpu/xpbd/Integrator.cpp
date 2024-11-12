#include "Integrator.h"

#include <pbat/gpu/Aliases.h>
#include <pbat/gpu/xpbd/Integrator.h>
#include <pbat/profiling/Profiling.h>
#include <pbat/sim/xpbd/Data.h>
#include <pbat/sim/xpbd/Enums.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <utility>

namespace pbat {
namespace py {
namespace gpu {
namespace xpbd {

void BindIntegrator([[maybe_unused]] pybind11::module& m)
{
    namespace pyb = pybind11;
#ifdef PBAT_USE_CUDA

    using namespace pbat;
    using pbat::gpu::xpbd::Integrator;
    using pbat::sim::xpbd::Data;
    using pbat::sim::xpbd::EConstraint;

    pyb::class_<Integrator>(m, "Integrator")
        .def(
            pyb::init([](Data const& data,
                         std::size_t nMaxVertexTetrahedronOverlaps,
                         std::size_t nMaxVertexTriangleContacts) {
                return pbat::profiling::Profile("pbat.gpu.xpbd.Integrator.Construct", [&]() {
                    Integrator xpbd(
                        data,
                        nMaxVertexTetrahedronOverlaps,
                        nMaxVertexTriangleContacts);
                    return xpbd;
                });
            }),
            pyb::arg("data"),
            pyb::arg("max_vertex_tetrahedron_overlaps"),
            pyb::arg("max_vertex_triangle_contacts"),
            "Constructs an XPBD algorithm with the given data, where "
            "max_vertex_tetrahedron_overlaps specifies the size of memory preallocated for "
            "vertex-tetrahedron overlaps detected in the broad phase. max_vertex_triangle_contacts "
            "specifies maximum number of collision constraints that will be supported.")
        .def(
            "step",
            [](Integrator& xpbd, GpuScalar dt, GpuIndex iterations, GpuIndex substeps) {
                pbat::profiling::Profile("pbat.gpu.xpbd.Integrator.Step", [&]() {
                    xpbd.Step(dt, iterations, substeps);
                });
            },
            pyb::arg("dt")         = 0.01f,
            pyb::arg("iterations") = 10,
            pyb::arg("substeps")   = 5,
            "Integrate 1 time step in time")
        .def_property(
            "x",
            [](Integrator const& xpbd) {
                return pbat::profiling::Profile("pbat.gpu.xpbd.Integrator.GetPositions", [&]() {
                    GpuMatrixX X = xpbd.Positions();
                    return X;
                });
            },
            [](Integrator& xpbd, Eigen::Ref<GpuMatrixX const> const& X) {
                pbat::profiling::Profile("pbat.gpu.xpbd.Integrator.SetPositions", [&]() {
                    xpbd.SetPositions(X);
                });
            },
            "|#dims|x|#particles| particle positions")
        .def_property(
            "v",
            [](Integrator const& xpbd) {
                return pbat::profiling::Profile("pbat.gpu.xpbd.Integrator.GetVelocities", [&]() {
                    GpuMatrixX v = xpbd.GetVelocity();
                    return v;
                });
            },
            [](Integrator& xpbd, Eigen::Ref<GpuMatrixX const> const& v) {
                pbat::profiling::Profile("pbat.gpu.xpbd.Integrator.SetVelocities", [&]() {
                    xpbd.SetVelocities(v);
                });
            },
            "|#dims|x|#particles| particle velocities")
        .def_property(
            "aext",
            [](Integrator const& xpbd) {
                return pbat::profiling::Profile(
                    "pbat.gpu.xpbd.Integrator.GetExternalAcceleration",
                    [&]() {
                        GpuMatrixX f = xpbd.GetExternalAcceleration();
                        return f;
                    });
            },
            [](Integrator& xpbd, Eigen::Ref<GpuMatrixX const> const& f) {
                pbat::profiling::Profile("pbat.gpu.xpbd.Integrator.SetExternalAcceleration", [&]() {
                    xpbd.SetExternalAcceleration(f);
                });
            },
            "|#dims|x|#particles| particle external accelerations")
        .def_property(
            "minv",
            [](Integrator const& xpbd) {
                return pbat::profiling::Profile("pbat.gpu.xpbd.Integrator.GetMassInverse", [&]() {
                    GpuVectorX minv = xpbd.GetMassInverse();
                    return minv;
                });
            },
            [](Integrator& xpbd, Eigen::Ref<GpuMatrixX const> const& minv) {
                pbat::profiling::Profile("pbat.gpu.xpbd.Integrator.SetMassInverse", [&]() {
                    xpbd.SetMassInverse(minv);
                });
            },
            "|#particles| particle mass inverses")
        .def_property(
            "lame",
            [](Integrator const& xpbd) {
                return pbat::profiling::Profile(
                    "pbat.gpu.xpbd.Integrator.GetLameCoefficients",
                    [&]() {
                        GpuMatrixX lame = xpbd.GetLameCoefficients();
                        return lame;
                    });
            },
            [](Integrator& xpbd, Eigen::Ref<GpuMatrixX const> const& lame) {
                pbat::profiling::Profile("pbat.gpu.xpbd.Integrator.SetLameCoefficients", [&]() {
                    xpbd.SetLameCoefficients(lame);
                });
            },
            "2x|#elements| Lame coefficients")
        .def_property_readonly(
            "shape_matrix_inverse",
            [](Integrator const& xpbd) {
                return pbat::profiling::Profile(
                    "pbat.gpu.xpbd.Integrator.GetShapeMatrixInverse",
                    [&]() {
                        GpuMatrixX DmInv = xpbd.GetShapeMatrixInverse();
                        return DmInv;
                    });
            },
            "3x|3*#elements| element shape matrix inverses")
        .def(
            "lagrange",
            [](Integrator const& xpbd, EConstraint eConstraint) {
                return pbat::profiling::Profile(
                    "pbat.gpu.xpbd.Integrator.GetLagrangeMultiplier",
                    [&]() {
                        GpuMatrixX lambda = xpbd.GetLagrangeMultiplier(eConstraint);
                        return lambda;
                    });
            },
            "|#lagrange multiplier per constraint|x|#constraint of type eConstraint| lagrange "
            "multipliers")
        .def(
            "alpha",
            [](Integrator const& xpbd, EConstraint eConstraint) {
                return pbat::profiling::Profile("pbat.gpu.xpbd.Integrator.GetCompliance", [&]() {
                    GpuMatrixX alpha = xpbd.GetCompliance(eConstraint);
                    return alpha;
                });
            },
            "|#lagrange multiplier per constraint|x|#constraint of type eConstraint| constraint "
            "compliances")
        .def(
            "set_compliance",
            [](Integrator& xpbd,
               Eigen::Ref<GpuMatrixX const> const& alpha,
               EConstraint eConstraint) {
                return pbat::profiling::Profile("pbat.gpu.xpbd.Integrator.SetCompliance", [&]() {
                    xpbd.SetCompliance(alpha, eConstraint);
                });
            },
            "Set the |#lagrange multiplier per constraint|x|#constraint of type eConstraint| "
            "constraint compliances for the given constraint type")
        .def_property(
            "mu",
            nullptr,
            [](Integrator& xpbd, std::pair<GpuScalar, GpuScalar> mu) {
                xpbd.SetFrictionCoefficients(mu.first, mu.second);
            },
            "Tuple of static and dynamic friction coefficients (muS, muK).")
        .def_property(
            "scene_bounding_box",
            nullptr,
            [](Integrator& xpbd,
               std::pair<Eigen::Vector<GpuScalar, 3> const&, Eigen::Vector<GpuScalar, 3> const&>
                   box) { xpbd.SetSceneBoundingBox(box.first, box.second); },
            "Tuple of (min,max) scene bounding box extremities.")
        .def_property(
            "partitions",
            [](Integrator const& xpbd) {
                return pbat::profiling::Profile(
                    "pbat.gpu.xpbd.Integrator.GetConstraintPartitions",
                    [&]() {
                        std::vector<std::vector<GpuIndex>> partitions = xpbd.GetPartitions();
                        return partitions;
                    });
            },
            [](Integrator& xpbd, std::vector<std::vector<GpuIndex>> const& partitions) {
                pbat::profiling::Profile("pbat.gpu.xpbd.Integrator.SetConstraintPartitions", [&]() {
                    xpbd.SetConstraintPartitions(partitions);
                });
            },
            "Set constraint partitions for the parallel constraint solve as list of lists of "
            "constraint indices")
        .def_property_readonly(
            "vertex_tetrahedron_overlaps",
            [](Integrator const& xpbd) {
                return pbat::profiling::Profile(
                    "pbat.gpu.xpbd.Integrator.GetVertexTetrahedronCollisionCandidates",
                    [&]() {
                        GpuIndexMatrixX O = xpbd.GetVertexTetrahedronCollisionCandidates();
                        return O;
                    });
            },
            "2x|#overlap| vertex tetrahedron overlap pairs O, s.t. O[0,:] and O[1,:] yield "
            "overlapping vertices and tetrahedra, respectively.")
        .def_property_readonly(
            "vertex_triangle_contacts",
            [](Integrator const& xpbd) {
                return pbat::profiling::Profile(
                    "pbat.gpu.xpbd.Integrator.GetVertexTriangleContactPairs",
                    [&]() {
                        GpuIndexMatrixX C = xpbd.GetVertexTriangleContactPairs();
                        return C;
                    });
            },
            "2x|#contacts| vertex triangle contacts pairs C, s.t. C[0,:] and C[1,:] yield "
            "contacting vertices and triangles, respectively.");
#endif // PBAT_USE_CUDA
}

} // namespace xpbd
} // namespace gpu
} // namespace py
} // namespace pbat