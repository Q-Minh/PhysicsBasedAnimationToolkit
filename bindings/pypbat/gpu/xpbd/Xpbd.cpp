#include "Xpbd.h"

#include <pbat/gpu/Aliases.h>
#include <pbat/gpu/xpbd/Xpbd.h>
#include <pbat/profiling/Profiling.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <utility>

namespace pbat {
namespace py {
namespace gpu {
namespace xpbd {

void Bind(pybind11::module& m)
{
    namespace pyb = pybind11;
#ifdef PBAT_USE_CUDA

    using namespace pbat;
    using pbat::gpu::xpbd::Xpbd;

    using EConstraint = Xpbd::EConstraint;
    pyb::enum_<EConstraint>(m, "ConstraintType")
        .value("StableNeoHookean", EConstraint::StableNeoHookean)
        .value("Collision", EConstraint::Collision)
        .export_values();

    pyb::class_<Xpbd>(m, "Xpbd")
        .def(
            pyb::init([](Eigen::Ref<GpuMatrixX const> const& X,
                         Eigen::Ref<GpuIndexMatrixX const> const& V,
                         Eigen::Ref<GpuIndexMatrixX const> const& F,
                         Eigen::Ref<GpuIndexMatrixX const> const& T,
                         Eigen::Ref<GpuIndexVectorX const> const& BV,
                         Eigen::Ref<GpuIndexVectorX const> const& BF,
                         std::size_t nMaxVertexTetrahedronOverlaps,
                         std::size_t nMaxVertexTriangleContacts) {
                return pbat::profiling::Profile("pbat.gpu.xpbd.Xpbd.Construct", [&]() {
                    Xpbd xpbd(
                        X,
                        V,
                        F,
                        T,
                        BV,
                        BF,
                        nMaxVertexTetrahedronOverlaps,
                        nMaxVertexTriangleContacts);
                    return xpbd;
                });
            }),
            pyb::arg("X"),
            pyb::arg("V"),
            pyb::arg("F"),
            pyb::arg("T"),
            pyb::arg("BV"),
            pyb::arg("BF"),
            pyb::arg("max_vertex_tetrahedron_overlaps"),
            pyb::arg("max_vertex_triangle_contacts"),
            "Construct an XPBD algorithm to run on the GPU using input particle positions X as an "
            "array of dimensions 3x|#particles|, collision vertices V as an array of dimensions "
            "1x|#collision vertices|, collision triangles F as an array of dimensions "
            "3x|#collision triangles| and tetrahedra T as an array of dimensions 4x|#tetrahedra|. "
            "max_vertex_tetrahedron_overlaps specifies the size of memory preallocated for "
            "vertex-tetrahedron overlaps detected in the broad phase. max_vertex_triangle_contacts "
            "specifies maximum number of collision constraints that will be supported.")
        .def(
            "prepare",
            [](Xpbd& xpbd) {
                pbat::profiling::Profile("pbat.gpu.xpbd.Xpbd.PrepareConstraints", [&]() {
                    xpbd.PrepareConstraints();
                });
            },
            "Precompute constraint data/parameters. Must be called before the simulation starts.")
        .def(
            "step",
            [](Xpbd& xpbd, GpuScalar dt, GpuIndex iterations, GpuIndex substeps) {
                pbat::profiling::Profile("pbat.gpu.xpbd.Xpbd.Step", [&]() {
                    xpbd.Step(dt, iterations, substeps);
                });
            },
            pyb::arg("dt")         = 0.01f,
            pyb::arg("iterations") = 10,
            pyb::arg("substeps")   = 5,
            "Integrate 1 time step in time")
        .def_property(
            "x",
            [](Xpbd const& xpbd) {
                return pbat::profiling::Profile("pbat.gpu.xpbd.Xpbd.GetPositions", [&]() {
                    GpuMatrixX X = xpbd.Positions();
                    return X;
                });
            },
            [](Xpbd& xpbd, Eigen::Ref<GpuMatrixX const> const& X) {
                pbat::profiling::Profile("pbat.gpu.xpbd.Xpbd.SetPositions", [&]() {
                    xpbd.SetPositions(X);
                });
            },
            "|#dims|x|#particles| particle positions")
        .def_property(
            "v",
            [](Xpbd const& xpbd) {
                return pbat::profiling::Profile("pbat.gpu.xpbd.Xpbd.GetVelocities", [&]() {
                    GpuMatrixX v = xpbd.GetVelocity();
                    return v;
                });
            },
            [](Xpbd& xpbd, Eigen::Ref<GpuMatrixX const> const& v) {
                pbat::profiling::Profile("pbat.gpu.xpbd.Xpbd.SetVelocities", [&]() {
                    xpbd.SetVelocities(v);
                });
            },
            "|#dims|x|#particles| particle velocities")
        .def_property(
            "f",
            [](Xpbd const& xpbd) {
                return pbat::profiling::Profile("pbat.gpu.xpbd.Xpbd.GetExternalForces", [&]() {
                    GpuMatrixX f = xpbd.GetExternalForce();
                    return f;
                });
            },
            [](Xpbd& xpbd, Eigen::Ref<GpuMatrixX const> const& f) {
                pbat::profiling::Profile("pbat.gpu.xpbd.Xpbd.SetExternalForces", [&]() {
                    xpbd.SetExternalForces(f);
                });
            },
            "|#dims|x|#particles| particle external forces")
        .def_property(
            "minv",
            [](Xpbd const& xpbd) {
                return pbat::profiling::Profile("pbat.gpu.xpbd.Xpbd.GetMassInverse", [&]() {
                    GpuVectorX minv = xpbd.GetMassInverse();
                    return minv;
                });
            },
            [](Xpbd& xpbd, Eigen::Ref<GpuMatrixX const> const& minv) {
                pbat::profiling::Profile("pbat.gpu.xpbd.Xpbd.SetMassInverse", [&]() {
                    xpbd.SetMassInverse(minv);
                });
            },
            "|#particles| particle mass inverses")
        .def_property(
            "lame",
            [](Xpbd const& xpbd) {
                return pbat::profiling::Profile("pbat.gpu.xpbd.Xpbd.GetLameCoefficients", [&]() {
                    GpuMatrixX lame = xpbd.GetLameCoefficients();
                    return lame;
                });
            },
            [](Xpbd& xpbd, Eigen::Ref<GpuMatrixX const> const& lame) {
                pbat::profiling::Profile("pbat.gpu.xpbd.Xpbd.SetLameCoefficients", [&]() {
                    xpbd.SetLameCoefficients(lame);
                });
            },
            "2x|#elements| Lame coefficients")
        .def_property_readonly(
            "shape_matrix_inverse",
            [](Xpbd const& xpbd) {
                return pbat::profiling::Profile("pbat.gpu.xpbd.Xpbd.GetShapeMatrixInverse", [&]() {
                    GpuMatrixX DmInv = xpbd.GetShapeMatrixInverse();
                    return DmInv;
                });
            },
            "3x|3*#elements| element shape matrix inverses")
        .def(
            "lagrange",
            [](Xpbd const& xpbd, EConstraint eConstraint) {
                return pbat::profiling::Profile("pbat.gpu.xpbd.Xpbd.GetLagrangeMultiplier", [&]() {
                    GpuMatrixX lambda = xpbd.GetLagrangeMultiplier(eConstraint);
                    return lambda;
                });
            },
            "|#lagrange multiplier per constraint|x|#constraint of type eConstraint| lagrange "
            "multipliers")
        .def(
            "alpha",
            [](Xpbd const& xpbd, EConstraint eConstraint) {
                return pbat::profiling::Profile("pbat.gpu.xpbd.Xpbd.GetCompliance", [&]() {
                    GpuMatrixX alpha = xpbd.GetCompliance(eConstraint);
                    return alpha;
                });
            },
            "|#lagrange multiplier per constraint|x|#constraint of type eConstraint| constraint "
            "compliances")
        .def(
            "set_compliance",
            [](Xpbd& xpbd, Eigen::Ref<GpuMatrixX const> const& alpha, EConstraint eConstraint) {
                return pbat::profiling::Profile("pbat.gpu.xpbd.Xpbd.SetCompliance", [&]() {
                    xpbd.SetCompliance(alpha, eConstraint);
                });
            },
            "Set the |#lagrange multiplier per constraint|x|#constraint of type eConstraint| "
            "constraint compliances for the given constraint type")
        .def_property(
            "mu",
            nullptr,
            [](Xpbd& xpbd, std::pair<GpuScalar, GpuScalar> mu) {
                xpbd.SetFrictionCoefficients(mu.first, mu.second);
            },
            "Tuple of static and dynamic friction coefficients (muS, muK).")
        .def_property(
            "scene_bounding_box",
            nullptr,
            [](Xpbd& xpbd,
               std::pair<Eigen::Vector<GpuScalar, 3> const&, Eigen::Vector<GpuScalar, 3> const&>
                   box) { xpbd.SetSceneBoundingBox(box.first, box.second); },
            "Tuple of (min,max) scene bounding box extremities.")
        .def_property(
            "partitions",
            [](Xpbd const& xpbd) {
                return pbat::profiling::Profile(
                    "pbat.gpu.xpbd.Xpbd.GetConstraintPartitions",
                    [&]() {
                        std::vector<std::vector<GpuIndex>> partitions = xpbd.GetPartitions();
                        return partitions;
                    });
            },
            [](Xpbd& xpbd, std::vector<std::vector<GpuIndex>> const& partitions) {
                pbat::profiling::Profile("pbat.gpu.xpbd.Xpbd.SetConstraintPartitions", [&]() {
                    xpbd.SetConstraintPartitions(partitions);
                });
            },
            "Set constraint partitions for the parallel constraint solve as list of lists of "
            "constraint indices")
        .def_property_readonly(
            "vertex_tetrahedron_overlaps",
            [](Xpbd const& xpbd) {
                return pbat::profiling::Profile(
                    "pbat.gpu.xpbd.Xpbd.GetVertexTetrahedronCollisionCandidates",
                    [&]() {
                        GpuIndexMatrixX O = xpbd.GetVertexTetrahedronCollisionCandidates();
                        return O;
                    });
            },
            "2x|#overlap| vertex tetrahedron overlap pairs O, s.t. O[0,:] and O[1,:] yield "
            "overlapping vertices and tetrahedra, respectively.")
        .def_property_readonly(
            "vertex_triangle_contacts",
            [](Xpbd const& xpbd) {
                return pbat::profiling::Profile(
                    "pbat.gpu.xpbd.Xpbd.GetVertexTriangleContactPairs",
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