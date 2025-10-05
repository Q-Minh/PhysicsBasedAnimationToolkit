#include "ValanisLandelQuasiNewton.h"

#include <nanobind/eigen/dense.h>
#include <pbat/Aliases.h>
#include <pbat/physics/Enums.h>
#include <pbat/sim/algorithm/pd/ValanisLandelQuasiNewton.h>

namespace pbat::py::sim::algorithm::pd {

void BindValanisLandelQuasiNewton(nanobind::module_& m)
{
    namespace nb = nanobind;

    m.def(
        "valanis_landel_quasi_newton_stiffness",
        &pbat::sim::algorithm::pd::ValanisLandelQuasiNewtonStiffness<Scalar>,
        nb::arg("mu"),
        nb::arg("lambda"),
        nb::arg("sigma"),
        nb::arg("sigmalo"),
        nb::arg("sigmahi"),
        nb::arg("energy") = pbat::physics::EHyperElasticEnergy::StableNeoHookean,
        "Compute the Valanis-Landel Quasi-Newton stiffness for Projective Dynamics element "
        "constraints.\n\n"
        "Args:\n"
        "    mu: First Lame coefficient (scalar)\n"
        "    lambda: Second Lame coefficient (scalar)\n"
        "    sigma: Singular values of deformation gradient (3D vector)\n"
        "    sigmalo: Lowest singular value of deformation gradient (scalar)\n"
        "    sigmahi: Highest singular value of deformation gradient (scalar)\n"
        "    energy: Hyperelastic energy model (enum)\n"
        "Returns:\n"
        "    k: quasi-Newton stiffness for Projective Dynamics element constraints (scalar)");

    m.def(
        "valanis_landel_quasi_newton_stiffness",
        [](nb::DRef<Eigen::Vector<Scalar, Eigen::Dynamic> const> mug,
           nb::DRef<Eigen::Vector<Scalar, Eigen::Dynamic> const> lambdag,
           nb::DRef<Eigen::Vector<Scalar, Eigen::Dynamic> const> sigmag,
           nb::DRef<Eigen::Vector<Scalar, Eigen::Dynamic> const> sigmalog,
           nb::DRef<Eigen::Vector<Scalar, Eigen::Dynamic> const> sigmahig,
           pbat::physics::EHyperElasticEnergy energy) {
            Eigen::Vector<Scalar, Eigen::Dynamic> kg;
            pbat::sim::algorithm::pd::ValanisLandelQuasiNewtonStiffness(
                mug,
                lambdag,
                sigmag,
                sigmalog,
                sigmahig,
                energy,
                kg);
            return kg;
        },
        nb::arg("mug"),
        nb::arg("lambdag"),
        nb::arg("sigmag"),
        nb::arg("sigmalog"),
        nb::arg("sigmahig"),
        nb::arg("energy") = pbat::physics::EHyperElasticEnergy::StableNeoHookean,
        "Vectorized version of Valanis-Landel Quasi-Newton stiffness for Projective Dynamics "
        "element "
        "constraints.\n\n"
        "Args:\n"
        "    mug: `|# constraints| x 1` First Lame coefficient (vector)\n"
        "    lambdag: `|# constraints| x 1` Second Lame coefficient (vector)\n"
        "    sigmag: `3 x |# constraints|` Singular values of deformation gradient (matrix)\n"
        "    sigmalog: `|# constraints| x 1` Lowest singular value of deformation gradient "
        "(vector)\n"
        "    sigmahig: `|# constraints| x 1` Highest singular value of deformation gradient "
        "(vector)\n"
        "    energy: Hyperelastic energy model (enum)\n"
        "Returns:\n"
        "    kg: `|# constraints| x 1` quasi-Newton stiffness for Projective Dynamics element "
        "constraints (vector)");
}

} // namespace pbat::py::sim::algorithm::pd