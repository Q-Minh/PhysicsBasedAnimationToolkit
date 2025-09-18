#include "HyperElasticPotential.h"

#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/stl/tuple.h>
#include <pbat/fem/Hexahedron.h>
#include <pbat/fem/HyperElasticPotential.h>
#include <pbat/physics/SaintVenantKirchhoffEnergy.h>
#include <pbat/physics/StableNeoHookeanEnergy.h>
#include <tuple>

namespace pbat {
namespace py {
namespace fem {

template <class Func>
inline void ApplyToElementInDimsWithHyperElasticEnergy(
    EElement eElement,
    int order,
    int dims,
    EHyperElasticEnergy eEnergy,
    Func f)
{
    ApplyToElementInDims(eElement, order, dims, [&]<pbat::fem::CElement ElementType, int Dims>() {
        switch (eEnergy)
        {
            case EHyperElasticEnergy::SaintVenantKirchhoff: {
                using EnergyType = pbat::physics::SaintVenantKirchhoffEnergy<Dims>;
                // clang-format off
                f.template operator()<ElementType, Dims, EnergyType>();
                // clang-format on
                break;
            }
            case EHyperElasticEnergy::StableNeoHookean: {
                using EnergyType = pbat::physics::StableNeoHookeanEnergy<Dims>;
                // clang-format off
                f.template operator()<ElementType, Dims, EnergyType>();
                // clang-format on
                break;
            }
            default: break;
        }
    });
}

void BindHyperElasticPotential(nanobind::module_& m)
{
    namespace nb = nanobind;
    nb::enum_<EHyperElasticEnergy>(m, "HyperElasticEnergy")
        .value("SaintVenantKirchhoff", EHyperElasticEnergy::SaintVenantKirchhoff)
        .value("StableNeoHookean", EHyperElasticEnergy::StableNeoHookean)
        .export_values();

    nb::enum_<pbat::fem::EHyperElasticSpdCorrection>(m, "HyperElasticSpdCorrection")
        .value("NoCorrection", pbat::fem::EHyperElasticSpdCorrection::None)
        .value("Projection", pbat::fem::EHyperElasticSpdCorrection::Projection)
        .value("Absolute", pbat::fem::EHyperElasticSpdCorrection::Absolute)
        .export_values();

    nb::enum_<pbat::fem::EElementElasticityComputationFlags>(
        m,
        "ElementElasticityComputationFlags",
        nb::is_arithmetic())
        .value("Potential", pbat::fem::EElementElasticityComputationFlags::Potential)
        .value("Gradient", pbat::fem::EElementElasticityComputationFlags::Gradient)
        .value("Hessian", pbat::fem::EElementElasticityComputationFlags::Hessian)
        .export_values();

    using TScalar = pbat::Scalar;
    using TIndex  = pbat::Index;

    m.def(
        "hyper_elastic_potential",
        [](nb::DRef<Eigen::Matrix<TIndex, Eigen::Dynamic, Eigen::Dynamic> const> E,
           TIndex nNodes,
           nb::DRef<Eigen::Vector<TIndex, Eigen::Dynamic> const> eg,
           nb::DRef<Eigen::Vector<TScalar, Eigen::Dynamic> const> wg,
           nb::DRef<Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> const> GNeg,
           nb::DRef<Eigen::Vector<TScalar, Eigen::Dynamic> const> mug,
           nb::DRef<Eigen::Vector<TScalar, Eigen::Dynamic> const> lambdag,
           nb::DRef<Eigen::Vector<TScalar, Eigen::Dynamic> const> x,
           EHyperElasticEnergy eEnergy,
           int eFlags,
           pbat::fem::EHyperElasticSpdCorrection eSpdCorrection,
           EElement eElement,
           int order,
           int dims)
            -> std::tuple<
                TScalar,
                Eigen::Vector<TScalar, Eigen::Dynamic>,
                Eigen::SparseMatrix<TScalar, Eigen::RowMajor, TIndex>> {
            if (eElement == EElement::Hexahedron and order > 2)
            {
                throw std::invalid_argument(
                    "Hyperelastic energy for hexahedra is only supported for order 1 and "
                    "2.");
            }
            TScalar potential;
            Eigen::Vector<TScalar, Eigen::Dynamic> gradient;
            Eigen::SparseMatrix<TScalar, Eigen::RowMajor, TIndex> hessian;
            ApplyToElementInDimsWithHyperElasticEnergy(
                eElement,
                order,
                dims,
                eEnergy,
                [&]<class ElementType, int Dims, class HyperElasticEnergyType>() {
                    if constexpr (not std::is_same_v<ElementType, pbat::fem::Hexahedron<3>>)
                    {
                        Eigen::Vector<TScalar, Eigen::Dynamic> Ug;
                        Eigen::Matrix<TScalar, Dims * ElementType::kNodes, Eigen::Dynamic> Gg;
                        Eigen::Matrix<TScalar, Dims * ElementType::kNodes, Eigen::Dynamic> Hg;
                        pbat::fem::ToElementElasticity<ElementType, Dims, HyperElasticEnergyType>(
                            E.template topRows<ElementType::kNodes>(),
                            nNodes,
                            eg,
                            wg,
                            GNeg.template topRows<ElementType::kNodes>(),
                            mug,
                            lambdag,
                            x,
                            Ug,
                            Gg,
                            Hg,
                            eFlags,
                            eSpdCorrection);
                        if (eFlags & pbat::fem::EElementElasticityComputationFlags::Potential)
                        {
                            potential = pbat::fem::HyperElasticPotential(Ug);
                        }
                        if (eFlags & pbat::fem::EElementElasticityComputationFlags::Gradient)
                        {
                            gradient = pbat::fem::HyperElasticGradient<ElementType, Dims>(
                                E.template topRows<ElementType::kNodes>(),
                                nNodes,
                                eg,
                                Gg);
                        }
                        if (eFlags & pbat::fem::EElementElasticityComputationFlags::Hessian)
                        {
                            hessian =
                                pbat::fem::HyperElasticHessian<ElementType, Dims, Eigen::RowMajor>(
                                    E.template topRows<ElementType::kNodes>(),
                                    nNodes,
                                    eg,
                                    Hg);
                        }
                    }
                });
            return {potential, gradient, hessian};
        },
        nb::arg("E"),
        nb::arg("n_nodes"),
        nb::arg("eg"),
        nb::arg("wg"),
        nb::arg("GNeg"),
        nb::arg("mug"),
        nb::arg("lambdag"),
        nb::arg("x"),
        nb::arg("energy")         = EHyperElasticEnergy::StableNeoHookean,
        nb::arg("flags")          = pbat::fem::EElementElasticityComputationFlags::Potential,
        nb::arg("spd_correction") = pbat::fem::EHyperElasticSpdCorrection::Absolute,
        nb::arg("element"),
        nb::arg("order") = 1,
        nb::arg("dims")  = 3,
        "Compute hyperelastic potential, gradient and/or hessian for a given mesh.\n\n"
        "Args\n"
        "    E (numpy.ndarray): `|# nodes per element| x |# elements|` matrix of mesh "
        "elements.\n"
        "    n_nodes (int): Number of nodes in the mesh.\n"
        "    eg (numpy.ndarray): `|# quad.pts.| x 1` vector of element indices associated with "
        "quadrature points.\n"
        "    wg (numpy.ndarray): `|# quad.pts.| x 1` vector of quadrature weights.\n"
        "    GNeg (numpy.ndarray): `|# nodes per element| x |# dims * # quad.pts.|` shape "
        "function gradients.\n"
        "    mug (numpy.ndarray): `|# quad.pts.| x 1` first Lame coefficient.\n"
        "    lambdag (numpy.ndarray): `|# quad.pts.| x 1` second Lame coefficient.\n"
        "    x (numpy.ndarray): `|# dims * # nodes| x 1` deformed nodal positions.\n"
        "    energy (HyperElasticEnergy): Type of hyperelastic energy to compute.\n"
        "    flags (ElementElasticityComputationFlags): Flags for the computation.\n"
        "    spd_correction (HyperElasticSpdCorrection): SPD correction type.\n"
        "    element (EElement): Type of element.\n"
        "    order (int): Order of the element (default: 1).\n"
        "    dims (int): Number of spatial dimensions (default: 3).\n\n"
        "Returns\n"
        "    Tuple[float, numpy.ndarray, scipy.sparse.csr_matrix]: The tuple (U, gradU, hessU) "
        "based on requested flags.\n");
}

} // namespace fem
} // namespace py
} // namespace pbat