#include "HyperElasticPotential.h"

#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>
#include <optional>
#include <pbat/fem/Hexahedron.h>
#include <variant>

namespace pbat {
namespace py {
namespace fem {

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
           nb::DRef<Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> const> x,
           EHyperElasticEnergy eEnergy,
           int eFlags,
           pbat::fem::EHyperElasticSpdCorrection eSpdCorrection,
           EElement eElement,
           int order,
           int dims)
            -> std::variant<
                TScalar,
                Eigen::Vector<TScalar, Eigen::Dynamic>,
                Eigen::SparseMatrix<TScalar, Eigen::RowMajor, TIndex>,
                nb::tuple> {
            if (eElement == EElement::Hexahedron and order > 2)
            {
                throw std::invalid_argument(
                    "Hyperelastic energy for hexahedra is only supported for order 1 and "
                    "2.");
            }
            std::optional<TScalar> potential;
            std::optional<Eigen::Vector<TScalar, Eigen::Dynamic>> gradient;
            std::optional<Eigen::SparseMatrix<TScalar, Eigen::RowMajor, TIndex>> hessian;
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
                            x.reshaped(),
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
            // Return variable tuple based on computation request
            if (potential)
            {
                if (gradient)
                {
                    if (hessian)
                    {
                        return nb::make_tuple(*potential, *gradient, *hessian);
                    }
                    else
                    {
                        return nb::make_tuple(*potential, *gradient);
                    }
                }
                else
                {
                    if (hessian)
                    {
                        return nb::make_tuple(*potential, *hessian);
                    }
                    else
                    {
                        return *potential;
                    }
                }
            }
            else
            {
                if (gradient)
                {
                    if (hessian)
                    {
                        return nb::make_tuple(*gradient, *hessian);
                    }
                    else
                    {
                        return *gradient;
                    }
                }
                else
                {
                    if (hessian)
                    {
                        return *hessian;
                    }
                    else
                    {
                        throw std::runtime_error("No potential, gradient or hessian requested.");
                    }
                }
            }
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
        "Compute hyperelastic potential, gradient and hessian for a given mesh.\n\n"
        "Args\n"
        "    E (numpy.ndarray): `|# nodes per element| x |# elements|` matrix of mesh "
        "elements.\n"
        "    n_nodes (int): Number of nodes in the mesh.\n"
        "    eg (numpy.ndarray): `|# quad.pts.| x 1` vector of element indices at "
        "quadrature points.\n"
        "    wg (numpy.ndarray): `|# quad.pts.| x 1` vector of quadrature weights.\n"
        "    GNeg (numpy.ndarray): `|# nodes per element| x |# dims * # quad.pts.|` shape "
        "function gradients.\n"
        "    mu (numpy.ndarray): First Lame coefficient.\n"
        "    lambda (numpy.ndarray): Second Lame coefficient.\n"
        "    x (numpy.ndarray): `|# dims * # nodes| x 1` deformed nodal positions.\n"
        "    energy (HyperElasticEnergy): Type of hyperelastic energy to compute.\n"
        "    flags (ElementElasticityComputationFlags): Flags for the computation.\n"
        "    spd_correction (HyperElasticSpdCorrection): SPD correction type.\n"
        "    element (EElement): Type of element.\n"
        "    order (int): Order of the element (default: 1).\n"
        "    dims (int): Number of spatial dimensions (default: 3).\n\n"
        "Returns\n"
        "    Tuple[float, numpy.ndarray, scipy.sparse.csr_matrix]: Any tuple combination "
        "of (U, gradU, hessU) based on requested flags.\n");

    m.def(
        "hyper_elastic_potential",
        [](nb::DRef<Eigen::Matrix<TIndex, Eigen::Dynamic, Eigen::Dynamic> const> E,
           TIndex nNodes,
           nb::DRef<Eigen::Vector<TScalar, Eigen::Dynamic> const> wg,
           nb::DRef<Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> const> GNeg,
           TScalar mu,
           TScalar lambda,
           nb::DRef<Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> const> x,
           EHyperElasticEnergy eEnergy,
           int eFlags,
           pbat::fem::EHyperElasticSpdCorrection eSpdCorrection,
           EElement eElement,
           int dims)
            -> std::variant<
                TScalar,
                Eigen::Vector<TScalar, Eigen::Dynamic>,
                Eigen::SparseMatrix<TScalar, Eigen::RowMajor, TIndex>,
                nb::tuple> {
            std::optional<TScalar> potential;
            std::optional<Eigen::Vector<TScalar, Eigen::Dynamic>> gradient;
            std::optional<Eigen::SparseMatrix<TScalar, Eigen::RowMajor, TIndex>> hessian;
            ApplyToElementInDimsWithHyperElasticEnergy(
                eElement,
                1 /* order */,
                dims,
                eEnergy,
                [&]<class ElementType, int Dims, class HyperElasticEnergyType>() {
                    if constexpr (not std::is_same_v<ElementType, pbat::fem::Hexahedron<3>>)
                    {
                        if (wg.size() != E.cols())
                        {
                            throw std::invalid_argument(
                                "Number of element volumes does not match number of "
                                "elements.");
                        }
                        Eigen::Vector<TScalar, Eigen::Dynamic> Ug;
                        Eigen::Matrix<TScalar, Dims * ElementType::kNodes, Eigen::Dynamic> Gg;
                        Eigen::Matrix<TScalar, Dims * ElementType::kNodes, Eigen::Dynamic> Hg;
                        auto const mug =
                            Eigen::Vector<TScalar, Eigen::Dynamic>::Constant(wg.size(), mu);
                        auto const lambdag =
                            Eigen::Vector<TScalar, Eigen::Dynamic>::Constant(wg.size(), lambda);
                        auto const eg = Eigen::Vector<TIndex, Eigen::Dynamic>::LinSpaced(
                            static_cast<TIndex>(wg.size()),
                            0,
                            static_cast<TIndex>(wg.size()) - 1);
                        pbat::fem::ToElementElasticity<ElementType, Dims, HyperElasticEnergyType>(
                            E.template topRows<ElementType::kNodes>(),
                            nNodes,
                            eg,
                            wg,
                            GNeg.template topRows<ElementType::kNodes>(),
                            mug,
                            lambdag,
                            x.reshaped(),
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
            // Return variable tuple based on computation request
            if (potential)
            {
                if (gradient)
                {
                    if (hessian)
                    {
                        return nb::make_tuple(*potential, *gradient, *hessian);
                    }
                    else
                    {
                        return nb::make_tuple(*potential, *gradient);
                    }
                }
                else
                {
                    if (hessian)
                    {
                        return nb::make_tuple(*potential, *hessian);
                    }
                    else
                    {
                        return *potential;
                    }
                }
            }
            else
            {
                if (gradient)
                {
                    if (hessian)
                    {
                        return nb::make_tuple(*gradient, *hessian);
                    }
                    else
                    {
                        return *gradient;
                    }
                }
                else
                {
                    if (hessian)
                    {
                        return *hessian;
                    }
                    else
                    {
                        throw std::runtime_error("No potential, gradient or hessian requested.");
                    }
                }
            }
        },
        nb::arg("E"),
        nb::arg("n_nodes"),
        nb::arg("element_volumes"),
        nb::arg("Gneg"),
        nb::arg("mu"),
        nb::arg("lambda"),
        nb::arg("x"),
        nb::arg("energy")         = EHyperElasticEnergy::StableNeoHookean,
        nb::arg("flags")          = pbat::fem::EElementElasticityComputationFlags::Potential,
        nb::arg("spd_correction") = pbat::fem::EHyperElasticSpdCorrection::Absolute,
        nb::arg("element"),
        nb::arg("dims") = 3,
        "Compute hyperelastic potential, gradient and hessian for a given mesh. This "
        "overload assumes linear elements and homogeneous material.\n\n"
        "Args\n"
        "    E (numpy.ndarray): `|# nodes per element| x |# elements|` matrix of mesh "
        "elements.\n"
        "    n_nodes (int): Number of nodes in the mesh.\n"
        "    element_volumes (numpy.ndarray): `|# elements| x 1` vector of element "
        "volumes.\n"
        "    Gneg (numpy.ndarray): `|# nodes per element| x |# dims * # quad.pts.|` shape "
        "function gradients.\n"
        "    mu (float): First Lame coefficient.\n"
        "    lambda (float): Second Lame coefficient.\n"
        "    x (numpy.ndarray): `|# dims * # nodes| x 1` deformed nodal positions.\n"
        "    energy (HyperElasticEnergy): Type of hyperelastic energy to compute.\n"
        "    flags (ElementElasticityComputationFlags): Flags for the computation.\n"
        "    spd_correction (HyperElasticSpdCorrection): SPD correction type.\n"
        "    element (EElement): Type of element.\n"
        "    dims (int): Number of spatial dimensions (default: 3).\n\n"
        "Returns\n"
        "    Tuple[float, numpy.ndarray, scipy.sparse.csr_matrix]: Any tuple combination "
        "of (U, gradU, hessU) based on requested flags.\n");
}

} // namespace fem
} // namespace py
} // namespace pbat