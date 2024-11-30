#include "Level.h"

#include <pbat/sim/vbd/Level.h>
#include <pybind11/eigen.h>

namespace pbat {
namespace py {
namespace sim {
namespace vbd {

void BindLevel(pybind11::module& m)
{
    namespace pyb = pybind11;

    using pbat::sim::vbd::Level;
    using Cage             = pbat::sim::vbd::Level::Cage;
    using Energy           = pbat::sim::vbd::Level::Energy;
    using RootParameterBus = pbat::sim::vbd::Level::RootParameterBus;

    auto mlevel = m.def_submodule("level");

    pyb::class_<Energy>(mlevel, "Energy")
        .def(pyb::init<>())
        .def(
            "with_quadrature",
            &Energy::WithQuadrature,
            pyb::arg("wg"),
            pyb::arg("sg"),
            "Defines the numerical integration scheme for evaluating this energy.\n"
            "Args:\n"
            "wg (np.ndarray): |#quad.pts.| array of quadrature weights.\n"
            "sg (np.ndarray): |#quad.pts.| boolean array identifying singular quadrature "
            "points.")
        .def(
            "with_adjacency",
            &Energy::WithAdjacency,
            pyb::arg("GVGp"),
            pyb::arg("GVGg"),
            pyb::arg("GVGe"),
            pyb::arg("GVGilocal"),
            "Defines the vertex-quad.pt. adjacency structure of this energy.\n"
            "Args:\n"
            "GVGp (np.ndarray): |#verts + 1| prefix array of vertex-quad.pt. adjacency graph\n"
            "GVGg (np.ndarray): |#vertex-quad.pt. adjacencies| array of adjacency graph edges\n"
            "GVGe (np.ndarray): |#vertex-quad.pt. adjacencies| array of element indices associated "
            "with adjacency graph edges\n"
            "GVGilocal (np.ndarray): |#vertex-quad.pt. adjacencies| array of local vertex indices "
            "associated with adjacency graph edges")
        .def(
            "with_dirichlet_adjacency",
            &Energy::WithDirichletAdjacency,
            pyb::arg("GVDGp"),
            pyb::arg("GVDGg"),
            pyb::arg("GVDGe"),
            pyb::arg("GVDGilocal"),
            "Defines the vertex-Dirichlet quad.pt. adjacency structure of this energy.\n"
            "Args:\n"
            "GVDGp (np.ndarray): |#verts + 1| prefix array of vertex-quad.pt. adjacency graph\n"
            "GVDGg (np.ndarray): |#vertex-Dirichlet quad.pt. adjacencies| array of Dirichlet "
            "adjacency graph edges\n"
            "GVDGe (np.ndarray): |#vertex-Dirichlet quad.pt. adjacencies| array of element indices "
            "associated with Dirichlet adjacency graph edges\n"
            "GVGilocal (np.ndarray): |#vertex-Dirichlet quad.pt. adjacencies| array of local "
            "vertex indices associated with Dirichlet adjacency graph edges")
        .def(
            "with_kinetic_energy",
            &Energy::WithKineticEnergy,
            pyb::arg("rhog"),
            pyb::arg("Ncg"),
            "Defines the kinetic term of this energy.\n"
            "Args:\n"
            "rhog (np.ndarray): |#quad.pts.| array of mass densities at quadrature points\n"
            "Ncg (np.ndarray): 4x|#quad.pts.| array of coarse cage element shape functions at "
            "quadrature points")
        .def(
            "with_potential_energy",
            &Energy::WithPotentialEnergy,
            pyb::arg("mug"),
            pyb::arg("lambdag"),
            pyb::arg("erg"),
            pyb::arg("Nrg"),
            pyb::arg("GNfg"),
            pyb::arg("GNcg"),
            "Defines the potential term of this energy\n"
            "Args:\n"
            "mug (np.ndarray): |#quad.pts.| array of first Lame coefficients at quadrature points\n"
            "lambdag (np.ndarray): |#quad.pts.| array of second Lame coefficients at quadrature "
            "points\n"
            "erg (np.ndarray): 4x|#quad.pts.| array of coarse element indices containing vertices "
            "of root level element embedding quadrature point g\n"
            "Nrg (np.ndarray): 4x|4*#quad.pts.| array of coarse cage element shape functions at "
            "root level elements' 4 vertices associated with quadrature points\n"
            "GNfg (np.ndarray): 4x|3*#quad.pts.| array of root level element shape function "
            "gradients at quadrature points\n"
            "GNcg (np.ndarray): 4x|3*#quad.pts.| array of coarse cage element shape function "
            "gradients at quadrature points")
        .def(
            "with_dirichlet_energy",
            &Energy::WithDirichletEnergy,
            pyb::arg("dwg"),
            pyb::arg("dNcg"),
            pyb::arg("dxg"),
            "Defines the Dirichlet term of this energy\n"
            "Args:\n"
            "dwg (np.ndarray): |#Dirichlet quad.pts.| array of Dirichlet quadrature weights\n"
            "dNcg (np.ndarray): 4x|#Dirichlet quad.pts.| array of coarse cage element shape "
            "functions at Dirichlet quadrature points\n"
            "dxg (np.ndarray): 3x|#Dirichlet quad.pts.| array of Dirichlet conditions at Dirichlet "
            "quadrature points")
        .def("construct", &Energy::Construct, pyb::arg("validate") = true)
        .def_readwrite("xtildeg", &Energy::xtildeg)
        .def_readwrite("rhog", &Energy::rhog)
        .def_readwrite("Ncg", &Energy::Ncg)
        .def_readwrite("mug", &Energy::mug)
        .def_readwrite("lambdag", &Energy::lambdag)
        .def_readwrite("erg", &Energy::erg)
        .def_readwrite("Nrg", &Energy::Nrg)
        .def_readwrite("GNfg", &Energy::GNfg)
        .def_readwrite("GNcg", &Energy::GNcg)
        .def_readwrite("sg", &Energy::sg)
        .def_readwrite("wg", &Energy::wg)
        .def_readwrite("GVGp", &Energy::GVGp)
        .def_readwrite("GVGg", &Energy::GVGg)
        .def_readwrite("GVGe", &Energy::GVGe)
        .def_readwrite("GVGilocal", &Energy::GVGilocal);

    pyb::class_<Cage>(mlevel, "Cage")
        .def(
            pyb::init<
                Eigen::Ref<MatrixX const> const&,
                Eigen::Ref<IndexMatrixX const> const&,
                Eigen::Ref<IndexVectorX const> const&,
                Eigen::Ref<IndexVectorX const> const&>(),
            pyb::arg("x"),
            pyb::arg("E"),
            pyb::arg("ptr"),
            pyb::arg("adj"),
            "Construct a multiscale VBD coarse cage.\n"
            "Args:\n"
            "x (np.ndarray): 3x|#verts| array of coarse cage vertex positions\n"
            "E (np.ndarray): 4x|#elements| array of coarse cage tetrahedra\n"
            "ptr (np.ndarray): |#partitions+1| array of vertex partition pointers\n"
            "adj (np.ndarray): |#free verts| array of partition vertices")
        .def_readwrite("E", &Cage::E)
        .def_readwrite("x", &Cage::x)
        .def_readwrite("ptr", &Cage::ptr)
        .def_readwrite("adj", &Cage::adj);

    pyb::class_<RootParameterBus>(mlevel, "RootParameterBus")
        .def(
            pyb::init<Eigen::Ref<IndexVectorX const> const&, Eigen::Ref<MatrixX const> const&>(),
            pyb::arg("erg"),
            pyb::arg("Nrg"),
            "Construct a multiscale VBD coarse cage.\n"
            "Args:\n"
            "erg (np.ndarray): |#quad.pts.| array of root element associated with quadrature "
            "points\n"
            "Nrg (np.ndarray): 4x|#quad.pts at level l| arrays of root level shape functions "
            "evaluated at quadrature points")
        .def_readwrite("erg", &RootParameterBus::erg)
        .def_readwrite("Nrg", &RootParameterBus::Nrg);

    pyb::class_<Level>(m, "Level")
        .def(
            pyb::init([](Cage& C, Energy& E, RootParameterBus& RPB) {
                return Level(std::move(C), std::move(E), std::move(RPB));
            }),
            pyb::arg("cage"),
            pyb::arg("energy"),
            pyb::arg("bus"))
        .def_readwrite("cage", &Level::C)
        .def_readwrite("energy", &Level::E)
        .def_readwrite("bus", &Level::RPB);
}

} // namespace vbd
} // namespace sim
} // namespace py
} // namespace pbat
