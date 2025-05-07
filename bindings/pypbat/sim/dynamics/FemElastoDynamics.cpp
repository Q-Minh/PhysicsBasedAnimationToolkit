#include "FemElastoDynamics.h"

#include "pypbat/fem/HyperElasticPotential.h"
#include "pypbat/fem/Mesh.h"

#include <pbat/io/Archive.h>
#include <pbat/sim/dynamics/FemElastoDynamics.h>
#include <pbat/sim/integration/Bdf.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace pbat::py::sim::dynamics {

class FemElastoDynamics
{
  public:
    FemElastoDynamics(
        int meshDims,
        int meshOrder,
        fem::EElement meshElement,
        fem::EHyperElasticEnergy hyperElasticEnergy);

    FemElastoDynamics(FemElastoDynamics const&) = delete;
    FemElastoDynamics(FemElastoDynamics&&);
    FemElastoDynamics& operator=(FemElastoDynamics const&) = delete;
    FemElastoDynamics& operator=(FemElastoDynamics&&);

    fem::Mesh Mesh() const;

    Eigen::Map<MatrixX const> x() const;
    Eigen::Map<MatrixX> x();
    Eigen::Map<MatrixX const> v() const;
    Eigen::Map<MatrixX> v();
    Eigen::Map<MatrixX const> fext() const;
    Eigen::Map<MatrixX> fext();
    Eigen::Map<VectorX const> m() const;
    Eigen::Map<VectorX> m();

    pbat::sim::dynamics::ElasticityQuadrature const& QU() const;
    pbat::sim::dynamics::ElasticityQuadrature& QU();

    std::optional<fem::HyperElasticPotential> U() const;

    IndexVectorX FreeNodes() const;
    IndexVectorX DirichletNodes() const;

    pbat::sim::integration::Bdf const& Bdf() const;
    pbat::sim::integration::Bdf& Bdf();

    void Construct(Eigen::Ref<MatrixX const> const& V, Eigen::Ref<IndexMatrixX const> const& C);

    void
    SetInitialConditions(Eigen::Ref<MatrixX const> const& x0, Eigen::Ref<MatrixX const> const& v0);

    void SetMassMatrix(Scalar rho);
    void SetElasticEnergy(Scalar mu, Scalar lambda, bool bWithElasticPotential = true);
    void SetExternalLoad(Eigen::Ref<MatrixX const> const& fext);
    void SetTimeIntegrationScheme(Scalar dt, int bdfstep);
    void Constrain(Eigen::Ref<Eigen::Vector<bool, Eigen::Dynamic> const> const& D);

    void SetMassMatrix(
        Eigen::Ref<IndexVectorX const> const& eg,
        Eigen::Ref<VectorX const> const& wg,
        Eigen::Ref<MatrixX const> const& Xg,
        Eigen::Ref<VectorX const> const& rhog);
    void SetElasticEnergy(
        Eigen::Ref<IndexVectorX const> const& eg,
        Eigen::Ref<VectorX const> const& wg,
        Eigen::Ref<MatrixX const> const& Xg,
        Eigen::Ref<VectorX const> const& mug,
        Eigen::Ref<VectorX const> const& lambdag,
        bool bWithElasticPotential = true);
    void SetExternalLoad(
        Eigen::Ref<IndexVectorX const> const& eg,
        Eigen::Ref<VectorX const> const& wg,
        Eigen::Ref<MatrixX const> const& Xg,
        Eigen::Ref<MatrixX const> const& bg);

    void Serialize(io::Archive& archive) const;
    void Deserialize(io::Archive const& archive);

    template <class Func>
    void Access(Func f) const
    {
        HepApplyToMesh(mDims, mOrder, mElement, [&]<class MeshType>() {
            pbat::common::ForTypes<
                pbat::physics::SaintVenantKirchhoffEnergy<MeshType::kDims>,
                pbat::physics::StableNeoHookeanEnergy<MeshType::kDims>>(
                [&]<class HyperElasticEnergyType>() {
                    using ElementType    = typename MeshType::ElementType;
                    auto constexpr kDims = MeshType::kDims;
                    using HyperElasticPotentialType =
                        pbat::fem::HyperElasticPotential<MeshType, HyperElasticEnergyType>;
                    using FemElastoDynamicsType = pbat::sim::dynamics::
                        FemElastoDynamics<ElementType, kDims, HyperElasticEnergyType>;
                    if (mHyperElasticEnergy == fem::EHyperElasticEnergy::SaintVenantKirchhoff and
                        std::is_same_v<
                            HyperElasticEnergyType,
                            pbat::physics::SaintVenantKirchhoffEnergy<kDims>>)
                    {
                        FemElastoDynamicsType* impl =
                            reinterpret_cast<FemElastoDynamicsType*>(mFemElastoDynamics);
                        f.template operator()<FemElastoDynamicsType>(impl);
                    }
                    if (mHyperElasticEnergy == fem::EHyperElasticEnergy::StableNeoHookean and
                        std::is_same_v<
                            HyperElasticEnergyType,
                            pbat::physics::StableNeoHookeanEnergy<kDims>>)
                    {
                        FemElastoDynamicsType* impl =
                            reinterpret_cast<FemElastoDynamicsType*>(mFemElastoDynamics);
                        f.template operator()<FemElastoDynamicsType>(impl);
                    }
                });
        });
    }

    ~FemElastoDynamics();

    int mDims;
    int mOrder;
    fem::EElement mElement;
    fem::EHyperElasticEnergy mHyperElasticEnergy;

  private:
    void* mFemElastoDynamics;
};

void BindFemElastoDynamics([[maybe_unused]] pybind11::module& m)
{
    namespace pyb = pybind11;

    pyb::class_<pbat::sim::dynamics::ElasticityQuadrature>(m, "ElasticityQuadrature")
        .def(pyb::init<>())
        .def_readwrite("eg", &pbat::sim::dynamics::ElasticityQuadrature::eg)
        .def_readwrite("wg", &pbat::sim::dynamics::ElasticityQuadrature::wg)
        .def_readwrite("GNeg", &pbat::sim::dynamics::ElasticityQuadrature::GNeg)
        .def_readwrite("lameg", &pbat::sim::dynamics::ElasticityQuadrature::lameg);

    pyb::class_<FemElastoDynamics>(m, "FemElastoDynamics")
        .def(pyb::init<int, int, fem::EElement, fem::EHyperElasticEnergy>())
        .def_property_readonly("mesh", &FemElastoDynamics::Mesh, "FEM mesh")
        .def_property(
            "x",
            [](FemElastoDynamics const& self) { return self.x(); },
            [](FemElastoDynamics& self, Eigen::Ref<MatrixX const> const& x) { self.x() = x; },
            "|#dims|x|#nodes| array of nodal positions")
        .def_property(
            "v",
            [](FemElastoDynamics const& self) { return self.v(); },
            [](FemElastoDynamics& self, Eigen::Ref<MatrixX const> const& v) { self.v() = v; },
            "|#dims|x|#nodes| array of nodal velocities")
        .def_property(
            "fext",
            [](FemElastoDynamics const& self) { return self.fext(); },
            [](FemElastoDynamics& self, Eigen::Ref<MatrixX const> const& fext) {
                self.fext() = fext;
            },
            "|#dims|x|#nodes| array of external forces at nodes")
        .def_property(
            "m",
            [](FemElastoDynamics const& self) { return self.m(); },
            [](FemElastoDynamics& self, Eigen::Ref<VectorX const> const& m) { self.m() = m; },
            "|#nodes| array of lumped mass matrix")
        .def_property(
            "elastic_quadrature",
            [](FemElastoDynamics const& self) { return self.QU(); },
            [](FemElastoDynamics& self, pbat::sim::dynamics::ElasticityQuadrature const& QU) {
                self.QU() = QU;
            },
            "Elasticity quadrature")
        .def_property_readonly("U", &FemElastoDynamics::U)
        .def(
            "with_initial_conditions",
            [](FemElastoDynamics& self,
               Eigen::Ref<MatrixX const> const& x0,
               Eigen::Ref<MatrixX const> const& v0) -> FemElastoDynamics& {
                self.SetInitialConditions(x0, v0);
                return self;
            },
            pyb::arg("x0"),
            pyb::arg("v0"),
            "Set initial conditions\n\n"
            "Args:\n"
            "    x0 (array): `|#dims| x |#nodes|` array of nodal positions\n"
            "    v0 (array): `|#dims| x |#nodes|` array of nodal velocities\n\n"
            "Returns:\n"
            "    pbat.sim.dynamics.FemElastoDynamics\n\n")
        .def(
            "with_mass_matrix",
            [](FemElastoDynamics& self, Scalar rho) -> FemElastoDynamics& {
                self.SetMassMatrix(rho);
                return self;
            },
            pyb::arg("rho") = 1e3,
            "Set mass matrix with constant density\n\n"
            "Args:\n"
            "    rho (float): Density of the material\n\n"
            "Returns:\n"
            "    pbat.sim.dynamics.FemElastoDynamics\n\n")
        .def(
            "with_mass_matrix",
            [](FemElastoDynamics& self,
               Eigen::Ref<IndexVectorX const> const& eg,
               Eigen::Ref<VectorX const> const& wg,
               Eigen::Ref<MatrixX const> const& Xg,
               Eigen::Ref<VectorX const> const& rhog) -> FemElastoDynamics& {
                self.SetMassMatrix(eg, wg, Xg, rhog);
                return self;
            },
            pyb::arg("eg"),
            pyb::arg("wg"),
            pyb::arg("Xg"),
            pyb::arg("rhog"),
            "Set mass matrix with variable density\n\n"
            "Args:\n"
            "    eg (array): `|# quad.pts.| x 1` vector of element indices at quadrature points\n"
            "    wg (array): `|# quad.pts.| x 1` vector of quadrature weights\n"
            "    Xg (array): `|# dims| x |# quad.pts.|` matrix of quadrature points\n"
            "    rhog (array): `|# quad.pts.| x 1` vector of mass density at quadrature points\n\n"
            "Returns:\n"
            "    pbat.sim.dynamics.FemElastoDynamics\n\n")
        .def(
            "with_elastic_energy",
            [](FemElastoDynamics& self, Scalar mu, Scalar lambda, bool bWithElasticPotential)
                -> FemElastoDynamics& {
                self.SetElasticEnergy(mu, lambda, bWithElasticPotential);
                return self;
            },
            pyb::arg("mu"),
            pyb::arg("lambda"),
            pyb::arg("bWithElasticPotential") = true,
            "Set elastic energy with constant Lame coefficients\n\n"
            "Args:\n"
            "    mu (float): 1st Lame coefficient\n"
            "    lambda (float): 2nd Lame coefficient\n"
            "    bWithElasticPotential (bool): If True, construct the elastic potential `U`\n\n"
            "Returns:\n"
            "    pbat.sim.dynamics.FemElastoDynamics\n\n")
        .def(
            "with_elastic_energy",
            [](FemElastoDynamics& self,
               Eigen::Ref<IndexVectorX const> const& eg,
               Eigen::Ref<VectorX const> const& wg,
               Eigen::Ref<MatrixX const> const& Xg,
               Eigen::Ref<VectorX const> const& mug,
               Eigen::Ref<VectorX const> const& lambdag,
               bool bWithElasticPotential) -> FemElastoDynamics& {
                self.SetElasticEnergy(eg, wg, Xg, mug, lambdag, bWithElasticPotential);
                return self;
            },
            pyb::arg("eg"),
            pyb::arg("wg"),
            pyb::arg("Xg"),
            pyb::arg("mug"),
            pyb::arg("lambdag"),
            pyb::arg("bWithElasticPotential") = true,
            "Set elastic energy with variable Lame coefficients\n\n"
            "Args:\n"
            "    eg (array): `|# quad.pts.| x 1` vector of element indices at quadrature points\n"
            "    wg (array): `|# quad.pts.| x 1` vector of quadrature weights\n"
            "    Xg (array): `|# dims| x |# quad.pts.|` matrix of quadrature points\n"
            "    mug (array): `|# quad.pts.| x 1` vector of 1st Lame coefficients at quadrature "
            "points\n"
            "    lambdag (array): `|# quad.pts.| x 1` vector of 2nd Lame coefficients at "
            "quadrature points\n"
            "    bWithElasticPotential (bool): If True, construct the elastic potential `U`\n\n"
            "Returns:\n"
            "    pbat.sim.dynamics.FemElastoDynamics\n\n")
        .def(
            "with_external_load",
            [](FemElastoDynamics& self,
               Eigen::Ref<MatrixX const> const& fext) -> FemElastoDynamics& {
                self.SetExternalLoad(fext);
                return self;
            },
            pyb::arg("fext"),
            "Set external load vector using homogeneous body forces fext\n\n"
            "Args:\n"
            "    fext (array): `|#dims| x 1` array of uniform external force\n\n"
            "Returns:\n"
            "    pbat.sim.dynamics.FemElastoDynamics\n\n")
        .def(
            "with_external_load",
            [](FemElastoDynamics& self,
               Eigen::Ref<IndexVectorX const> const& eg,
               Eigen::Ref<VectorX const> const& wg,
               Eigen::Ref<MatrixX const> const& Xg,
               Eigen::Ref<MatrixX const> const& bg) -> FemElastoDynamics& {
                self.SetExternalLoad(eg, wg, Xg, bg);
                return self;
            },
            pyb::arg("eg"),
            pyb::arg("wg"),
            pyb::arg("Xg"),
            pyb::arg("bg"),
            "Set external load vector using variable body forces\n\n"
            "Args:\n"
            "    eg (array): `|# quad.pts.| x 1` vector of element indices at quadrature points\n"
            "    wg (array): `|# quad.pts.| x 1` vector of quadrature weights\n"
            "    Xg (array): `|# dims| x |# quad.pts.|` matrix of quadrature points\n"
            "    bg (array): `|# dims| x |# quad.pts.|` matrix of body forces at quadrature "
            "points\n\n"
            "Returns:\n"
            "    pbat.sim.dynamics.FemElastoDynamics\n\n")
        .def(
            "with_time_integration_scheme",
            [](FemElastoDynamics& self, Scalar dt, int s) -> FemElastoDynamics& {
                self.SetTimeIntegrationScheme(dt, s);
                return self;
            },
            pyb::arg("dt") = 1e-2,
            pyb::arg("s")  = 1,
            "Set time integration scheme\n\n"
            "Args:\n"
            "    dt (float): Time step size\n"
            "    s (int): BDF step\n\n"
            "Returns:\n"
            "    pbat.sim.dynamics.FemElastoDynamics\n\n")
        .def(
            "with_dirichlet_boundary_conditions",
            [](FemElastoDynamics& self,
               Eigen::Ref<Eigen::Vector<bool, Eigen::Dynamic> const> const& D)
                -> FemElastoDynamics& {
                self.Constrain(D);
                return self;
            },
            pyb::arg("D"),
            "Set Dirichlet boundary conditions\n\n"
            "Args:\n"
            "    D (array): `|# nodes| x 1` mask of Dirichlet boundary conditions s.t. "
            "`D(i) == true` if node `i` is constrained\n\n"
            "Returns:\n"
            "    pbat.sim.dynamics.FemElastoDynamics\n\n")
        .def_readonly("dims", &FemElastoDynamics::mDims)
        .def_readonly("order", &FemElastoDynamics::mOrder)
        .def_readonly("element", &FemElastoDynamics::mElement)
        .def_readonly("psi", &FemElastoDynamics::mHyperElasticEnergy);
}

FemElastoDynamics::FemElastoDynamics(
    int meshDims,
    int meshOrder,
    fem::EElement meshElement,
    fem::EHyperElasticEnergy hyperElasticEnergy)
    : mDims(meshDims),
      mOrder(meshOrder),
      mElement(meshElement),
      mHyperElasticEnergy(hyperElasticEnergy),
      mFemElastoDynamics(nullptr)
{
    Access([&]<class FemElastoDynamicsType>([[maybe_unused]] FemElastoDynamicsType* impl) {
        mFemElastoDynamics = new FemElastoDynamicsType();
    });
}

FemElastoDynamics::FemElastoDynamics(FemElastoDynamics&& other)
    : mDims(other.mDims),
      mOrder(other.mOrder),
      mElement(other.mElement),
      mHyperElasticEnergy(other.mHyperElasticEnergy),
      mFemElastoDynamics(other.mFemElastoDynamics)
{
    other.mFemElastoDynamics = nullptr;
}

FemElastoDynamics& FemElastoDynamics::operator=(FemElastoDynamics&& other)
{
    if (this != std::addressof(other))
    {
        mDims                    = other.mDims;
        mOrder                   = other.mOrder;
        mElement                 = other.mElement;
        mHyperElasticEnergy      = other.mHyperElasticEnergy;
        mFemElastoDynamics       = other.mFemElastoDynamics;
        other.mFemElastoDynamics = nullptr;
    }
    return *this;
}

fem::Mesh FemElastoDynamics::Mesh() const
{
    void* meshImpl{nullptr};
    Access([&]<class FemElastoDynamicsType>(FemElastoDynamicsType* femElastoDynamics) {
        meshImpl = std::addressof(femElastoDynamics->mesh);
    });
    return fem::Mesh(meshImpl, mElement, mOrder, mDims);
}

Eigen::Map<MatrixX const> FemElastoDynamics::x() const
{
    Scalar* data{nullptr};
    Index rows{-1};
    Index cols{-1};
    Access([&]<class FemElastoDynamicsType>(FemElastoDynamicsType* femElastoDynamics) {
        data = femElastoDynamics->x.data();
        rows = femElastoDynamics->x.rows();
        cols = femElastoDynamics->x.cols();
    });
    return Eigen::Map<MatrixX const>(data, rows, cols);
}

Eigen::Map<MatrixX> FemElastoDynamics::x()
{
    Scalar* data{nullptr};
    Index rows{-1};
    Index cols{-1};
    Access([&]<class FemElastoDynamicsType>(FemElastoDynamicsType* femElastoDynamics) {
        data = femElastoDynamics->x.data();
        rows = femElastoDynamics->x.rows();
        cols = femElastoDynamics->x.cols();
    });
    return Eigen::Map<MatrixX>(data, rows, cols);
}

Eigen::Map<MatrixX const> FemElastoDynamics::v() const
{
    Scalar* data{nullptr};
    Index rows{-1};
    Index cols{-1};
    Access([&]<class FemElastoDynamicsType>(FemElastoDynamicsType* femElastoDynamics) {
        data = femElastoDynamics->v.data();
        rows = femElastoDynamics->v.rows();
        cols = femElastoDynamics->v.cols();
    });
    return Eigen::Map<MatrixX const>(data, rows, cols);
}

Eigen::Map<MatrixX> FemElastoDynamics::v()
{
    Scalar* data{nullptr};
    Index rows{-1};
    Index cols{-1};
    Access([&]<class FemElastoDynamicsType>(FemElastoDynamicsType* femElastoDynamics) {
        data = femElastoDynamics->v.data();
        rows = femElastoDynamics->v.rows();
        cols = femElastoDynamics->v.cols();
    });
    return Eigen::Map<MatrixX>(data, rows, cols);
}

Eigen::Map<MatrixX const> FemElastoDynamics::fext() const
{
    Scalar* data{nullptr};
    Index rows{-1};
    Index cols{-1};
    Access([&]<class FemElastoDynamicsType>(FemElastoDynamicsType* femElastoDynamics) {
        data = femElastoDynamics->fext.data();
        rows = femElastoDynamics->fext.rows();
        cols = femElastoDynamics->fext.cols();
    });
    return Eigen::Map<MatrixX const>(data, rows, cols);
}

Eigen::Map<MatrixX> FemElastoDynamics::fext()
{
    Scalar* data{nullptr};
    Index rows{-1};
    Index cols{-1};
    Access([&]<class FemElastoDynamicsType>(FemElastoDynamicsType* femElastoDynamics) {
        data = femElastoDynamics->fext.data();
        rows = femElastoDynamics->fext.rows();
        cols = femElastoDynamics->fext.cols();
    });
    return Eigen::Map<MatrixX>(data, rows, cols);
}

Eigen::Map<VectorX const> FemElastoDynamics::m() const
{
    Scalar* data{nullptr};
    Index rows{-1};
    Index cols{-1};
    Access([&]<class FemElastoDynamicsType>(FemElastoDynamicsType* femElastoDynamics) {
        data = femElastoDynamics->m.data();
        rows = femElastoDynamics->m.rows();
        cols = femElastoDynamics->m.cols();
    });
    return Eigen::Map<VectorX const>(data, rows, cols);
}

Eigen::Map<VectorX> FemElastoDynamics::m()
{
    Scalar* data{nullptr};
    Index rows{-1};
    Index cols{-1};
    Access([&]<class FemElastoDynamicsType>(FemElastoDynamicsType* femElastoDynamics) {
        data = femElastoDynamics->m.data();
        rows = femElastoDynamics->m.rows();
        cols = femElastoDynamics->m.cols();
    });
    return Eigen::Map<VectorX>(data, rows, cols);
}

pbat::sim::dynamics::ElasticityQuadrature const& FemElastoDynamics::QU() const
{
    pbat::sim::dynamics::ElasticityQuadrature const* QUPtr{nullptr};
    Access([&]<class FemElastoDynamicsType>(FemElastoDynamicsType* femElastoDynamics) {
        QUPtr = std::addressof(femElastoDynamics->QU);
    });
    return *QUPtr;
}

pbat::sim::dynamics::ElasticityQuadrature& FemElastoDynamics::QU()
{
    pbat::sim::dynamics::ElasticityQuadrature* QUPtr{nullptr};
    Access([&]<class FemElastoDynamicsType>(FemElastoDynamicsType* femElastoDynamics) {
        QUPtr = std::addressof(femElastoDynamics->QU);
    });
    return *QUPtr;
}

std::optional<fem::HyperElasticPotential> FemElastoDynamics::U() const
{
    std::optional<fem::HyperElasticPotential> U;
    Access([&]<class FemElastoDynamicsType>(FemElastoDynamicsType* femElastoDynamics) {
        if (femElastoDynamics->U)
        {
            U.emplace(
                std::addressof(femElastoDynamics->U.value()),
                mElement,
                mOrder,
                mDims,
                mHyperElasticEnergy);
        }
    });
    return U;
}

IndexVectorX FemElastoDynamics::FreeNodes() const
{
    IndexVectorX freeNodes{};
    Access([&]<class FemElastoDynamicsType>(FemElastoDynamicsType* femElastoDynamics) {
        freeNodes = femElastoDynamics->FreeNodes();
    });
    return freeNodes;
}

IndexVectorX FemElastoDynamics::DirichletNodes() const
{
    IndexVectorX dirichletNodes{};
    Access([&]<class FemElastoDynamicsType>(FemElastoDynamicsType* femElastoDynamics) {
        dirichletNodes = femElastoDynamics->DirichletNodes();
    });
    return dirichletNodes;
}

pbat::sim::integration::Bdf const& FemElastoDynamics::Bdf() const
{
    pbat::sim::integration::Bdf const* bdfPtr{nullptr};
    Access([&]<class FemElastoDynamicsType>(FemElastoDynamicsType* femElastoDynamics) {
        bdfPtr = std::addressof(femElastoDynamics->bdf);
    });
    return *bdfPtr;
}

pbat::sim::integration::Bdf& FemElastoDynamics::Bdf()
{
    pbat::sim::integration::Bdf* bdfPtr{nullptr};
    Access([&]<class FemElastoDynamicsType>(FemElastoDynamicsType* femElastoDynamics) {
        bdfPtr = std::addressof(femElastoDynamics->bdf);
    });
    return *bdfPtr;
}

void FemElastoDynamics::Construct(
    Eigen::Ref<MatrixX const> const& V,
    Eigen::Ref<IndexMatrixX const> const& C)
{
    Access([&]<class FemElastoDynamicsType>(FemElastoDynamicsType* femElastoDynamics) {
        femElastoDynamics->Construct(V, C);
    });
}

void FemElastoDynamics::SetInitialConditions(
    Eigen::Ref<MatrixX const> const& x0,
    Eigen::Ref<MatrixX const> const& v0)
{
    Access([&]<class FemElastoDynamicsType>(FemElastoDynamicsType* femElastoDynamics) {
        femElastoDynamics->SetInitialConditions(x0, v0);
    });
}

void FemElastoDynamics::SetMassMatrix(Scalar rho)
{
    Access([&]<class FemElastoDynamicsType>(FemElastoDynamicsType* femElastoDynamics) {
        femElastoDynamics->SetMassMatrix(rho);
    });
}

void FemElastoDynamics::SetElasticEnergy(Scalar mu, Scalar lambda, bool bWithElasticPotential)
{
    Access([&]<class FemElastoDynamicsType>(FemElastoDynamicsType* femElastoDynamics) {
        femElastoDynamics->SetElasticEnergy(mu, lambda, bWithElasticPotential);
    });
}

void FemElastoDynamics::SetExternalLoad(Eigen::Ref<MatrixX const> const& fext)
{
    Access([&]<class FemElastoDynamicsType>(FemElastoDynamicsType* femElastoDynamics) {
        femElastoDynamics->SetExternalLoad(fext);
    });
}

void FemElastoDynamics::SetTimeIntegrationScheme(Scalar dt, int s)
{
    Access([&]<class FemElastoDynamicsType>(FemElastoDynamicsType* femElastoDynamics) {
        femElastoDynamics->SetTimeIntegrationScheme(dt, s);
    });
}

void FemElastoDynamics::Constrain(Eigen::Ref<Eigen::Vector<bool, Eigen::Dynamic> const> const& D)
{
    Access([&]<class FemElastoDynamicsType>(FemElastoDynamicsType* femElastoDynamics) {
        femElastoDynamics->Constrain(D);
    });
}

void FemElastoDynamics::SetMassMatrix(
    Eigen::Ref<IndexVectorX const> const& eg,
    Eigen::Ref<VectorX const> const& wg,
    Eigen::Ref<MatrixX const> const& Xg,
    Eigen::Ref<VectorX const> const& rhog)
{
    Access([&]<class FemElastoDynamicsType>(FemElastoDynamicsType* femElastoDynamics) {
        femElastoDynamics->SetMassMatrix(eg, wg, Xg, rhog);
    });
}

void FemElastoDynamics::SetElasticEnergy(
    Eigen::Ref<IndexVectorX const> const& eg,
    Eigen::Ref<VectorX const> const& wg,
    Eigen::Ref<MatrixX const> const& Xg,
    Eigen::Ref<VectorX const> const& mug,
    Eigen::Ref<VectorX const> const& lambdag,
    bool bWithElasticPotential)
{
    Access([&]<class FemElastoDynamicsType>(FemElastoDynamicsType* femElastoDynamics) {
        femElastoDynamics->SetElasticEnergy(eg, wg, Xg, mug, lambdag, bWithElasticPotential);
    });
}

void FemElastoDynamics::SetExternalLoad(
    Eigen::Ref<IndexVectorX const> const& eg,
    Eigen::Ref<VectorX const> const& wg,
    Eigen::Ref<MatrixX const> const& Xg,
    Eigen::Ref<MatrixX const> const& bg)
{
    Access([&]<class FemElastoDynamicsType>(FemElastoDynamicsType* femElastoDynamics) {
        femElastoDynamics->SetExternalLoad(eg, wg, Xg, bg);
    });
}

void FemElastoDynamics::Serialize(io::Archive& archive) const
{
    Access([&]<class FemElastoDynamicsType>(FemElastoDynamicsType* femElastoDynamics) {
        femElastoDynamics->Serialize(archive);
    });
}

void FemElastoDynamics::Deserialize(io::Archive const& archive)
{
    Access([&]<class FemElastoDynamicsType>(FemElastoDynamicsType* femElastoDynamics) {
        femElastoDynamics->Deserialize(archive);
    });
}

FemElastoDynamics::~FemElastoDynamics()
{
    if (mFemElastoDynamics)
    {
        Access([&]<class FemElastoDynamicsType>(FemElastoDynamicsType* femElastoDynamics) {
            delete femElastoDynamics;
        });
    }
}

} // namespace pbat::py::sim::dynamics