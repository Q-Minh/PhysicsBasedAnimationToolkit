#include "HyperElasticPotential.h"

#include <pybind11/eigen.h>

namespace pbat {
namespace py {
namespace fem {

void BindHyperElasticPotential(pybind11::module& m)
{
    namespace pyb = pybind11;

    pyb::enum_<EHyperElasticEnergy>(m, "HyperElasticEnergy")
        .value("SaintVenantKirchhoff", EHyperElasticEnergy::SaintVenantKirchhoff)
        .value("StableNeoHookean", EHyperElasticEnergy::StableNeoHookean)
        .export_values();

    pyb::class_<HyperElasticPotential>(m, "HyperElasticPotential")
        .def(
            pyb::init<
                Mesh const&,
                Eigen::Ref<IndexVectorX const> const&,
                Eigen::Ref<VectorX const> const&,
                Eigen::Ref<MatrixX const> const&,
                Eigen::Ref<MatrixX const> const&,
                EHyperElasticEnergy>(),
            pyb::arg("mesh"),
            pyb::arg("eg").noconvert(),
            pyb::arg("wg").noconvert(),
            pyb::arg("GNeg").noconvert(),
            pyb::arg("lameg").noconvert(),
            pyb::arg("energy") = EHyperElasticEnergy::StableNeoHookean,
            "Construct a HyperElasticPotential on mesh mesh, given precomputed quadrature "
            "weights wg in elements eg and shape function gradients GNeg at mesh element "
            "quadrature points. The corresponding energy has piecewise constant (at elements) "
            "Young's modulus Y and Poisson's ratio nu.")
        .def_readonly("dims", &HyperElasticPotential::mDims)
        .def(
            "precompute_hessian_sparsity",
            &HyperElasticPotential::PrecomputeHessianSparsity,
            "Precompute sparsity pattern of the hessian for reusable and efficient hessian "
            "construction.")
        .def(
            "compute_element_elasticity",
            &HyperElasticPotential::ComputeElementElasticity,
            pyb::arg("x"),
            pyb::arg("grad")    = true,
            pyb::arg("hessian") = true,
            pyb::arg("spd")     = true,
            "Compute per-element potential energy and its derivatives, projecting the hessian to a "
            "positive definite state if spd=True.")
        .def("eval", &HyperElasticPotential::Eval)
        .def("gradient", &HyperElasticPotential::ToVector)
        .def("hessian", &HyperElasticPotential::ToMatrix)
        .def_property_readonly(
            "Ug",
            [](HyperElasticPotential const& M) { return M.Potentials(); },
            "|#quad.pts.| vector of hyper elastic potentials at quadrature points")
        .def_property_readonly(
            "Gg",
            [](HyperElasticPotential const& M) { return M.Gradients(); },
            "|#element nodes * #dims|x|#quad.pts.| matrix of element hyper elastic potential "
            "gradients at quadrature points")
        .def_property_readonly(
            "Hg",
            [](HyperElasticPotential const& M) { return M.Hessians(); },
            "|#element nodes * dims|x|#elements nodes * dims * #quad.pts.| matrix of element hyper "
            "elastic potential hessians at quadrature points")
        .def_property_readonly("shape", &HyperElasticPotential::Shape)
        .def("to_matrix", &HyperElasticPotential::ToMatrix);
}

HyperElasticPotential::HyperElasticPotential(
    Mesh const& M,
    Eigen::Ref<IndexVectorX const> const& eg,
    Eigen::Ref<VectorX const> const& wg,
    Eigen::Ref<MatrixX const> const& GNeg,
    Eigen::Ref<MatrixX const> const& lameg,
    EHyperElasticEnergy ePsi)
    : eMeshElement(M.eElement),
      mMeshDims(M.mDims),
      mMeshOrder(M.mOrder),
      eHyperElasticEnergy(ePsi),
      mDims(),
      mHyperElasticPotential(nullptr),
      bOwning(true)
{
    M.Apply([&]<pbat::fem::CMesh MeshType>(MeshType* mesh) {
        pbat::common::ForTypes<
            pbat::physics::SaintVenantKirchhoffEnergy<MeshType::kDims>,
            pbat::physics::StableNeoHookeanEnergy<MeshType::kDims>>(
            [&]<class HyperElasticEnergyType>() {
                using HyperElasticPotentialType =
                    pbat::fem::HyperElasticPotential<MeshType, HyperElasticEnergyType>;
                if (ePsi == EHyperElasticEnergy::SaintVenantKirchhoff and
                    std::is_same_v<
                        HyperElasticEnergyType,
                        pbat::physics::SaintVenantKirchhoffEnergy<MeshType::kDims>>)
                {
                    mHyperElasticPotential =
                        new HyperElasticPotentialType(*mesh, eg, wg, GNeg, lameg);
                    mDims = HyperElasticPotentialType::kDims;
                }
                if (ePsi == EHyperElasticEnergy::StableNeoHookean and
                    std::is_same_v<
                        HyperElasticEnergyType,
                        pbat::physics::StableNeoHookeanEnergy<MeshType::kDims>>)
                {
                    mHyperElasticPotential =
                        new HyperElasticPotentialType(*mesh, eg, wg, GNeg, lameg);
                    mDims = HyperElasticPotentialType::kDims;
                }
            });
    });
}

HyperElasticPotential::HyperElasticPotential(
    void* impl,
    EElement meshElement,
    int meshOrder,
    int meshDims,
    EHyperElasticEnergy eHyperElasticEnergy)
    : eMeshElement(meshElement),
      mMeshDims(meshDims),
      mMeshOrder(meshOrder),
      eHyperElasticEnergy(eHyperElasticEnergy),
      mDims(meshDims),
      mHyperElasticPotential(impl),
      bOwning(false)
{
}

HyperElasticPotential::HyperElasticPotential(HyperElasticPotential&& other)
    : eMeshElement(other.eMeshElement),
      mMeshDims(other.mMeshDims),
      mMeshOrder(other.mMeshOrder),
      eHyperElasticEnergy(other.eHyperElasticEnergy),
      mDims(other.mDims),
      mHyperElasticPotential(other.mHyperElasticPotential),
      bOwning(other.bOwning)
{
    other.mHyperElasticPotential = nullptr;
    other.bOwning                = false;
}

HyperElasticPotential& HyperElasticPotential::operator=(HyperElasticPotential&& other)
{
    if (this != std::addressof(other))
    {
        eMeshElement                 = other.eMeshElement;
        mMeshDims                    = other.mMeshDims;
        mMeshOrder                   = other.mMeshOrder;
        eHyperElasticEnergy          = other.eHyperElasticEnergy;
        mDims                        = other.mDims;
        mHyperElasticPotential       = other.mHyperElasticPotential;
        other.mHyperElasticPotential = nullptr;
        bOwning                      = other.bOwning;
        other.bOwning                = false;
    }
    return *this;
}

void HyperElasticPotential::PrecomputeHessianSparsity()
{
    Apply([]<class HyperElasticPotentialType>(HyperElasticPotentialType* hyperElasticPotential) {
        hyperElasticPotential->PrecomputeHessianSparsity();
    });
}

void HyperElasticPotential::ComputeElementElasticity(
    Eigen::Ref<VectorX const> const& x,
    bool bWithGradient,
    bool bWithHessian,
    bool bWithSpdProjection)
{
    Apply([&]<class HyperElasticPotentialType>(HyperElasticPotentialType* hyperElasticPotential) {
        hyperElasticPotential->template ComputeElementElasticity<Eigen::Ref<VectorX const>>(
            x,
            bWithGradient,
            bWithHessian,
            bWithSpdProjection);
    });
}

Scalar HyperElasticPotential::Eval() const
{
    Scalar U{};
    Apply([&]<class HyperElasticPotentialType>(HyperElasticPotentialType* hyperElasticPotential) {
        U = hyperElasticPotential->Eval();
    });
    return U;
}

VectorX HyperElasticPotential::ToVector() const
{
    VectorX G{};
    Apply([&]<class HyperElasticPotentialType>(HyperElasticPotentialType* hyperElasticPotential) {
        G = hyperElasticPotential->ToVector();
    });
    return G;
}

CSCMatrix HyperElasticPotential::ToMatrix() const
{
    CSCMatrix H;
    Apply([&]<class HyperElasticPotentialType>(HyperElasticPotentialType* hyperElasticPotential) {
        H = hyperElasticPotential->ToMatrix();
    });
    return H;
}

std::tuple<Index, Index> HyperElasticPotential::Shape() const
{
    Index rows{0}, cols{0};
    Apply([&]<class HyperElasticPotentialType>(HyperElasticPotentialType* hyperElasticPotential) {
        rows = hyperElasticPotential->OutputDimensions();
        cols = hyperElasticPotential->InputDimensions();
    });
    return std::make_tuple(rows, cols);
}

MatrixX const& HyperElasticPotential::Hessians() const
{
    MatrixX* HgPtr;
    Apply([&]<class HyperElasticPotentialType>(HyperElasticPotentialType* hyperElasticPotential) {
        HgPtr = std::addressof(hyperElasticPotential->Hg);
    });
    return *HgPtr;
}

MatrixX& HyperElasticPotential::Hessians()
{
    MatrixX* HgPtr;
    Apply([&]<class HyperElasticPotentialType>(HyperElasticPotentialType* hyperElasticPotential) {
        HgPtr = std::addressof(hyperElasticPotential->Hg);
    });
    return *HgPtr;
}

MatrixX const& HyperElasticPotential::Gradients() const
{
    MatrixX* ggPtr;
    Apply([&]<class HyperElasticPotentialType>(HyperElasticPotentialType* hyperElasticPotential) {
        ggPtr = std::addressof(hyperElasticPotential->Gg);
    });
    return *ggPtr;
}

MatrixX& HyperElasticPotential::Gradients()
{
    MatrixX* ggPtr;
    Apply([&]<class HyperElasticPotentialType>(HyperElasticPotentialType* hyperElasticPotential) {
        ggPtr = std::addressof(hyperElasticPotential->Gg);
    });
    return *ggPtr;
}

VectorX const& HyperElasticPotential::Potentials() const
{
    VectorX* UgPtr;
    Apply([&]<class HyperElasticPotentialType>(HyperElasticPotentialType* hyperElasticPotential) {
        UgPtr = std::addressof(hyperElasticPotential->Ug);
    });
    return *UgPtr;
}

VectorX& HyperElasticPotential::Potentials()
{
    VectorX* UgPtr;
    Apply([&]<class HyperElasticPotentialType>(HyperElasticPotentialType* hyperElasticPotential) {
        UgPtr = std::addressof(hyperElasticPotential->Ug);
    });
    return *UgPtr;
}

HyperElasticPotential::~HyperElasticPotential()
{
    if (mHyperElasticPotential != nullptr and bOwning)
    {
        Apply(
            [&]<class HyperElasticPotentialType>(HyperElasticPotentialType* hyperElasticPotential) {
                delete hyperElasticPotential;
            });
    }
}

template <class Func>
void HyperElasticPotential::Apply(Func&& f) const
{
    HepApplyToMesh(mMeshDims, mMeshOrder, eMeshElement, [&]<pbat::fem::CMesh MeshType>() {
        pbat::common::ForTypes<
            pbat::physics::SaintVenantKirchhoffEnergy<MeshType::kDims>,
            pbat::physics::StableNeoHookeanEnergy<MeshType::kDims>>(
            [&]<class HyperElasticEnergyType>() {
                using HyperElasticPotentialType =
                    pbat::fem::HyperElasticPotential<MeshType, HyperElasticEnergyType>;
                if (eHyperElasticEnergy == EHyperElasticEnergy::SaintVenantKirchhoff and
                    std::is_same_v<
                        HyperElasticEnergyType,
                        pbat::physics::SaintVenantKirchhoffEnergy<MeshType::kDims>>)
                {
                    HyperElasticPotentialType* hyperElasticPotential =
                        reinterpret_cast<HyperElasticPotentialType*>(mHyperElasticPotential);
                    f.template operator()<HyperElasticPotentialType>(hyperElasticPotential);
                }
                if (eHyperElasticEnergy == EHyperElasticEnergy::StableNeoHookean and
                    std::is_same_v<
                        HyperElasticEnergyType,
                        pbat::physics::StableNeoHookeanEnergy<MeshType::kDims>>)
                {
                    HyperElasticPotentialType* hyperElasticPotential =
                        reinterpret_cast<HyperElasticPotentialType*>(mHyperElasticPotential);
                    f.template operator()<HyperElasticPotentialType>(hyperElasticPotential);
                }
            });
    });
}

} // namespace fem
} // namespace py
} // namespace pbat