#include "HyperElasticPotential.h"

#include "Mesh.h"

#include <pbat/fem/HyperElasticPotential.h>
#include <pbat/physics/SaintVenantKirchhoffEnergy.h>
#include <pbat/physics/StableNeoHookeanEnergy.h>
#include <pybind11/eigen.h>
#include <tuple>
#include <type_traits>

namespace pbat {
namespace py {
namespace fem {

template <class Func>
inline void HepApplyToMesh(int meshDims, int meshOrder, EElement meshElement, Func&& f)
{
    using namespace pbat::fem;
    using namespace pbat::common;
    ForValues<1, 2, 3>([&]<auto Order>() {
        ForTypes<Line<Order>, Triangle<Order>, Quadrilateral<Order>, Tetrahedron<Order>>(
            [&]<CElement ElementType>() {
                EElement const eElementCandidate = []() {
                    if constexpr (std::is_same_v<ElementType, Line<Order>>)
                        return EElement::Line;
                    if constexpr (std::is_same_v<ElementType, Triangle<Order>>)
                        return EElement::Triangle;
                    if constexpr (std::is_same_v<ElementType, Quadrilateral<Order>>)
                        return EElement::Quadrilateral;
                    if constexpr (std::is_same_v<ElementType, Tetrahedron<Order>>)
                        return EElement::Tetrahedron;
                }();

                auto constexpr DimsIn = ElementType::kDims;
                if ((meshOrder == Order) and (meshDims == DimsIn) and
                    (meshElement == eElementCandidate))
                {
                    using MeshType = pbat::fem::Mesh<ElementType, DimsIn>;
                    f.template operator()<MeshType>();
                }
            });
    });

    // Elasticity can't handle cubic (or higher) hexahedra, because elastic hessians are stored on
    // the stack, and 3rd (or higher) order hexahedra have too many DOFs for the stack to allocate.
    ForValues<1, 2>([&]<auto Order>() {
        ForTypes<Hexahedron<Order>>([&]<CElement ElementType>() {
            EElement const eElementCandidate = EElement::Hexahedron;

            auto constexpr DimsIn = ElementType::kDims;
            if ((meshOrder == Order) and (meshDims == DimsIn) and
                (meshElement == eElementCandidate))
            {
                using MeshType = pbat::fem::Mesh<ElementType, DimsIn>;
                f.template operator()<MeshType>();
            }
        });
    });
}

enum class EHyperElasticEnergy { SaintVenantKirchhoff, StableNeoHookean };

class HyperElasticPotential
{
  public:
    HyperElasticPotential(
        Mesh const& M,
        Eigen::Ref<IndexVectorX const> const& eg,
        Eigen::Ref<VectorX const> const& wg,
        Eigen::Ref<MatrixX const> const& GNeg,
        Scalar Y,
        Scalar nu,
        EHyperElasticEnergy eHyperElasticEnergy);

    HyperElasticPotential(
        Mesh const& M,
        Eigen::Ref<IndexVectorX const> const& eg,
        Eigen::Ref<VectorX const> const& wg,
        Eigen::Ref<MatrixX const> const& GNeg,
        Eigen::Ref<MatrixX const> const& Y,
        Eigen::Ref<MatrixX const> const& nu,
        EHyperElasticEnergy eHyperElasticEnergy);

    HyperElasticPotential(HyperElasticPotential&& other);
    HyperElasticPotential& operator=(HyperElasticPotential&& other);

    HyperElasticPotential(HyperElasticPotential const&)            = delete;
    HyperElasticPotential& operator=(HyperElasticPotential const&) = delete;

    template <class Func>
    void Apply(Func&& f) const;

    void PrecomputeHessianSparsity();

    void ComputeElementElasticity(
        Eigen::Ref<VectorX const> const& x,
        bool bWithGradient,
        bool bWithHessian,
        bool bWithSpdProjection);

    Scalar Eval() const;
    VectorX ToVector() const;
    CSCMatrix ToMatrix() const;
    std::tuple<Index, Index> Shape() const;

    VectorX const& mug() const;
    VectorX& mug();

    VectorX const& lambdag() const;
    VectorX& lambdag();

    MatrixX const& Hessians() const;
    MatrixX& Hessians();

    MatrixX const& Gradients() const;
    MatrixX& Gradients();

    VectorX const& Potentials() const;
    VectorX& Potentials();

    ~HyperElasticPotential();

    EElement eMeshElement;
    int mMeshDims;
    int mMeshOrder;
    EHyperElasticEnergy eHyperElasticEnergy;
    int mDims;

  private:
    void* mHyperElasticPotential;
};

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
                Scalar,
                Scalar,
                EHyperElasticEnergy>(),
            pyb::arg("mesh"),
            pyb::arg("eg"),
            pyb::arg("wg"),
            pyb::arg("GNeg"),
            pyb::arg("Y")      = 1e6,
            pyb::arg("nu")     = 0.45,
            pyb::arg("energy") = EHyperElasticEnergy::StableNeoHookean,
            "Construct a HyperElasticPotential on mesh mesh, given precomputed quadrature "
            "weights wg in elements eg and shape function gradients GNeg at mesh element "
            "quadrature points. The corresponding energy has Young's modulus Y and Poisson's ratio "
            "nu.")
        .def(
            pyb::init<
                Mesh const&,
                Eigen::Ref<IndexVectorX const> const&,
                Eigen::Ref<VectorX const> const&,
                Eigen::Ref<MatrixX const> const&,
                Eigen::Ref<MatrixX const> const&,
                Eigen::Ref<MatrixX const> const&,
                EHyperElasticEnergy>(),
            pyb::arg("mesh"),
            pyb::arg("eg"),
            pyb::arg("wg"),
            pyb::arg("GNeg"),
            pyb::arg("Y"),
            pyb::arg("nu"),
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
        .def_property(
            "mug",
            [](HyperElasticPotential const& M) { return M.mug(); },
            [](HyperElasticPotential& M, Eigen::Ref<MatrixX const> const& mug) { M.mug() = mug; },
            "|#quad.pts.|x1 array of first Lame coefficients")
        .def_property(
            "lambdag",
            [](HyperElasticPotential const& M) { return M.lambdag(); },
            [](HyperElasticPotential& M, Eigen::Ref<MatrixX const> const& lambdag) {
                M.lambdag() = lambdag;
            },
            "|#quad.pts.|x1 array of second Lame coefficients")
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
    Scalar Y,
    Scalar nu,
    EHyperElasticEnergy ePsi)
    : eMeshElement(M.eElement),
      mMeshDims(M.mDims),
      mMeshOrder(M.mOrder),
      eHyperElasticEnergy(ePsi),
      mDims(),
      mHyperElasticPotential(nullptr)
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
                        new HyperElasticPotentialType(*mesh, eg, wg, GNeg, Y, nu);
                    mDims = HyperElasticPotentialType::kDims;
                }
                if (ePsi == EHyperElasticEnergy::StableNeoHookean and
                    std::is_same_v<
                        HyperElasticEnergyType,
                        pbat::physics::StableNeoHookeanEnergy<MeshType::kDims>>)
                {
                    mHyperElasticPotential =
                        new HyperElasticPotentialType(*mesh, eg, wg, GNeg, Y, nu);
                    mDims = HyperElasticPotentialType::kDims;
                }
            });
    });
}

HyperElasticPotential::HyperElasticPotential(
    Mesh const& M,
    Eigen::Ref<IndexVectorX const> const& eg,
    Eigen::Ref<VectorX const> const& wg,
    Eigen::Ref<MatrixX const> const& GNeg,
    Eigen::Ref<MatrixX const> const& Y,
    Eigen::Ref<MatrixX const> const& nu,
    EHyperElasticEnergy ePsi)
    : eMeshElement(M.eElement),
      mMeshDims(M.mDims),
      mMeshOrder(M.mOrder),
      eHyperElasticEnergy(ePsi),
      mDims(),
      mHyperElasticPotential(nullptr)
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
                        new HyperElasticPotentialType(*mesh, eg, wg, GNeg, Y, nu);
                    mDims = HyperElasticPotentialType::kDims;
                }
                if (ePsi == EHyperElasticEnergy::StableNeoHookean and
                    std::is_same_v<
                        HyperElasticEnergyType,
                        pbat::physics::StableNeoHookeanEnergy<MeshType::kDims>>)
                {
                    mHyperElasticPotential =
                        new HyperElasticPotentialType(*mesh, eg, wg, GNeg, Y, nu);
                    mDims = HyperElasticPotentialType::kDims;
                }
            });
    });
}

HyperElasticPotential::HyperElasticPotential(HyperElasticPotential&& other)
    : mHyperElasticPotential(other.mHyperElasticPotential)
{
    other.mHyperElasticPotential = nullptr;
}

HyperElasticPotential& HyperElasticPotential::operator=(HyperElasticPotential&& other)
{
    mHyperElasticPotential       = other.mHyperElasticPotential;
    other.mHyperElasticPotential = nullptr;
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

VectorX const& HyperElasticPotential::mug() const
{
    VectorX* mugPtr;
    Apply([&]<class HyperElasticPotentialType>(HyperElasticPotentialType* hyperElasticPotential) {
        mugPtr = std::addressof(hyperElasticPotential->mug);
    });
    return *mugPtr;
}

VectorX& HyperElasticPotential::mug()
{
    VectorX* mugPtr;
    Apply([&]<class HyperElasticPotentialType>(HyperElasticPotentialType* hyperElasticPotential) {
        mugPtr = std::addressof(hyperElasticPotential->mug);
    });
    return *mugPtr;
}

VectorX const& HyperElasticPotential::lambdag() const
{
    VectorX* lambdagPtr;
    Apply([&]<class HyperElasticPotentialType>(HyperElasticPotentialType* hyperElasticPotential) {
        lambdagPtr = std::addressof(hyperElasticPotential->lambdag);
    });
    return *lambdagPtr;
}

VectorX& HyperElasticPotential::lambdag()
{
    VectorX* lambdagPtr;
    Apply([&]<class HyperElasticPotentialType>(HyperElasticPotentialType* hyperElasticPotential) {
        lambdagPtr = std::addressof(hyperElasticPotential->lambdag);
    });
    return *lambdagPtr;
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
    if (mHyperElasticPotential != nullptr)
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