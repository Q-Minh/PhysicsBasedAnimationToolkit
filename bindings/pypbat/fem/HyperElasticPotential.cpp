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

template <auto MaxQuadratureOrder, class Func>
inline void HepApplyToMeshWithQuadrature(
    int meshDims,
    int meshOrder,
    EElement meshElement,
    int qOrder,
    Func&& f)
{
    if (qOrder > MaxQuadratureOrder or qOrder < 1)
    {
        std::string const what = fmt::format(
            "Invalid quadrature order={}, supported orders are [1,{}]",
            qOrder,
            MaxQuadratureOrder);
        throw std::invalid_argument(what);
    }
    HepApplyToMesh(meshDims, meshOrder, meshElement, [&]<class MeshType>() {
        pbat::common::ForRange<1, MaxQuadratureOrder + 1>([&]<auto QuadratureOrder>() {
            if (qOrder == QuadratureOrder)
            {
                f.template operator()<MeshType, QuadratureOrder>();
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
        Eigen::Ref<MatrixX const> const& detJe,
        Eigen::Ref<MatrixX const> const& GNe,
        Scalar Y,
        Scalar nu,
        EHyperElasticEnergy eHyperElasticEnergy,
        int qOrder);

    HyperElasticPotential(
        Mesh const& M,
        Eigen::Ref<MatrixX const> const& detJe,
        Eigen::Ref<MatrixX const> const& GNe,
        Eigen::Ref<MatrixX const> const& Y,
        Eigen::Ref<MatrixX const> const& nu,
        EHyperElasticEnergy eHyperElasticEnergy,
        int qOrder);

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

    MatrixX const& mue() const;
    MatrixX& mue();

    MatrixX const& lambdae() const;
    MatrixX& lambdae();

    MatrixX const& ElementHessians() const;
    MatrixX& ElementHessians();

    MatrixX const& ElementGradients() const;
    MatrixX& ElementGradients();

    VectorX const& ElementPotentials() const;
    VectorX& ElementPotentials();

    ~HyperElasticPotential();

    EElement eMeshElement;
    int mMeshDims;
    int mMeshOrder;
    EHyperElasticEnergy eHyperElasticEnergy;
    int mDims;
    int mOrder;
    int mQuadratureOrder;

    static auto constexpr kMaxQuadratureOrder = 8;

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
                Eigen::Ref<MatrixX const> const&,
                Eigen::Ref<MatrixX const> const&,
                Scalar,
                Scalar,
                EHyperElasticEnergy,
                int>(),
            pyb::arg("mesh"),
            pyb::arg("detJe"),
            pyb::arg("GNe"),
            pyb::arg("Y")                = 1e6,
            pyb::arg("nu")               = 0.45,
            pyb::arg("energy")           = EHyperElasticEnergy::StableNeoHookean,
            pyb::arg("quadrature_order") = 1,
            "Construct a HyperElasticPotential on mesh mesh, given precomputed jacobian "
            "determinants detJe and shape function gradients GNe at mesh element quadrature points "
            "given by quadrature rule of order quadrature_order. The corresponding energy has "
            "Young's modulus Y and Poisson's ratio nu.")
        .def(
            pyb::init<
                Mesh const&,
                Eigen::Ref<MatrixX const> const&,
                Eigen::Ref<MatrixX const> const&,
                Eigen::Ref<MatrixX const> const&,
                Eigen::Ref<MatrixX const> const&,
                EHyperElasticEnergy,
                int>(),
            pyb::arg("mesh"),
            pyb::arg("detJe"),
            pyb::arg("GNe"),
            pyb::arg("Y"),
            pyb::arg("nu"),
            pyb::arg("energy")           = EHyperElasticEnergy::StableNeoHookean,
            pyb::arg("quadrature_order") = 1,
            "Construct a HyperElasticPotential on mesh mesh, given precomputed jacobian "
            "determinants detJe and shape function gradients GNe at mesh element quadrature points "
            "given by quadrature rule of order quadrature_order. The corresponding energy has "
            "piecewise constant (at elements) Young's modulus Y and Poisson's ratio nu.")
        .def_readonly("dims", &HyperElasticPotential::mDims)
        //.def_readonly("order", &HyperElasticPotential::mOrder)
        .def_readonly("quadrature_order", &HyperElasticPotential::mQuadratureOrder)
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
            "mue",
            [](HyperElasticPotential const& M) { return M.mue(); },
            [](HyperElasticPotential& M, Eigen::Ref<MatrixX const> const& mue) { M.mue() = mue; },
            "|#quad.pts.|x|#elements| array of first Lame coefficients")
        .def_property(
            "lambdae",
            [](HyperElasticPotential const& M) { return M.lambdae(); },
            [](HyperElasticPotential& M, Eigen::Ref<MatrixX const> const& lambdae) {
                M.lambdae() = lambdae;
            },
            "|#quad.pts.|x|#elements| array of second Lame coefficients")
        .def_property_readonly(
            "UE",
            [](HyperElasticPotential const& M) { return M.ElementPotentials(); },
            "|#elements| vector of element hyper elastic potentials")
        .def_property_readonly(
            "GE",
            [](HyperElasticPotential const& M) { return M.ElementGradients(); },
            "|#element nodes * dims|x|#elements| matrix of element hyper elastic potential "
            "gradients")
        .def_property_readonly(
            "HE",
            [](HyperElasticPotential const& M) { return M.ElementHessians(); },
            "|#element nodes * dims|x|#elements nodes * dims * #elements| matrix of element hyper "
            "elastic "
            "potential hessians")
        .def_property_readonly("shape", &HyperElasticPotential::Shape)
        .def("to_matrix", &HyperElasticPotential::ToMatrix);
}

HyperElasticPotential::HyperElasticPotential(
    Mesh const& M,
    Eigen::Ref<MatrixX const> const& detJe,
    Eigen::Ref<MatrixX const> const& GNe,
    Scalar Y,
    Scalar nu,
    EHyperElasticEnergy ePsi,
    int qOrder)
    : eMeshElement(M.eElement),
      mMeshDims(M.mDims),
      mMeshOrder(M.mOrder),
      eHyperElasticEnergy(ePsi),
      mDims(),
      mOrder(),
      mQuadratureOrder(),
      mHyperElasticPotential(nullptr)
{
    M.ApplyWithQuadrature<kMaxQuadratureOrder>(
        [&]<pbat::fem::CMesh MeshType, auto QuadratureOrder>(MeshType* mesh) {
            pbat::common::ForTypes<
                pbat::physics::SaintVenantKirchhoffEnergy<MeshType::kDims>,
                pbat::physics::StableNeoHookeanEnergy<MeshType::kDims>>(
                [&]<class HyperElasticEnergyType>() {
                    using HyperElasticPotentialType = pbat::fem::
                        HyperElasticPotential<MeshType, HyperElasticEnergyType, QuadratureOrder>;
                    mHyperElasticPotential =
                        new HyperElasticPotentialType(*mesh, detJe, GNe, Y, nu);
                    mDims            = HyperElasticPotentialType::kDims;
                    mOrder           = HyperElasticPotentialType::kOrder;
                    mQuadratureOrder = HyperElasticPotentialType::kQuadratureOrder;
                });
        },
        qOrder);
}

HyperElasticPotential::HyperElasticPotential(
    Mesh const& M,
    Eigen::Ref<MatrixX const> const& detJe,
    Eigen::Ref<MatrixX const> const& GNe,
    Eigen::Ref<MatrixX const> const& Y,
    Eigen::Ref<MatrixX const> const& nu,
    EHyperElasticEnergy ePsi,
    int qOrder)
    : eMeshElement(M.eElement),
      mMeshDims(M.mDims),
      mMeshOrder(M.mOrder),
      eHyperElasticEnergy(ePsi),
      mDims(),
      mOrder(),
      mQuadratureOrder(),
      mHyperElasticPotential(nullptr)
{
    M.ApplyWithQuadrature<kMaxQuadratureOrder>(
        [&]<pbat::fem::CMesh MeshType, auto QuadratureOrder>(MeshType* mesh) {
            pbat::common::ForTypes<
                pbat::physics::SaintVenantKirchhoffEnergy<MeshType::kDims>,
                pbat::physics::StableNeoHookeanEnergy<MeshType::kDims>>(
                [&]<class HyperElasticEnergyType>() {
                    using HyperElasticPotentialType = pbat::fem::
                        HyperElasticPotential<MeshType, HyperElasticEnergyType, QuadratureOrder>;
                    mHyperElasticPotential =
                        new HyperElasticPotentialType(*mesh, detJe, GNe, Y, nu);
                    mDims            = HyperElasticPotentialType::kDims;
                    mOrder           = HyperElasticPotentialType::kOrder;
                    mQuadratureOrder = HyperElasticPotentialType::kQuadratureOrder;
                });
        },
        qOrder);
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

MatrixX const& HyperElasticPotential::mue() const
{
    MatrixX* muePtr;
    Apply([&]<class HyperElasticPotentialType>(HyperElasticPotentialType* hyperElasticPotential) {
        muePtr = std::addressof(hyperElasticPotential->mue);
    });
    return *muePtr;
}

MatrixX& HyperElasticPotential::mue()
{
    MatrixX* muePtr;
    Apply([&]<class HyperElasticPotentialType>(HyperElasticPotentialType* hyperElasticPotential) {
        muePtr = std::addressof(hyperElasticPotential->mue);
    });
    return *muePtr;
}

MatrixX const& HyperElasticPotential::lambdae() const
{
    MatrixX* lambdaePtr;
    Apply([&]<class HyperElasticPotentialType>(HyperElasticPotentialType* hyperElasticPotential) {
        lambdaePtr = std::addressof(hyperElasticPotential->lambdae);
    });
    return *lambdaePtr;
}

MatrixX& HyperElasticPotential::lambdae()
{
    MatrixX* lambdaePtr;
    Apply([&]<class HyperElasticPotentialType>(HyperElasticPotentialType* hyperElasticPotential) {
        lambdaePtr = std::addressof(hyperElasticPotential->lambdae);
    });
    return *lambdaePtr;
}

MatrixX const& HyperElasticPotential::ElementHessians() const
{
    MatrixX* HePtr;
    Apply([&]<class HyperElasticPotentialType>(HyperElasticPotentialType* hyperElasticPotential) {
        HePtr = std::addressof(hyperElasticPotential->He);
    });
    return *HePtr;
}

MatrixX& HyperElasticPotential::ElementHessians()
{
    MatrixX* HePtr;
    Apply([&]<class HyperElasticPotentialType>(HyperElasticPotentialType* hyperElasticPotential) {
        HePtr = std::addressof(hyperElasticPotential->He);
    });
    return *HePtr;
}

MatrixX const& HyperElasticPotential::ElementGradients() const
{
    MatrixX* gePtr;
    Apply([&]<class HyperElasticPotentialType>(HyperElasticPotentialType* hyperElasticPotential) {
        gePtr = std::addressof(hyperElasticPotential->Ge);
    });
    return *gePtr;
}

MatrixX& HyperElasticPotential::ElementGradients()
{
    MatrixX* gePtr;
    Apply([&]<class HyperElasticPotentialType>(HyperElasticPotentialType* hyperElasticPotential) {
        gePtr = std::addressof(hyperElasticPotential->Ge);
    });
    return *gePtr;
}

VectorX const& HyperElasticPotential::ElementPotentials() const
{
    VectorX* UePtr;
    Apply([&]<class HyperElasticPotentialType>(HyperElasticPotentialType* hyperElasticPotential) {
        UePtr = std::addressof(hyperElasticPotential->Ue);
    });
    return *UePtr;
}

VectorX& HyperElasticPotential::ElementPotentials()
{
    VectorX* UePtr;
    Apply([&]<class HyperElasticPotentialType>(HyperElasticPotentialType* hyperElasticPotential) {
        UePtr = std::addressof(hyperElasticPotential->Ue);
    });
    return *UePtr;
}

HyperElasticPotential::~HyperElasticPotential()
{
    if (mHyperElasticPotential != nullptr)
        Apply(
            [&]<class HyperElasticPotentialType>(HyperElasticPotentialType* hyperElasticPotential) {
                delete hyperElasticPotential;
            });
}

template <class Func>
void HyperElasticPotential::Apply(Func&& f) const
{
    HepApplyToMeshWithQuadrature<kMaxQuadratureOrder>(
        mMeshDims,
        mMeshOrder,
        eMeshElement,
        mQuadratureOrder,
        [&]<pbat::fem::CMesh MeshType, auto QuadratureOrder>() {
            pbat::common::ForTypes<
                pbat::physics::SaintVenantKirchhoffEnergy<MeshType::kDims>,
                pbat::physics::StableNeoHookeanEnergy<MeshType::kDims>>(
                [&]<class HyperElasticEnergyType>() {
                    using HyperElasticPotentialType = pbat::fem::
                        HyperElasticPotential<MeshType, HyperElasticEnergyType, QuadratureOrder>;
                    HyperElasticPotentialType* hyperElasticPotential =
                        reinterpret_cast<HyperElasticPotentialType*>(mHyperElasticPotential);
                    f.template operator()<HyperElasticPotentialType>(hyperElasticPotential);
                });
        });
}

} // namespace fem
} // namespace py
} // namespace pbat