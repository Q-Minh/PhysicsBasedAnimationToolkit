#include "Gradient.h"

#include "Mesh.h"

#include <pbat/fem/Gradient.h>
#include <pybind11/eigen.h>
#include <tuple>

namespace pbat {
namespace py {
namespace fem {

class Gradient
{
  public:
    Gradient(Mesh const& M, Eigen::Ref<MatrixX const> const& GNe, int qOrder);

    template <class Func>
    void Apply(Func&& f) const;

    CSCMatrix ToMatrix() const;
    std::tuple<Index, Index> Shape() const;
    ~Gradient();

    EElement eMeshElement;
    int mMeshDims;
    int mMeshOrder;
    int mDims;
    int mOrder;
    int mQuadratureOrder;

    static auto constexpr kMaxQuadratureOrder = 2;

  private:
    void* mGradient;
};

void BindGradient(pybind11::module& m)
{
    namespace pyb = pybind11;

    pyb::class_<Gradient>(m, "Gradient")
        .def(
            pyb::init<Mesh const&, Eigen::Ref<MatrixX const> const&, int>(),
            pyb::arg("mesh"),
            pyb::arg("GNe"),
            pyb::arg("quadrature_order") = 1)
        .def_readonly("dims", &Gradient::mDims)
        .def_readonly("order", &Gradient::mOrder)
        .def_readonly("quadrature_order", &Gradient::mQuadratureOrder)
        .def_property_readonly("shape", &Gradient::Shape)
        .def("to_matrix", &Gradient::ToMatrix);
}

Gradient::Gradient(Mesh const& M, Eigen::Ref<MatrixX const> const& GNe, int qOrder)
    : eMeshElement(M.eElement),
      mMeshDims(M.mDims),
      mMeshOrder(M.mOrder),
      mDims(),
      mOrder(),
      mQuadratureOrder(),
      mGradient(nullptr)
{
    M.ApplyWithQuadrature<kMaxQuadratureOrder>(
        [&]<pbat::fem::CMesh MeshType, auto QuadratureOrder>(MeshType* mesh) {
            using GradientType = pbat::fem::Gradient<MeshType, QuadratureOrder>;
            mGradient          = new GradientType(*mesh, GNe);
            mDims              = GradientType::kDims;
            mOrder             = GradientType::kOrder;
            mQuadratureOrder   = GradientType::kQuadratureOrder;
        },
        qOrder);
}

CSCMatrix Gradient::ToMatrix() const
{
    CSCMatrix G;
    Apply([&]<class GradientType>(GradientType* gradient) { G = gradient->ToMatrix(); });
    return G;
}

std::tuple<Index, Index> Gradient::Shape() const
{
    Index rows{0}, cols{0};
    Apply([&]<class GradientType>(GradientType* gradient) {
        rows = gradient->OutputDimensions();
        cols = gradient->InputDimensions();
    });
    return std::make_tuple(rows, cols);
}

Gradient::~Gradient()
{
    if (mGradient != nullptr)
        Apply([&]<class GradientType>(GradientType* gradient) { delete gradient; });
}

template <class Func>
void Gradient::Apply(Func&& f) const
{
    ApplyToMeshWithQuadrature<kMaxQuadratureOrder>(
        mMeshDims,
        mMeshOrder,
        eMeshElement,
        mQuadratureOrder,
        [&]<pbat::fem::CMesh MeshType, auto QuadratureOrder>() {
            using GradientType     = pbat::fem::Gradient<MeshType, QuadratureOrder>;
            GradientType* gradient = reinterpret_cast<GradientType*>(mGradient);
            f.template operator()<GradientType>(gradient);
        });
}

} // namespace fem
} // namespace py
} // namespace pbat