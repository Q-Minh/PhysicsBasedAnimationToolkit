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
    Gradient(
        Mesh const& M,
        Eigen::Ref<IndexVectorX const> const& eg,
        Eigen::Ref<MatrixX const> const& GNeg);

    Gradient(Gradient const&)            = delete;
    Gradient& operator=(Gradient const&) = delete;

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

  private:
    void* mGradient;
};

void BindGradient(pybind11::module& m)
{
    namespace pyb = pybind11;

    pyb::class_<Gradient>(m, "Gradient")
        .def(
            pyb::init<
                Mesh const&,
                Eigen::Ref<IndexVectorX const> const&,
                Eigen::Ref<MatrixX const> const&>(),
            pyb::arg("mesh"),
            pyb::arg("eg").noconvert(),
            pyb::arg("GNeg").noconvert(),
            "Construct Gradient operator from mesh mesh, using precomputed shape function "
            "gradients GNeg at quadrature points at elements eg.")
        .def_readonly("dims", &Gradient::mDims)
        .def_readonly("order", &Gradient::mOrder, "Polynomial order of the gradient")
        .def_property_readonly("shape", &Gradient::Shape)
        .def("to_matrix", &Gradient::ToMatrix);
}

Gradient::Gradient(
    Mesh const& M,
    Eigen::Ref<IndexVectorX const> const& eg,
    Eigen::Ref<MatrixX const> const& GNeg)
    : eMeshElement(M.eElement),
      mMeshDims(M.mDims),
      mMeshOrder(M.mOrder),
      mDims(),
      mOrder(),
      mGradient(nullptr)
{
    M.Apply([&]<pbat::fem::CMesh MeshType>(MeshType* mesh) {
        using GradientType = pbat::fem::Gradient<MeshType>;
        mGradient          = new GradientType(*mesh, eg, GNeg);
        mDims              = GradientType::kDims;
        mOrder             = GradientType::kOrder;
    });
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
    ApplyToMesh(mMeshDims, mMeshOrder, eMeshElement, [&]<pbat::fem::CMesh MeshType>() {
        using GradientType     = pbat::fem::Gradient<MeshType>;
        GradientType* gradient = reinterpret_cast<GradientType*>(mGradient);
        f.template operator()<GradientType>(gradient);
    });
}

} // namespace fem
} // namespace py
} // namespace pbat