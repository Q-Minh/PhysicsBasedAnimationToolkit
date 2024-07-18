#include "Laplacian.h"

#include "Mesh.h"

#include <pbat/fem/LaplacianMatrix.h>
#include <pybind11/eigen.h>
#include <tuple>

namespace pbat {
namespace py {
namespace fem {

class Laplacian
{
  public:
    Laplacian(
        Mesh const& M,
        Eigen::Ref<MatrixX const> const& detJe,
        Eigen::Ref<MatrixX const> const& GNe,
        int dims,
        int qOrder);

    Laplacian(Laplacian const&)            = delete;
    Laplacian& operator=(Laplacian const&) = delete;

    template <class Func>
    void Apply(Func&& f) const;

    CSCMatrix ToMatrix() const;
    std::tuple<Index, Index> Shape() const;

    MatrixX const& ElementLaplacians() const;
    MatrixX& ElementLaplacians();
    int const& dims() const;
    int& dims();

    ~Laplacian();

    EElement eMeshElement;
    int mMeshDims;
    int mMeshOrder;
    int mOrder;
    int mQuadratureOrder;

    static auto constexpr kMaxQuadratureOrder = 4;

  private:
    void* mLaplacian;
};

void BindLaplacian(pybind11::module& m)
{
    namespace pyb = pybind11;

    pyb::class_<Laplacian>(m, "Laplacian")
        .def(
            pyb::init<
                Mesh const&,
                Eigen::Ref<MatrixX const> const&,
                Eigen::Ref<MatrixX const> const&,
                int,
                int>(),
            pyb::arg("mesh"),
            pyb::arg("detJe"),
            pyb::arg("GNe"),
            pyb::arg("dims")             = 1,
            pyb::arg("quadrature_order") = 1)
        .def_property(
            "dims",
            [](Laplacian const& L) { return L.dims(); },
            [](Laplacian& L, int dims) { L.dims() = dims; })
        .def_readonly("order", &Laplacian::mOrder)
        .def_readonly("quadrature_order", &Laplacian::mQuadratureOrder)
        .def_property(
            "deltaE",
            [](Laplacian const& L) { return L.ElementLaplacians(); },
            [](Laplacian& L, Eigen::Ref<MatrixX const> const& deltaE) {
                L.ElementLaplacians() = deltaE;
            })
        .def_property_readonly("shape", &Laplacian::Shape)
        .def("to_matrix", &Laplacian::ToMatrix);
}

Laplacian::Laplacian(
    Mesh const& M,
    Eigen::Ref<MatrixX const> const& detJe,
    Eigen::Ref<MatrixX const> const& GNe,
    int dims,
    int qOrder)
    : eMeshElement(M.eElement),
      mMeshDims(M.mDims),
      mMeshOrder(M.mOrder),
      mOrder(),
      mQuadratureOrder(),
      mLaplacian(nullptr)
{
    M.ApplyWithQuadrature<kMaxQuadratureOrder>(
        [&]<pbat::fem::CMesh MeshType, auto QuadratureOrder>(MeshType* mesh) {
            using LaplacianType = pbat::fem::SymmetricLaplacianMatrix<MeshType, QuadratureOrder>;
            mLaplacian          = new LaplacianType(*mesh, detJe, GNe, dims);
            mOrder              = LaplacianType::kOrder;
            mQuadratureOrder    = LaplacianType::kQuadratureOrder;
        },
        qOrder);
}

CSCMatrix Laplacian::ToMatrix() const
{
    CSCMatrix L;
    Apply([&]<class LaplacianType>(LaplacianType* laplacian) { L = laplacian->ToMatrix(); });
    return L;
}

std::tuple<Index, Index> Laplacian::Shape() const
{
    Index rows{0}, cols{0};
    Apply([&]<class LaplacianType>(LaplacianType* laplacian) {
        rows = laplacian->OutputDimensions();
        cols = laplacian->InputDimensions();
    });
    return std::make_tuple(rows, cols);
}

MatrixX const& Laplacian::ElementLaplacians() const
{
    MatrixX* deltaEPtr;
    Apply([&]<class LaplacianType>(LaplacianType* laplacian) {
        deltaEPtr = std::addressof(laplacian->deltaE);
    });
    return *deltaEPtr;
}

MatrixX& Laplacian::ElementLaplacians()
{
    MatrixX* deltaEPtr;
    Apply([&]<class LaplacianType>(LaplacianType* laplacian) {
        deltaEPtr = std::addressof(laplacian->deltaE);
    });
    return *deltaEPtr;
}

int const& Laplacian::dims() const
{
    int* dimsPtr;
    Apply([&]<class LaplacianType>(LaplacianType* laplacian) {
        dimsPtr = std::addressof(laplacian->dims);
    });
    return *dimsPtr;
}

int& Laplacian::dims()
{
    int* dimsPtr;
    Apply([&]<class LaplacianType>(LaplacianType* laplacian) {
        dimsPtr = std::addressof(laplacian->dims);
    });
    return *dimsPtr;
}

Laplacian::~Laplacian()
{
    if (mLaplacian != nullptr)
        Apply([&]<class LaplacianType>(LaplacianType* laplacian) { delete laplacian; });
}

template <class Func>
void Laplacian::Apply(Func&& f) const
{
    ApplyToMeshWithQuadrature<kMaxQuadratureOrder>(
        mMeshDims,
        mMeshOrder,
        eMeshElement,
        mQuadratureOrder,
        [&]<pbat::fem::CMesh MeshType, auto QuadratureOrder>() {
            using LaplacianType = pbat::fem::SymmetricLaplacianMatrix<MeshType, QuadratureOrder>;
            LaplacianType* laplacian = reinterpret_cast<LaplacianType*>(mLaplacian);
            f.template operator()<LaplacianType>(laplacian);
        });
}

} // namespace fem
} // namespace py
} // namespace pbat