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
        Eigen::Ref<IndexVectorX const> const& eg,
        Eigen::Ref<MatrixX const> const& wg,
        Eigen::Ref<MatrixX const> const& GNeg,
        int dims);

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
                Eigen::Ref<IndexVectorX const> const&,
                Eigen::Ref<MatrixX const> const&,
                Eigen::Ref<MatrixX const> const&,
                int>(),
            pyb::arg("mesh"),
            pyb::arg("eg"),
            pyb::arg("wg"),
            pyb::arg("GNe"),
            pyb::arg("dims") = 1,
            "Construct the symmetric part of the Laplacian operator on mesh mesh, using "
            "precomputed shape function gradients GNeg evaluated at quadrature points g at "
            "elements eg with weights wg. The discretization is based on Galerkin projection. The "
            "dimensions dims can be set to accommodate vector-valued functions.")
        .def_property(
            "dims",
            [](Laplacian const& L) { return L.dims(); },
            [](Laplacian& L, int dims) { L.dims() = dims; })
        .def_readonly("order", &Laplacian::mOrder)
        .def_property(
            "deltag",
            [](Laplacian const& L) { return L.ElementLaplacians(); },
            [](Laplacian& L, Eigen::Ref<MatrixX const> const& deltag) {
                L.ElementLaplacians() = deltag;
            },
            "|#element nodes|x|#element nodes * #quad.pts.| matrix of element Laplacians")
        .def_property_readonly("shape", &Laplacian::Shape)
        .def("to_matrix", &Laplacian::ToMatrix);
}

Laplacian::Laplacian(
    Mesh const& M,
    Eigen::Ref<IndexVectorX const> const& eg,
    Eigen::Ref<MatrixX const> const& wg,
    Eigen::Ref<MatrixX const> const& GNe,
    int dims)
    : eMeshElement(M.eElement),
      mMeshDims(M.mDims),
      mMeshOrder(M.mOrder),
      mOrder(),
      mLaplacian(nullptr)
{
    M.Apply([&]<pbat::fem::CMesh MeshType>(MeshType* mesh) {
        using LaplacianType = pbat::fem::SymmetricLaplacianMatrix<MeshType>;
        mLaplacian          = new LaplacianType(*mesh, eg, wg, GNe, dims);
        mOrder              = LaplacianType::kOrder;
    });
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
    MatrixX* deltagPtr;
    Apply([&]<class LaplacianType>(LaplacianType* laplacian) {
        deltagPtr = std::addressof(laplacian->deltag);
    });
    return *deltagPtr;
}

MatrixX& Laplacian::ElementLaplacians()
{
    MatrixX* deltagPtr;
    Apply([&]<class LaplacianType>(LaplacianType* laplacian) {
        deltagPtr = std::addressof(laplacian->deltag);
    });
    return *deltagPtr;
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
    ApplyToMesh(mMeshDims, mMeshOrder, eMeshElement, [&]<pbat::fem::CMesh MeshType>() {
        using LaplacianType      = pbat::fem::SymmetricLaplacianMatrix<MeshType>;
        LaplacianType* laplacian = reinterpret_cast<LaplacianType*>(mLaplacian);
        f.template operator()<LaplacianType>(laplacian);
    });
}

} // namespace fem
} // namespace py
} // namespace pbat