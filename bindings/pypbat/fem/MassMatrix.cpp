#include "MassMatrix.h"

#include "Mesh.h"

#include <pbat/fem/MassMatrix.h>
#include <pybind11/eigen.h>
#include <tuple>

namespace pbat {
namespace py {
namespace fem {

class MassMatrix
{
  public:
    MassMatrix(
        Mesh const& M,
        Eigen::Ref<MatrixX const> const& detJe,
        Scalar rho,
        int dims,
        int qOrder);

    MassMatrix(
        Mesh const& M,
        Eigen::Ref<MatrixX const> const& detJe,
        Eigen::Ref<MatrixX const> const& rho,
        int dims,
        int qOrder);

    MassMatrix(MassMatrix const&)            = delete;
    MassMatrix& operator=(MassMatrix const&) = delete;
    MassMatrix(MassMatrix&&)                 = delete;
    MassMatrix& operator=(MassMatrix&&)      = delete;

    template <class Func>
    void Apply(Func&& f) const;

    CSCMatrix ToMatrix() const;
    std::tuple<Index, Index> Shape() const;

    MatrixX const& ElementMassMatrices() const;
    MatrixX& ElementMassMatrices();
    int const& dims() const;
    int& dims();

    ~MassMatrix();

    EElement eMeshElement;
    int mMeshDims;
    int mMeshOrder;
    int mOrder;
    int mQuadratureOrder;

    static auto constexpr kMaxQuadratureOrder = 6;

  private:
    void* mMassMatrix;
};

void BindMassMatrix(pybind11::module& m)
{
    namespace pyb = pybind11;

    pyb::class_<MassMatrix>(m, "MassMatrix")
        .def(
            pyb::init<Mesh const&, Eigen::Ref<MatrixX const> const&, Scalar, int, int>(),
            pyb::arg("mesh"),
            pyb::arg("detJe").noconvert(),
            pyb::arg("rho")              = 1.,
            pyb::arg("dims")             = 1,
            pyb::arg("quadrature_order") = 1,
            "Construct the mass matrix operator on mesh mesh, using "
            "precomputed jacobian determinants detJe evaluated at "
            "quadrature points given by the quadrature rule of order quadrature_order. The "
            "dimensions dims can be set to accommodate vector-valued functions. rho is a uniform "
            "mass density.")
        .def(
            pyb::init<
                Mesh const&,
                Eigen::Ref<MatrixX const> const&,
                Eigen::Ref<MatrixX const> const&,
                int,
                int>(),
            pyb::arg("mesh"),
            pyb::arg("detJe").noconvert(),
            pyb::arg("rho")              = 1.,
            pyb::arg("dims")             = 1,
            pyb::arg("quadrature_order") = 1,
            "Construct the mass matrix operator on mesh mesh, using "
            "precomputed jacobian determinants detJe evaluated at "
            "quadrature points given by the quadrature rule of order quadrature_order. The "
            "dimensions dims can be set to accommodate vector-valued functions. rho is a piecewise "
            "constant (per element) mass density.")
        .def_property(
            "dims",
            [](MassMatrix const& L) { return L.dims(); },
            [](MassMatrix& L, int dims) { L.dims() = dims; })
        .def_readonly("order", &MassMatrix::mOrder)
        .def_readonly("quadrature_order", &MassMatrix::mQuadratureOrder)
        .def_property(
            "ME",
            [](MassMatrix const& M) { return M.ElementMassMatrices(); },
            [](MassMatrix& M, Eigen::Ref<MatrixX const> const& ME) {
                M.ElementMassMatrices() = ME;
            },
            "|#element nodes| x |#elements nodes * #elements| matrix of element mass matrices")
        .def_property_readonly("shape", &MassMatrix::Shape)
        .def("to_matrix", &MassMatrix::ToMatrix);
}

MassMatrix::MassMatrix(
    Mesh const& M,
    Eigen::Ref<MatrixX const> const& detJe,
    Scalar rho,
    int dims,
    int qOrder)
    : eMeshElement(M.eElement),
      mMeshDims(M.mDims),
      mMeshOrder(M.mOrder),
      mOrder(),
      mQuadratureOrder(),
      mMassMatrix(nullptr)
{
    M.ApplyWithQuadrature<kMaxQuadratureOrder>(
        [&]<pbat::fem::CMesh MeshType, auto QuadratureOrder>(MeshType* mesh) {
            using MassMatrixType = pbat::fem::MassMatrix<MeshType, QuadratureOrder>;
            mMassMatrix          = new MassMatrixType(*mesh, detJe, rho, dims);
            mOrder               = MassMatrixType::kOrder;
            mQuadratureOrder     = MassMatrixType::kQuadratureOrder;
        },
        qOrder);
}

MassMatrix::MassMatrix(
    Mesh const& M,
    Eigen::Ref<MatrixX const> const& detJe,
    Eigen::Ref<MatrixX const> const& rho,
    int dims,
    int qOrder)
    : eMeshElement(M.eElement),
      mMeshDims(M.mDims),
      mMeshOrder(M.mOrder),
      mOrder(),
      mQuadratureOrder(),
      mMassMatrix(nullptr)
{
    M.ApplyWithQuadrature<kMaxQuadratureOrder>(
        [&]<pbat::fem::CMesh MeshType, auto QuadratureOrder>(MeshType* mesh) {
            using MassMatrixType = pbat::fem::MassMatrix<MeshType, QuadratureOrder>;
            mMassMatrix          = new MassMatrixType(*mesh, detJe, rho, dims);
            mOrder               = MassMatrixType::kOrder;
            mQuadratureOrder     = MassMatrixType::kQuadratureOrder;
        },
        qOrder);
}

CSCMatrix MassMatrix::ToMatrix() const
{
    CSCMatrix M;
    Apply([&]<class MassMatrixType>(MassMatrixType* massMatrix) { M = massMatrix->ToMatrix(); });
    return M;
}

std::tuple<Index, Index> MassMatrix::Shape() const
{
    Index rows{0}, cols{0};
    Apply([&]<class MassMatrixType>(MassMatrixType* massMatrix) {
        rows = massMatrix->OutputDimensions();
        cols = massMatrix->InputDimensions();
    });
    return std::make_tuple(rows, cols);
}

MatrixX const& MassMatrix::ElementMassMatrices() const
{
    MatrixX* MePtr;
    Apply([&]<class MassMatrixType>(MassMatrixType* massMatrix) {
        MePtr = std::addressof(massMatrix->Me);
    });
    return *MePtr;
}

MatrixX& MassMatrix::ElementMassMatrices()
{
    MatrixX* MePtr;
    Apply([&]<class MassMatrixType>(MassMatrixType* massMatrix) {
        MePtr = std::addressof(massMatrix->Me);
    });
    return *MePtr;
}

int const& MassMatrix::dims() const
{
    int* dimsPtr;
    Apply([&]<class MassMatrixType>(MassMatrixType* massMatrix) {
        dimsPtr = std::addressof(massMatrix->dims);
    });
    return *dimsPtr;
}

int& MassMatrix::dims()
{
    int* dimsPtr;
    Apply([&]<class MassMatrixType>(MassMatrixType* massMatrix) {
        dimsPtr = std::addressof(massMatrix->dims);
    });
    return *dimsPtr;
}

MassMatrix::~MassMatrix()
{
    if (mMassMatrix != nullptr)
        Apply([&]<class MassMatrixType>(MassMatrixType* massMatrix) { delete massMatrix; });
}

template <class Func>
void MassMatrix::Apply(Func&& f) const
{
    ApplyToMeshWithQuadrature<kMaxQuadratureOrder>(
        mMeshDims,
        mMeshOrder,
        eMeshElement,
        mQuadratureOrder,
        [&]<pbat::fem::CMesh MeshType, auto QuadratureOrder>() {
            using MassMatrixType       = pbat::fem::MassMatrix<MeshType, QuadratureOrder>;
            MassMatrixType* massMatrix = reinterpret_cast<MassMatrixType*>(mMassMatrix);
            f.template operator()<MassMatrixType>(massMatrix);
        });
}

} // namespace fem
} // namespace py
} // namespace pbat