#ifndef PBAT_FEM_LOAD_VECTOR_H
#define PBAT_FEM_LOAD_VECTOR_H

#include "Concepts.h"
#include "ShapeFunctions.h"

#include <exception>
#include <fmt/core.h>
#include <pbat/Aliases.h>
#include <pbat/common/Eigen.h>
#include <pbat/profiling/Profiling.h>
#include <tbb/parallel_for.h>

namespace pbat {
namespace fem {

template <CMesh TMesh, int QuadratureOrder>
struct LoadVector
{
  public:
    using SelfType              = LoadVector<TMesh, QuadratureOrder>;
    using MeshType              = TMesh;
    using ElementType           = typename TMesh::ElementType;
    using QuadratureRuleType    = typename ElementType::template QuadratureType<QuadratureOrder>;
    static int constexpr kOrder = ElementType::kOrder;
    static int constexpr kQuadratureOrder = QuadratureOrder;

    /**
     * @brief
     * @tparam TDerived
     * @param mesh
     * @param detJe |#quad.pts.|x|#elements| affine element jacobian determinants at quadrature
     * points
     * @param fe |dims|x|#elements| piecewise element constant load, or |dims|x1 constant load
     */
    template <class TDerived>
    LoadVector(
        MeshType const& mesh,
        Eigen::Ref<MatrixX const> const& detJe,
        Eigen::DenseBase<TDerived> const& fe,
        int dims = 1);

    SelfType& operator=(SelfType const&) = delete;

    /**
     * @brief Transforms this matrix-free mass matrix representation into sparse compressed format.
     * @return
     */
    VectorX ToVector() const;

    /**
     * @brief
     * @tparam TDerived
     * @param fe |dims|x|#elements| piecewise element constant load, or |dims|x1 constant load
     */
    template <class TDerived>
    void SetLoad(Eigen::DenseBase<TDerived> const& fe);

    void CheckValidState() const;

    MeshType const& mesh; ///< The finite element mesh
    MatrixX fe;           ///< |dims|x|#elements| piecewise element constant load
    MatrixX N; ///< |ElementType::kNodes|x|#elements| integrated element shape functions. To
               ///< obtain the element force vectors, compute Neint \kron I_{dims} * f
    Eigen::Ref<MatrixX const> detJe; ///< |# element quadrature points|x|#elements| matrix of
                                     ///< jacobian determinants at element quadrature points
    int dims; ///< Dimensionality of image of FEM function space, i.e. this load vector is
              ///< actually f \kronecker 1_{dims \times dims}. Should be >= 1.
};

template <CMesh TMesh, int QuadratureOrder>
template <class TDerived>
inline LoadVector<TMesh, QuadratureOrder>::LoadVector(
    MeshType const& meshIn,
    Eigen::Ref<MatrixX const> const& detJe,
    Eigen::DenseBase<TDerived> const& load,
    int dims)
    : mesh(meshIn), fe(), N(), detJe(detJe), dims(dims)
{
    SetLoad(load);
    N = IntegratedShapeFunctions<kQuadratureOrder>(mesh, detJe);
}

template <CMesh TMesh, int QuadratureOrder>
inline VectorX LoadVector<TMesh, QuadratureOrder>::ToVector() const
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.fem.LoadVector.ToVector");
    CheckValidState();
    auto const n                = mesh.X.cols() * dims;
    auto const numberOfElements = mesh.E.cols();
    VectorX f                   = VectorX::Zero(n);
    for (auto e = 0; e < numberOfElements; ++e)
    {
        auto const nodes = mesh.E.col(e);
        for (auto i = 0; i < nodes.size(); ++i)
        {
            f.segment(dims * nodes(i), dims) += N(i, e) * fe.col(e);
        }
    }
    return f;
}

template <CMesh TMesh, int QuadratureOrder>
inline void LoadVector<TMesh, QuadratureOrder>::CheckValidState() const
{
    if (dims < 1)
    {
        std::string const what =
            fmt::format("Expected output dimensionality >= 1, got {} instead", dims);
        throw std::invalid_argument(what);
    }
}

template <CMesh TMesh, int QuadratureOrder>
template <class TDerived>
inline void LoadVector<TMesh, QuadratureOrder>::SetLoad(Eigen::DenseBase<TDerived> const& load)
{
    auto const numberOfElements = mesh.E.cols();
    if (load.rows() != dims)
    {
        std::string const what = fmt::format(
            "LoadVector<TMesh,{0}> discretizes a {0}-dimensional load, but received "
            "{1}-dimensional input load",
            dims,
            load.rows());
        throw std::invalid_argument(what);
    }
    if (load.cols() != 1 && load.cols() != numberOfElements)
    {
        std::string const what = fmt::format(
            "Input load vector must be constant or piecewise element constant, but size was {}",
            load.cols());
        throw std::invalid_argument(what);
    }

    fe.resize(dims, numberOfElements);
    if (load.cols() == 1)
        fe.colwise() = load.col(0);
    else // load.cols() == numberOfElements
        fe = load;
}

} // namespace fem
} // namespace pbat

#endif // PBAT_FEM_LOAD_VECTOR_H