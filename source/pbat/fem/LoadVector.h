#ifndef PBAT_FEM_LOAD_VECTOR_H
#define PBAT_FEM_LOAD_VECTOR_H

#include "Concepts.h"
#include "ShapeFunctions.h"
#include "pbat/aliases.h"
#include "pbat/common/Eigen.h"
#include "pbat/profiling/Profiling.h"

#include <exception>
#include <format>
#include <tbb/parallel_for.h>

namespace pbat {
namespace fem {

template <CMesh TMesh, int Dims, int QuadratureOrder>
struct LoadVector
{
  public:
    using SelfType                        = LoadVector<TMesh, Dims, QuadratureOrder>;
    using MeshType                        = TMesh;
    using ElementType                     = typename TMesh::ElementType;
    using QuadratureRuleType              = ElementType::template QuadratureType<QuadratureOrder>;
    static int constexpr kDims            = Dims;
    static int constexpr kOrder           = ElementType::kOrder;
    static int constexpr kQuadratureOrder = QuadratureOrder;

    /**
     * @brief
     * @tparam TDerived
     * @param mesh
     * @param detJe |#quad.pts.|x|#elements| affine element jacobian determinants at quadrature
     * points
     * @param fe |kDims|x|#elements| piecewise element constant load, or |kDims|x1 constant load
     */
    template <class TDerived>
    LoadVector(
        MeshType const& mesh,
        Eigen::Ref<MatrixX const> const& detJe,
        Eigen::DenseBase<TDerived> const& fe);

    SelfType& operator=(SelfType const&) = delete;

    /**
     * @brief Transforms this matrix-free mass matrix representation into sparse compressed format.
     * @return
     */
    VectorX ToVector() const;

    /**
     * @brief
     * @tparam TDerived
     * @param fe |kDims|x|#elements| piecewise element constant load, or |kDims|x1 constant load
     */
    template <class TDerived>
    void SetLoad(Eigen::DenseBase<TDerived> const& fe);

    MeshType const& mesh; ///< The finite element mesh
    MatrixX fe;           ///< |kDims|x|#elements| piecewise element constant load
    MatrixX N; ///< |ElementType::kNodes|x|#elements| integrated element shape functions. To
               ///< obtain the element force vectors, compute Neint \kron I_{kDims} * f
    Eigen::Ref<MatrixX const> detJe; ///< |# element quadrature points|x|#elements| matrix of
                                     ///< jacobian determinants at element quadrature points
};

template <CMesh TMesh, int Dims, int QuadratureOrder>
template <class TDerived>
inline LoadVector<TMesh, Dims, QuadratureOrder>::LoadVector(
    MeshType const& meshIn,
    Eigen::Ref<MatrixX const> const& detJe,
    Eigen::DenseBase<TDerived> const& load)
    : mesh(meshIn), fe(), N(), detJe(detJe)
{
    SetLoad(load);
    N = IntegratedShapeFunctions<kQuadratureOrder>(mesh, detJe);
}

template <CMesh TMesh, int Dims, int QuadratureOrder>
inline VectorX LoadVector<TMesh, Dims, QuadratureOrder>::ToVector() const
{
    PBA_PROFILE_SCOPE;
    auto const n                = mesh.X.cols() * kDims;
    auto const numberOfElements = mesh.E.cols();
    VectorX f                   = VectorX::Zero(n);
    for (auto e = 0; e < numberOfElements; ++e)
    {
        auto const nodes = mesh.E.col(e);
        for (auto i = 0; i < nodes.size(); ++i)
        {
            f.segment<kDims>(kDims * nodes(i)) += N(i, e) * fe.col(e);
        }
    }
    return f;
}

template <CMesh TMesh, int Dims, int QuadratureOrder>
template <class TDerived>
inline void
LoadVector<TMesh, Dims, QuadratureOrder>::SetLoad(Eigen::DenseBase<TDerived> const& load)
{
    auto const numberOfElements = mesh.E.cols();
    if (load.rows() != kDims)
    {
        std::string const what = std::format(
            "LoadVector<TMesh,{0}> discretizes a {0}-dimensional load, but received "
            "{1}-dimensional input load",
            kDims,
            load.rows());
        throw std::invalid_argument(what);
    }
    if (load.cols() != 1 && load.cols() != numberOfElements)
    {
        std::string const what = std::format(
            "Input load vector must be constant or piecewise element constant, but size was {}",
            load.cols());
        throw std::invalid_argument(what);
    }

    fe.resize(kDims, numberOfElements);
    if (load.cols() == 1)
        fe.colwise() = load.col(0);
    else // load.cols() == numberOfElements
        fe = load;
}

} // namespace fem
} // namespace pbat

#endif // PBAT_FEM_LOAD_VECTOR_H