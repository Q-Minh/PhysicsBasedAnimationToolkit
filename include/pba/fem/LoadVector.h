#ifndef PBA_CORE_FEM_LOAD_VECTOR_H
#define PBA_CORE_FEM_LOAD_VECTOR_H

#include "Concepts.h"
#include "Jacobian.h"
#include "pba/aliases.h"
#include "pba/common/Eigen.h"
#include "pba/common/Profiling.h"

#include <exception>
#include <format>
#include <tbb/parallel_for.h>

namespace pba {
namespace fem {

template <CMesh TMesh, int Dims>
struct LoadVector
{
  public:
    using SelfType              = LoadVector<TMesh, Dims>;
    using MeshType              = TMesh;
    using ElementType           = typename TMesh::ElementType;
    using QuadratureRuleType    = ElementType::template QuadratureType<ElementType::kOrder>;
    static int constexpr kDims  = Dims;
    static int constexpr kOrder = ElementType::kOrder;

    template <class TDerived>
    LoadVector(MeshType const& mesh, Eigen::DenseBase<TDerived> const& f);

    SelfType& operator=(SelfType const&) = delete;

    /**
     * @brief Transforms this matrix-free mass matrix representation into sparse compressed format.
     * @return
     */
    VectorX ToVector() const;

    /**
     * @brief Computes the piecewise load representations
     */
    void IntegrateShapeFunctions();

    MeshType const& mesh; ///< The finite element mesh
    MatrixX fe;           ///< kDims x |#elements| piecewise constant load
    MatrixX N;     ///< |ElementType::kNodes|x|#elements| integrated element shape functions. To
                   ///< obtain the element force vectors, compute Neint \kron I_{kDims} * f
    MatrixX detJe; ///< |# element quadrature points|x|#elements| matrix of jacobian determinants at
                   ///< element quadrature points
};

template <CMesh TMesh, int Dims>
template <class TDerived>
inline LoadVector<TMesh, Dims>::LoadVector(
    MeshType const& meshIn,
    Eigen::DenseBase<TDerived> const& load)
    : mesh(meshIn), fe(), N(), detJe()
{
    PBA_PROFILE_NAMED_SCOPE("Construct fem::LoadVector");
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
        fe.colwise() = load;
    else // load.cols() == numberOfElements
        fe = load;

    detJe = DeterminantOfJacobian<QuadratureRuleType::kOrder>(mesh);
    IntegrateShapeFunctions();
}

template <CMesh TMesh, int Dims>
inline VectorX LoadVector<TMesh, Dims>::ToVector() const
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

template <CMesh TMesh, int Dims>
inline void LoadVector<TMesh, Dims>::IntegrateShapeFunctions()
{
    PBA_PROFILE_SCOPE;
    auto const numberOfElements = mesh.E.cols();
    N.setZero(ElementType::kNodes, numberOfElements);
    auto const Xg = common::ToEigen(QuadratureRuleType::points)
                        .reshaped(QuadratureRuleType::kDims + 1, QuadratureRuleType::kPoints)
                        .bottomRows(QuadratureRuleType::kDims);
    auto const wg = common::ToEigen(QuadratureRuleType::weights);
    tbb::parallel_for(Index{0}, Index{numberOfElements}, [&](Index e) {
        auto const nodes    = mesh.E.col(e);
        auto const vertices = nodes(ElementType::Vertices);
        for (auto g = 0; g < QuadratureRuleType::kPoints; ++g)
        {
            Vector<ElementType::kNodes> const Ng = ElementType::N(Xg.col(g));
            N.col(e) += (wg(g) * detJe(g, e)) * Ng;
        }
    });
}

} // namespace fem
} // namespace pba

#endif // PBA_CORE_FEM_LOAD_VECTOR_H