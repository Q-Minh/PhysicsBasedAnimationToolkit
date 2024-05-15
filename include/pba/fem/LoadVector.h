#ifndef PBA_CORE_FEM_LOAD_VECTOR_H
#define PBA_CORE_FEM_LOAD_VECTOR_H

#include "Concepts.h"
#include "Jacobian.h"
#include "pba/aliases.h"
#include "pba/common/Eigen.h"

#include <exception>
#include <format>
#include <functional>
#include <tbb/parallel_for.h>

namespace pba {
namespace fem {

template <CMesh TMesh, int Dims>
struct LoadVector
{
  public:
    using MeshType              = TMesh;
    using ElementType           = typename TMesh::ElementType;
    using QuadratureRuleType    = ElementType::template QuadratureType<ElementType::kOrder>;
    static int constexpr kDims  = Dims;
    static int constexpr kOrder = ElementType::kOrder;

    template <class TDerived>
    LoadVector(MeshType const& mesh, Eigen::DenseBase<TDerived> const& f);

    /**
     * @brief Transforms this matrix-free mass matrix representation into sparse compressed format.
     * @return
     */
    VectorX ToVector() const;

    /**
     * @brief Computes the piecewise load representations
     */
    void IntegrateShapeFunctions();

    std::reference_wrapper<MeshType const> mesh; ///< The finite element mesh
    MatrixX fe;                                  ///< kDims x |#elements| piecewise constant load
    MatrixX N; ///< |ElementType::kNodes|x|#elements| integrated element shape functions. To
               ///< obtain the element force vectors, compute Neint \kron I_{kDims} * f
};

template <CMesh TMesh, int Dims>
template <class TDerived>
inline LoadVector<TMesh, Dims>::LoadVector(
    MeshType const& meshIn,
    Eigen::DenseBase<TDerived> const& load)
    : mesh(meshIn), fe(), N()
{
    MeshType const& M           = mesh.get();
    auto const numberOfElements = M.E.cols();
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

    IntegrateShapeFunctions();
}

template <CMesh TMesh, int Dims>
inline VectorX LoadVector<TMesh, Dims>::ToVector() const
{
    MeshType const& M           = mesh.get();
    auto const n                = M.X.cols() * kDims;
    auto const numberOfElements = M.E.cols();
    VectorX f                   = VectorX::Zero(n);
    for (auto e = 0; e < numberOfElements; ++e)
    {
        auto const nodes = M.E.col(e);
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
    using AffineElementType = typename ElementType::AffineBaseType;

    MeshType const& M           = mesh.get();
    auto const numberOfElements = M.E.cols();

    N.setZero(ElementType::kNodes, numberOfElements);
    auto const Xg = common::ToEigen(QuadratureRuleType::points)
                        .reshaped(QuadratureRuleType::kDims + 1, QuadratureRuleType::kPoints)
                        .bottomRows(QuadratureRuleType::kDims);
    tbb::parallel_for(Index{0}, Index{numberOfElements}, [&](Index e) {
        auto const nodes                = M.E.col(e);
        auto const vertices             = nodes(ElementType::Vertices);
        auto constexpr kRowsJ           = MeshType::kDims;
        auto constexpr kColsJ           = AffineElementType::kNodes;
        Matrix<kRowsJ, kColsJ> const Ve = M.X(Eigen::all, vertices);
        Scalar detJ{};
        if constexpr (AffineElementType::bHasConstantJacobian)
            detJ = DeterminantOfJacobian(Jacobian<AffineElementType>({}, Ve));

        auto const wg = common::ToEigen(QuadratureRuleType::weights);
        for (auto g = 0; g < QuadratureRuleType::kPoints; ++g)
        {
            if constexpr (!AffineElementType::bHasConstantJacobian)
                detJ = DeterminantOfJacobian(Jacobian<AffineElementType>(Xg.col(g), Ve));

            Vector<ElementType::kNodes> const Ng = ElementType::N(Xg.col(g));
            N.col(e) += (wg(g) * detJ) * Ng;
        }
    });
}

} // namespace fem
} // namespace pba

#endif // PBA_CORE_FEM_LOAD_VECTOR_H