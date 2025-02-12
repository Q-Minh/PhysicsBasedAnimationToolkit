/**
 * @file LoadVector.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief LoadVector API and implementation.
 * @date 2025-02-11
 *
 * @copyright Copyright (c) 2025
 *
 */

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

/**
 * @brief A matrix-free representation of a finite element load vector \f$ \mathbf{f}_i =
 * \int_\Omega \mathbf{f}(X) \phi_i(X) \f$ under Galerkin projection.
 *
 * \note Assumes \f$ \mathbf{f}(X) \f$ is piecewise constant over the elements.
 *
 * \todo Link to my higher-level FEM crash course doc.
 *
 * @tparam TMesh Type satisfying concept CMesh
 * @tparam QuadratureOrder Quadrature order for integrating the load vector
 */
template <CMesh TMesh, int QuadratureOrder>
struct LoadVector
{
  public:
    using SelfType    = LoadVector<TMesh, QuadratureOrder>; ///< Self type
    using MeshType    = TMesh;                              ///< Mesh type
    using ElementType = typename TMesh::ElementType;        ///< Element type
    using QuadratureRuleType =
        typename ElementType::template QuadratureType<QuadratureOrder>; ///< Quadrature rule type
    static int constexpr kOrder = ElementType::kOrder; ///< Polynomial order of the load vector
    static int constexpr kQuadratureOrder = QuadratureOrder; ///< Quadrature order

    /**
     * @brief Construct a new LoadVector object
     * @tparam TDerived Eigen dense expression type
     * @param mesh Finite element mesh
     * @param detJe `|# quad.pts.|x|# elements|` affine element jacobian determinants at quadrature
     * points
     * @param fe `|dims|x|# elements|` piecewise element constant load, or `|dims|x1` constant load
     * @param dims Dimensionality of image of FEM function space
     * @pre `dims >= 1` and `fe.cols() == 1 || fe.cols() == mesh.E.cols()` and `fe.rows() == dims`
     */
    template <class TDerived>
    LoadVector(
        MeshType const& mesh,
        Eigen::Ref<MatrixX const> const& detJe,
        Eigen::DenseBase<TDerived> const& fe,
        int dims = 1);

    SelfType& operator=(SelfType const&) = delete;

    /**
     * @brief Transforms this element-wise load vector representation into a global load vector.
     * @return VectorX Global load vector
     */
    VectorX ToVector() const;

    /**
     * @brief Set the external loading
     * @tparam TDerived Eigen dense expression type
     * @param fe `|dims|x|# elements|` piecewise element constant load, or `|dims|x1` constant load
     */
    template <class TDerived>
    void SetLoad(Eigen::DenseBase<TDerived> const& fe);

    /**
     * @brief Check if the state of this load vector is valid
     */
    void CheckValidState() const;

    MeshType const& mesh; ///< The finite element mesh
    MatrixX fe;           ///< `|dims|x|# elements|` piecewise element constant load
    MatrixX N; ///< `|ElementType::kNodes|x|# elements|` integrated element shape functions. See
               ///< (IntegratedShapeFunctions()).
    Eigen::Ref<MatrixX const> detJe; ///< `|# element quadrature points|x|# elements|` matrix of
                                     ///< jacobian determinants at element quadrature points
    int dims; ///< Dimensionality of external loading. Should have `dims >= 1`.
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