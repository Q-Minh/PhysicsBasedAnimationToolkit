/**
 * @file HyperElasticPotential.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Hyper elastic potential energy
 * @date 2025-02-11
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBA_FEM_HYPER_ELASTIC_POTENTIAL_H
#define PBA_FEM_HYPER_ELASTIC_POTENTIAL_H

#include "Concepts.h"
#include "DeformationGradient.h"
#include "pbat/Aliases.h"
#include "pbat/common/Eigen.h"
#include "pbat/math/linalg/SparsityPattern.h"
#include "pbat/math/linalg/mini/Eigen.h"
#include "pbat/math/linalg/mini/Product.h"
#include "pbat/physics/HyperElasticity.h"
#include "pbat/profiling/Profiling.h"

#include <Eigen/SVD>
#include <exception>
#include <fmt/core.h>
#include <span>
#include <string>
#include <tbb/parallel_for.h>

namespace pbat {
namespace fem {

/**
 * @brief Total hyper elastic potential \f$ U(\mathbf{x}) = \int_\Omega \Psi(\mathbf{F}) d\Omega \f$
 *
 * where \f$ \mathbf{F} \f$ is the deformation gradient and \f$ \Psi \f$ is the hyper elastic energy
 * density.
 *
 * HyperElasticPotential's depends on a \f$ |Q| \f$-point quadrature specified by element
 * indices \f$ e_g \f$, quadrature weights \f$ w_g \f$, and shape function gradients \f$ \nabla
 * \phi_g \f$ at quadrature points. This allows users to try out different quadrature rules
 * seamlessly.
 *
 * @tparam TMesh Type satisfying concept CMesh
 * @tparam THyperElasticEnergy Type satisfying concept CHyperElasticEnergy
 */
template <CMesh TMesh, physics::CHyperElasticEnergy THyperElasticEnergy>
struct HyperElasticPotential
{
  public:
    using SelfType          = HyperElasticPotential<TMesh, THyperElasticEnergy>; ///< Self type
    using MeshType          = TMesh;                                             ///< Mesh type
    using ElementType       = typename TMesh::ElementType; ///< FEM element type
    using ElasticEnergyType = THyperElasticEnergy;         ///< Hyper elastic energy density type
    static_assert(
        MeshType::kDims == ElasticEnergyType::kDims,
        "Embedding dimensions of mesh must match dimensionality of hyper elastic energy.");

    static auto constexpr kDims = THyperElasticEnergy::kDims; ///< Number of spatial dimensions

    SelfType& operator=(SelfType const&) = delete;

    /**
     * @brief Construct a new Hyper Elastic Potential object
     *
     * @param mesh FEM mesh
     * @param eg \f$ |Q| \f$ array of element indices at quadrature points
     * @param wg \f$ |Q| \f$ array of quadrature weights
     * @param GNeg Shape function gradients at quadrature points. See ShapeFunctionGradients().
     * @param Y Young's modulus
     * @param nu Poisson's ratio
     */
    HyperElasticPotential(
        MeshType const& mesh,
        Eigen::Ref<IndexVectorX const> const& eg,
        Eigen::Ref<VectorX const> const& wg,
        Eigen::Ref<MatrixX const> const& GNeg,
        Scalar Y,
        Scalar nu);
    /**
     * @brief Construct a new Hyper Elastic Potential object
     *
     * @tparam TDerivedY Eigen dense expression type
     * @tparam TDerivednu Eigen dense expression type
     * @param mesh FEM mesh
     * @param eg \f$ |Q| \f$ array of element indices at quadrature points
     * @param wg \f$ |Q| \f$ array of quadrature weights
     * @param GNeg Shape function gradients at quadrature points. See ShapeFunctionGradients().
     * @param Y \f$ |Q| \f$ Young's moduli
     * @param nu \f$ |Q| \f$ Poisson's ratios
     */
    template <class TDerivedY, class TDerivednu>
    HyperElasticPotential(
        MeshType const& mesh,
        Eigen::Ref<IndexVectorX const> const& eg,
        Eigen::Ref<VectorX const> const& wg,
        Eigen::Ref<MatrixX const> const& GNeg,
        Eigen::DenseBase<TDerivedY> const& Y,
        Eigen::DenseBase<TDerivednu> const& nu);
    /**
     * @brief Construct a new Hyper Elastic Potential object
     *
     * Eagerly computes element elasticity (and its derivatives)
     *
     * @tparam TDerived Eigen matrix expression type
     * @param mesh FEM mesh
     * @param eg \f$ |Q| \f$ array of element indices at quadrature points
     * @param wg \f$ |Q| \f$ array of quadrature weights
     * @param GNeg Shape function gradients at quadrature points. See ShapeFunctionGradients().
     * @param x \f$ d \times n \f$ matrix of deformed nodal positions
     * @param Y Young's modulus
     * @param nu Poisson's ratio
     */
    template <class TDerived>
    HyperElasticPotential(
        MeshType const& mesh,
        Eigen::Ref<IndexVectorX const> const& eg,
        Eigen::Ref<VectorX const> const& wg,
        Eigen::Ref<MatrixX const> const& GNeg,
        Eigen::MatrixBase<TDerived> const& x,
        Scalar Y,
        Scalar nu);
    /**
     * @brief Construct a new Hyper Elastic Potential object
     *
     * Eagerly computes element elasticity (and its derivatives)
     *
     * @tparam TDerivedx Eigen matrix expression type
     * @tparam TDerivedY Eigen dense expression type
     * @tparam TDerivednu Eigen dense expression type
     * @param mesh FEM mesh
     * @param eg \f$ |Q| \f$ array of element indices at quadrature points
     * @param wg \f$ |Q| \f$ array of quadrature weights
     * @param GNeg Shape function gradients at quadrature points. See ShapeFunctionGradients().
     * @param x \f$ d \times n \f$ matrix of deformed nodal positions
     * @param Y \f$ |Q| \f$ Young's moduli
     * @param nu \f$ |Q| \f$ Poisson's ratios
     */
    template <class TDerivedx, class TDerivedY, class TDerivednu>
    HyperElasticPotential(
        MeshType const& mesh,
        Eigen::Ref<IndexVectorX const> const& eg,
        Eigen::Ref<VectorX const> const& wg,
        Eigen::Ref<MatrixX const> const& GNeg,
        Eigen::MatrixBase<TDerivedx> const& x,
        Eigen::DenseBase<TDerivedY> const& Y,
        Eigen::DenseBase<TDerivednu> const& nu);
    /**
     * @brief Precomputes the sparsity pattern of the hessian matrix
     *
     * Enables parallel sparse hessian assembly in all future operations.
     */
    void PrecomputeHessianSparsity();
    /**
     * @brief Computes the element elasticity and its derivatives at the given shape
     *
     * @tparam TDerived Eigen matrix expression type
     * @param x \f$ d \times n \f$ matrix of deformed nodal positions
     * @param bWithGradient Compute gradient
     * @param bWithHessian Compute hessian
     * @param bUseSpdProjection Project per quadrature point hessians to nearest symmetric positive
     * definite (SPD) matrix
     */
    template <class TDerived>
    void ComputeElementElasticity(
        Eigen::MatrixBase<TDerived> const& x,
        bool bWithGradient     = true,
        bool bWithHessian      = true,
        bool bUseSpdProjection = true);

    /**
     * @brief Applies the hessian matrix of this potential as a linear operator on x, adding result
     * to y.
     *
     * @tparam TDerivedIn Input matrix type
     * @tparam TDerivedOut Output matrix type
     * @param x Input matrix
     * @param y Output matrix
     */
    template <class TDerivedIn, class TDerivedOut>
    void Apply(Eigen::MatrixBase<TDerivedIn> const& x, Eigen::DenseBase<TDerivedOut>& y) const;

    /**
     * @brief Transforms this matrix-free hessian matrix representation into sparse compressed
     * column format.
     * @return CSCMatrix Sparse compressed column matrix representation of the hessian operator
     */
    CSCMatrix ToMatrix() const;

    /**
     * @brief Transforms this per quadrature point gradient representation into the global gradient.
     * @return VectorX Global gradient
     */
    VectorX ToVector() const;

    /**
     * @brief Computes the total elastic potential
     * @return Scalar Total elastic potential
     */
    Scalar Eval() const;

    /**
     * @brief Number of input dimensions
     *
     * Effectively the number of nodes in the system
     *
     * @return Index
     */
    Index InputDimensions() const;
    /**
     * @brief Number of output dimensions
     *
     * Effectively the number of nodes in the system
     *
     * @return Index
     */
    Index OutputDimensions() const;
    /**
     * @brief Checks the validity of the held data
     */
    void CheckValidState() const;

    MeshType const& mesh; ///< The finite element mesh
    Eigen::Ref<IndexVectorX const>
        eg;                       ///< Maps quadrature point index g to its corresponding element e
    Eigen::Ref<VectorX const> wg; ///< Vector of quadrature weights \f$ w \in \mathbb{R}^{|Q|} \f$
    Eigen::Ref<MatrixX const>
        GNeg; ///< `|ElementType::kNodes| x |MeshType::kDims * # element quadrature points *
              ///< # elements|` shape function gradients at quadrature points

    VectorX mug; ///< 1st Lame coefficients \f$ \mu \in \mathbb{R}^{|Q|} \f$ at quadrature points
    VectorX lambdag; ///< 2nd Lame coefficients \f$ \lambda \in \mathbb{R}^{|Q|} \f$ at quadrature
                     ///< points
    MatrixX Hg;      ///< `|(ElementType::kNodes*kDims)| x |# quad.pts. *
                     ///< ElementType::kNodes*kDims|` element hessian matrices at quadrature points
    MatrixX Gg;      ///< `|ElementType::kNodes*kDims| x |#quad.pts.|` element gradient vectors at
                     ///< quadrature points
    VectorX Ug;      ///< `|# quad.pts.|` array of elastic potentials at quadrature points
    math::linalg::SparsityPattern GH; ///< Directed adjacency graph of hessian
};

template <CMesh TMesh, physics::CHyperElasticEnergy THyperElasticEnergy>
inline HyperElasticPotential<TMesh, THyperElasticEnergy>::HyperElasticPotential(
    MeshType const& meshIn,
    Eigen::Ref<IndexVectorX const> const& eg,
    Eigen::Ref<VectorX const> const& wg,
    Eigen::Ref<MatrixX const> const& GNeg,
    Scalar Y,
    Scalar nu)
    : HyperElasticPotential<TMesh, THyperElasticEnergy>(
          meshIn,
          eg,
          wg,
          GNeg,
          VectorX::Constant(wg.size(), Y),
          VectorX::Constant(wg.size(), nu))
{
}

template <CMesh TMesh, physics::CHyperElasticEnergy THyperElasticEnergy>
template <class TDerivedY, class TDerivednu>
inline HyperElasticPotential<TMesh, THyperElasticEnergy>::HyperElasticPotential(
    MeshType const& meshIn,
    Eigen::Ref<IndexVectorX const> const& eg,
    Eigen::Ref<VectorX const> const& wg,
    Eigen::Ref<MatrixX const> const& GNeg,
    Eigen::DenseBase<TDerivedY> const& Y,
    Eigen::DenseBase<TDerivednu> const& nu)
    : mesh(meshIn), eg(eg), wg(wg), GNeg(GNeg), mug(), lambdag(), Hg(), Gg(), Ug(), GH()
{
    std::tie(mug, lambdag)              = physics::LameCoefficients(Y.reshaped(), nu.reshaped());
    auto const numberOfQuadraturePoints = wg.size();
    auto constexpr kNodesPerElement     = ElementType::kNodes;
    auto constexpr kDofsPerElement      = kNodesPerElement * kDims;
    Ug.setZero(numberOfQuadraturePoints);
    Gg.setZero(kDofsPerElement, numberOfQuadraturePoints);
    Hg.setZero(kDofsPerElement, kDofsPerElement * numberOfQuadraturePoints);
}

template <CMesh TMesh, physics::CHyperElasticEnergy THyperElasticEnergy>
template <class TDerived>
inline HyperElasticPotential<TMesh, THyperElasticEnergy>::HyperElasticPotential(
    MeshType const& meshIn,
    Eigen::Ref<IndexVectorX const> const& eg,
    Eigen::Ref<VectorX const> const& wg,
    Eigen::Ref<MatrixX const> const& GNeg,
    Eigen::MatrixBase<TDerived> const& x,
    Scalar Y,
    Scalar nu)
    : HyperElasticPotential<TMesh, THyperElasticEnergy>(
          meshIn,
          eg,
          wg,
          GNeg,
          x,
          VectorX::Constant(wg.size(), Y),
          VectorX::Constant(wg.size(), nu))
{
}

template <CMesh TMesh, physics::CHyperElasticEnergy THyperElasticEnergy>
template <class TDerivedx, class TDerivedY, class TDerivednu>
inline HyperElasticPotential<TMesh, THyperElasticEnergy>::HyperElasticPotential(
    MeshType const& meshIn,
    Eigen::Ref<IndexVectorX const> const& eg,
    Eigen::Ref<VectorX const> const& wg,
    Eigen::Ref<MatrixX const> const& GNeg,
    Eigen::MatrixBase<TDerivedx> const& x,
    Eigen::DenseBase<TDerivedY> const& Y,
    Eigen::DenseBase<TDerivednu> const& nu)
    : HyperElasticPotential<TMesh, THyperElasticEnergy>(meshIn, eg, wg, GNeg, Y, nu)
{
    ComputeElementElasticity(x);
}

template <CMesh TMesh, physics::CHyperElasticEnergy THyperElasticEnergy>
template <class TDerived>
inline void HyperElasticPotential<TMesh, THyperElasticEnergy>::ComputeElementElasticity(
    Eigen::MatrixBase<TDerived> const& x,
    bool bWithGradient,
    bool bWithHessian,
    bool bUseSpdProjection)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.fem.HyperElasticPotential.ComputeElementElasticity");
    // Check inputs
    CheckValidState();
    auto const numberOfNodes = mesh.X.cols();
    if (x.size() != numberOfNodes * kDims)
    {
        std::string const what = fmt::format(
            "Generalized coordinate vector must have dimensions |#nodes|*kDims={}, but got "
            "x.size()={}",
            numberOfNodes * kDims,
            x.size());
        throw std::invalid_argument(what);
    }

    Ug.setZero();
    if (bWithGradient)
        Gg.setZero();
    if (bWithHessian)
        Hg.setZero();

    ElasticEnergyType Psi{};

    // Compute element elastic energies and their derivatives
    auto const numberOfQuadraturePoints = wg.size();
    auto constexpr kNodesPerElement     = ElementType::kNodes;
    auto constexpr kDofsPerElement      = kNodesPerElement * kDims;
    namespace mini                      = math::linalg::mini;
    using mini::FromEigen;
    using mini::ToEigen;
    if (not bWithGradient and not bWithHessian)
    {
        tbb::parallel_for(Index{0}, Index{numberOfQuadraturePoints}, [&](Index g) {
            auto const e     = eg(g);
            auto const nodes = mesh.E.col(e);
            auto const xe    = x.reshaped(kDims, numberOfNodes)(Eigen::placeholders::all, nodes);
            auto const GPeg = GNeg.block<kNodesPerElement, MeshType::kDims>(0, g * MeshType::kDims);
            Matrix<kDims, kDims> const F = xe * GPeg;
            auto vecF                    = FromEigen(F);
            auto psiF                    = Psi.eval(vecF, mug(g), lambdag(g));
            Ug(g) += wg(g) * psiF;
        });
    }
    else if (bWithGradient and not bWithHessian)
    {
        tbb::parallel_for(Index{0}, Index{numberOfQuadraturePoints}, [&](Index g) {
            auto const e     = eg(g);
            auto const nodes = mesh.E.col(e);
            auto const xe    = x.reshaped(kDims, numberOfNodes)(Eigen::placeholders::all, nodes);
            auto const GPeg = GNeg.block<kNodesPerElement, MeshType::kDims>(0, g * MeshType::kDims);
            Matrix<kDims, kDims> const F = xe * GPeg;
            auto vecF                    = FromEigen(F);
            mini::SVector<Scalar, kDims * kDims> gradPsiF;
            auto psiF = Psi.evalWithGrad(vecF, mug(g), lambdag(g), gradPsiF);
            Ug(g) += wg(g) * psiF;
            auto const GP = FromEigen(GPeg);
            auto GPsix    = GradientWrtDofs<ElementType, kDims>(gradPsiF, GP);
            Gg.col(g) += wg(g) * ToEigen(GPsix);
        });
    }
    else if (not bWithGradient and bWithHessian)
    {
        tbb::parallel_for(Index{0}, Index{numberOfQuadraturePoints}, [&](Index g) {
            auto const e     = eg(g);
            auto const nodes = mesh.E.col(e);
            auto const xe    = x.reshaped(kDims, numberOfNodes)(Eigen::placeholders::all, nodes);
            auto const gradPhi =
                GNeg.block<kNodesPerElement, MeshType::kDims>(0, g * MeshType::kDims);
            Matrix<kDims, kDims> const F = xe * gradPhi;
            auto vecF                    = FromEigen(F);
            auto psiF                    = Psi.eval(vecF, mug(g), lambdag(g));
            auto hessPsiF                = Psi.hessian(vecF, mug(g), lambdag(g));
            Ug(g) += wg(g) * psiF;
            auto const GP = FromEigen(gradPhi);
            auto HPsix    = HessianWrtDofs<ElementType, kDims>(hessPsiF, GP);
            auto heg      = Hg.block<kDofsPerElement, kDofsPerElement>(0, g * kDofsPerElement);
            heg += wg(g) * ToEigen(HPsix);
        });
    }
    else
    {
        tbb::parallel_for(Index{0}, Index{numberOfQuadraturePoints}, [&](Index g) {
            auto const e     = eg(g);
            auto const nodes = mesh.E.col(e);
            auto const xe    = x.reshaped(kDims, numberOfNodes)(Eigen::placeholders::all, nodes);
            auto const GPeg = GNeg.block<kNodesPerElement, MeshType::kDims>(0, g * MeshType::kDims);
            Matrix<kDims, kDims> const F = xe * GPeg;
            auto vecF                    = FromEigen(F);
            mini::SVector<Scalar, kDims * kDims> gradPsiF;
            mini::SMatrix<Scalar, kDims * kDims, kDims * kDims> hessPsiF;
            auto psiF = Psi.evalWithGradAndHessian(vecF, mug(g), lambdag(g), gradPsiF, hessPsiF);
            auto const GP = FromEigen(GPeg);
            auto GPsix    = GradientWrtDofs<ElementType, kDims>(gradPsiF, GP);
            auto HPsix    = HessianWrtDofs<ElementType, kDims>(hessPsiF, GP);
            auto heg      = Hg.block<kDofsPerElement, kDofsPerElement>(0, g * kDofsPerElement);
            Ug(g) += wg(g) * psiF;
            Gg.col(g) += wg(g) * ToEigen(GPsix);
            heg += wg(g) * ToEigen(HPsix);
        });
    }
    if (bWithHessian and bUseSpdProjection)
    {
        tbb::parallel_for(Index{0}, Index{numberOfQuadraturePoints}, [&](Index g) {
            auto heg = Hg.block<kDofsPerElement, kDofsPerElement>(0, g * kDofsPerElement);
            Eigen::JacobiSVD<
                Matrix<kDofsPerElement, kDofsPerElement>,
                Eigen::ComputeFullU | Eigen::ComputeFullV>
                SVD{};
            SVD.compute(heg);
            Vector<kDofsPerElement> sigma = SVD.singularValues();
            for (auto s = sigma.size() - 1; s >= 0; --s)
            {
                if (sigma(s) >= 0.)
                    break;
                sigma(s) = -sigma(s);
            }
            heg = SVD.matrixU() * sigma.asDiagonal() * SVD.matrixV().transpose();
        });
    }
}

template <CMesh TMesh, physics::CHyperElasticEnergy THyperElasticEnergy>
template <class TDerivedIn, class TDerivedOut>
inline void HyperElasticPotential<TMesh, THyperElasticEnergy>::Apply(
    Eigen::MatrixBase<TDerivedIn> const& x,
    Eigen::DenseBase<TDerivedOut>& y) const
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.fem.HyperElasticPotential.Apply");
    auto const numberOfDofs = InputDimensions();
    if (x.rows() != numberOfDofs or y.rows() != numberOfDofs or x.cols() != y.cols())
    {
        std::string const what = fmt::format(
            "Expected inputs and outputs to have rows |#nodes*kDims|={} and same number of "
            "columns, but got dimensions "
            "x,y=({},{}), ({},{})",
            numberOfDofs,
            x.rows(),
            x.cols(),
            y.rows(),
            y.cols());
        throw std::invalid_argument(what);
    }

    auto constexpr kDofsPerElement      = kDims * ElementType::kNodes;
    auto const numberOfQuadraturePoints = wg.size();
    // NOTE: Outer loop could be parallelized over columns, and using graph coloring, inner loop
    // could also be parallelized, if it's worth it.
    for (auto c = 0; c < x.cols(); ++c)
    {
        for (auto g = 0; g < numberOfQuadraturePoints; ++g)
        {
            auto const e     = eg(g);
            auto const nodes = mesh.E.col(e);
            auto const heg   = Hg.block<kDofsPerElement, kDofsPerElement>(0, g * kDofsPerElement);
            auto const xe =
                x.col(c).reshaped(kDims, x.size() / kDims)(Eigen::placeholders::all, nodes);
            auto ye = y.col(c).reshaped(kDims, y.size() / kDims)(Eigen::placeholders::all, nodes);
            ye.reshaped() += heg * xe.reshaped();
        }
    }
}

template <CMesh TMesh, physics::CHyperElasticEnergy THyperElasticEnergy>
inline void HyperElasticPotential<TMesh, THyperElasticEnergy>::PrecomputeHessianSparsity()
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.fem.HyperElasticPotential.PrecomputeHessianSparsity");
    auto const numberOfQuadraturePoints = wg.size();
    auto const kNodesPerElement         = ElementType::kNodes;
    auto const kDofsPerElement          = kNodesPerElement * kDims;
    std::vector<Index> nonZeroRowIndices{};
    std::vector<Index> nonZeroColIndices{};
    nonZeroRowIndices.reserve(
        static_cast<std::size_t>(kDofsPerElement * kDofsPerElement * numberOfQuadraturePoints));
    nonZeroColIndices.reserve(
        static_cast<std::size_t>(kDofsPerElement * kDofsPerElement * numberOfQuadraturePoints));
    // Insert non-zero indices in the storage order of our Hg matrix of element hessians at
    // quadrature points
    for (auto g = 0; g < numberOfQuadraturePoints; ++g)
    {
        auto const e     = eg(g);
        auto const nodes = mesh.E.col(e);
        for (auto j = 0; j < kNodesPerElement; ++j)
        {
            for (auto dj = 0; dj < kDims; ++dj)
            {
                for (auto i = 0; i < kNodesPerElement; ++i)
                {
                    for (auto di = 0; di < kDims; ++di)
                    {
                        nonZeroRowIndices.push_back(kDims * nodes(i) + di);
                        nonZeroColIndices.push_back(kDims * nodes(j) + dj);
                    }
                }
            }
        }
    }
    GH.Compute(OutputDimensions(), InputDimensions(), nonZeroRowIndices, nonZeroColIndices);
}

template <CMesh TMesh, physics::CHyperElasticEnergy THyperElasticEnergy>
inline CSCMatrix HyperElasticPotential<TMesh, THyperElasticEnergy>::ToMatrix() const
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.fem.HyperElasticPotential.ToMatrix");
    if (!GH.IsEmpty())
    {
        using SpanType = std::span<Scalar const>;
        using SizeType = typename SpanType::size_type;
        return GH.ToMatrix(SpanType(Hg.data(), static_cast<SizeType>(Hg.size())));
    }
    else
    {
        // Construct hessian from triplets
        using SparseIndex = typename CSCMatrix::StorageIndex;
        using Triplet     = Eigen::Triplet<Scalar, SparseIndex>;
        std::vector<Triplet> triplets{};
        triplets.reserve(static_cast<std::size_t>(Hg.size()));
        auto const numberOfQuadraturePoints = wg.size();
        for (auto g = 0; g < numberOfQuadraturePoints; ++g)
        {
            auto const e         = eg(g);
            auto const nodes     = mesh.E.col(e);
            auto constexpr Hrows = ElementType::kNodes * kDims;
            auto constexpr Hcols = Hrows;
            auto const heg       = Hg.block<Hrows, Hcols>(0, g * Hcols);
            for (auto j = 0; j < ElementType::kNodes; ++j)
                for (auto dj = 0; dj < kDims; ++dj)
                    for (auto i = 0; i < ElementType::kNodes; ++i)
                        for (auto di = 0; di < kDims; ++di)
                            triplets.push_back(Triplet{
                                static_cast<SparseIndex>(kDims * nodes(i) + di),
                                static_cast<SparseIndex>(kDims * nodes(j) + dj),
                                heg(kDims * i + di, kDims * j + dj)});
        }

        auto const n = InputDimensions();
        CSCMatrix H(n, n);
        H.setFromTriplets(triplets.begin(), triplets.end());
        return H;
    }
}

template <CMesh TMesh, physics::CHyperElasticEnergy THyperElasticEnergy>
inline VectorX HyperElasticPotential<TMesh, THyperElasticEnergy>::ToVector() const
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.fem.HyperElasticPotential.ToVector");
    auto constexpr kNodesPerElement     = ElementType::kNodes;
    auto const numberOfQuadraturePoints = wg.size();
    auto const numberOfNodes            = mesh.X.cols();
    auto const n                        = InputDimensions();
    VectorX G                           = VectorX::Zero(n);
    for (auto g = 0; g < numberOfQuadraturePoints; ++g)
    {
        auto const e     = eg(g);
        auto const nodes = mesh.E.col(e);
        auto const geg   = Gg.col(g).reshaped(kDims, kNodesPerElement);
        auto gi          = G.reshaped(kDims, numberOfNodes)(Eigen::placeholders::all, nodes);
        gi += geg;
    }
    return G;
}

template <CMesh TMesh, physics::CHyperElasticEnergy THyperElasticEnergy>
inline Scalar HyperElasticPotential<TMesh, THyperElasticEnergy>::Eval() const
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.fem.HyperElasticPotential.Eval");
    return Ug.sum();
}

template <CMesh TMesh, physics::CHyperElasticEnergy THyperElasticEnergy>
inline Index HyperElasticPotential<TMesh, THyperElasticEnergy>::InputDimensions() const
{
    auto const numberOfNodes = mesh.X.cols();
    auto const numberOfDofs  = numberOfNodes * kDims;
    return numberOfDofs;
}

template <CMesh TMesh, physics::CHyperElasticEnergy THyperElasticEnergy>
inline Index HyperElasticPotential<TMesh, THyperElasticEnergy>::OutputDimensions() const
{
    return InputDimensions();
}

template <CMesh TMesh, physics::CHyperElasticEnergy THyperElasticEnergy>
inline void HyperElasticPotential<TMesh, THyperElasticEnergy>::CheckValidState() const
{
    auto const numberOfQuadraturePoints = wg.size();
    auto constexpr kExpectedGNegRows    = ElementType::kNodes;
    auto const expectedGNegCols         = MeshType::kDims * numberOfQuadraturePoints;
    bool const bShapeFunctionGradientsHaveCorrectDimensions =
        (GNeg.rows() == kExpectedGNegRows) and (GNeg.cols() == expectedGNegCols);
    if (not bShapeFunctionGradientsHaveCorrectDimensions)
    {
        std::string const what = fmt::format(
            "Expected shape function gradients at element quadrature points of dimensions "
            "|#nodes-per-element|={} x |#mesh-dims * #quad.pts.|={} for but got {}x{} instead",
            kExpectedGNegRows,
            expectedGNegCols,
            GNeg.rows(),
            GNeg.cols());
        throw std::invalid_argument(what);
    }
    bool const bLameCoefficientsHaveCorrectDimensions =
        (mug.size() == numberOfQuadraturePoints) and (lambdag.size() == numberOfQuadraturePoints);
    if (not bLameCoefficientsHaveCorrectDimensions)
    {
        std::string const what = fmt::format(
            "Expected quadrature point lame coefficients with dimensions {0}x1 and "
            "{0}x1 for mug and lambdag, but got {1}x1 and {2}x1 instead.",
            numberOfQuadraturePoints,
            mug.size(),
            lambdag.size());
        throw std::invalid_argument(what);
    }
}

} // namespace fem
} // namespace pbat

#endif // PBA_FEM_HYPER_ELASTIC_POTENTIAL_H