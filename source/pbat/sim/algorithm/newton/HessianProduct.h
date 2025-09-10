/**
 * @file HessianProduct.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Header file for Hessian product algorithms.
 * @date 2025-05-05
 * @copyright Copyright (c) 2025
 */

#ifndef PBAT_SIM_ALGORITHM_NEWTON_HESSIANPRODUCT_H
#define PBAT_SIM_ALGORITHM_NEWTON_HESSIANPRODUCT_H

#include "pbat/Aliases.h"
#include "pbat/common/ConstexprFor.h"
#include "pbat/math/linalg/SelectionMatrix.h"
#include "pbat/math/linalg/SparsityPattern.h"

#include <vector>

namespace pbat::sim::algorithm::newton {
struct HessianOperator;
} // namespace pbat::sim::algorithm::newton

namespace Eigen {
namespace internal {

/**
 * @brief Traits specialization for the Hessian inverse product
 */
template <>
struct traits<pbat::sim::algorithm::newton::HessianOperator>
    : public Eigen::internal::traits<pbat::CSCMatrix>
{
};

} // namespace internal
} // namespace Eigen

namespace pbat::sim::algorithm::newton {

/**
 * @brief Concept for elastic potential
 * @note Should probably move to pbat/fem/HyperElasticPotential.h or other appropriate location
 * @tparam T Elastic potential type
 */
template <class T>
concept CElasticPotential = requires(T U)
{
    {U.ComputeElementElasticity(std::declval<VectorX>(), false, true, true)};
    {U.ToMatrix(std::declval<CSCMatrix&>())};
    {U.GH}->std::convertible_to<math::linalg::SparsityPattern<>>;
};

/**
 * @brief Storage for the dynamics time integration optimization Hessian
 */
struct Hessian
{
    static auto constexpr kDims        = 3; ///< Number of spatial dimensions
    static auto constexpr kContactInds = 4; ///< Number of block indices per contact
    using TripletType = Eigen::Triplet<Scalar, CSCMatrix::StorageIndex>; ///< Triplet type
    /**
     * @brief Allocate initial memory for the contact hessian blocks
     * @param nContacts Number of contact hessian blocks
     */
    void AllocateContacts(std::size_t nContacts);
    /**
     * @brief Compute the necessary sparsity patterns of the hessian
     * @param HS `kDims*n x kDims*n` value-less sparse matrix with same sparsity pattern as the
     * FEM elastic potential
     * @param dims Number of spatial dimensions
     */
    void SetSparsityPattern(CSCMatrix const& HS);
    /**
     * @brief Impose Dirichlet constraints on the hessian, i.e. remove rows/cols corresponding to
     * fixed dofs
     * @tparam TDerivedF Eigen type of the free variable indices
     * @param F `# dofs x 1` vector of free variable indices
     */
    template <class TDerivedF>
    void ImposeDirichletConstraints(Eigen::DenseBase<TDerivedF> const& F);
    /**
     * @brief Add contact hessian block
     * @tparam TDerivedHCB Eigen type of the contact hessian block
     * @tparam TDerivedHCIB Eigen type of the contact hessian block indices
     * @param HCB `kDims*kInds x kDims*kInds` contact hessian block coefficients, where `kInds =
     * ContactBlock::kInds`
     * @param HCIB `kInds x 1` non-zero block indices of the contact hessian block, where `kInds =
     * ContactBlock::kInds`
     */
    template <class TDerivedHCB, class TDerivedHCIB>
    void AddContactHessianBlock(
        Eigen::DenseBase<TDerivedHCB> const& HCB,
        Eigen::DenseBase<TDerivedHCIB> const& HCIB);
    /**
     * @brief Construct the contact-less hessian
     *
     * Construct \f$ M + \tilde{\beta}^2 \nabla^2 U \f$,
     * where
     * \f$ M \f$ is the mass matrix, \f$ \tilde{\beta} \f$ is the BDF forcing term coefficient, \f$
     * U \f$ is the elastic potential.
     *
     * @tparam TElasticPotential Elastic potential type
     * @tparam TDerivedM Eigen type of the mass matrix
     * @param diagM `kDims*n x 1` vector of lumped mass matrix diagonal coefficients
     * @param bt2 BDF forcing term coefficient squared
     * @param U Elastic potential object
     */
    template <CElasticPotential TElasticPotential, class TDerivedM>
    void ConstructContactLessHessian(
        Eigen::DenseBase<TDerivedM> const& diagM,
        Scalar bt2,
        TElasticPotential const& U);
    /**
     * @brief Construct the contact hessian
     */
    void ConstructContactHessian();

    IndexVectorX diag; ///< `kDims*n x 1` vector of indices for the contact-less hessian's diagonal
    CSCMatrix HNC;     ///< `kDims*n x kDims*n` contact-less Hessian matrix
    std::vector<TripletType>
        HCij; ///< `|# edge-edge contacts + # vertex-face contacts|*(kDims*kContactInds)^2` list of
              ///< contact hessian non-zero entries
    CSCMatrix HC; ///< `kDims*n x kDims*n` contact hessian matrix
    CSCMatrix S;  ///< `kDims*n x # dofs` selection matrix for the free variables s.t. \f$ H_{ff} =
                  ///< S^T H S \$ is the free Hessian
};

/**
 * @brief Hessian operator
 *
 * Non-owning view over stateful Hessian, to use with Eigen's matrix-free solvers.
 */
struct HessianOperator : public Eigen::EigenBase<HessianOperator>
{
    using Scalar       = pbat::Scalar;                     ///< Eigen typedef
    using RealScalar   = pbat::Scalar;                     ///< Eigen typedef
    using StorageIndex = typename CSCMatrix::StorageIndex; ///< Eigen typedef
    enum {
        ColsAtCompileTime    = Eigen::Dynamic,
        MaxColsAtCompileTime = Eigen::Dynamic,
        IsRowMajor           = false
    };
    using SelfType = HessianOperator; ///< Type of this object
    /**
     * @brief Construct an empty Hessian Operator object
     */
    HessianOperator() = default;
    /**
     * @brief Construct a new HessianOperator object
     * @param data Pointer to the elasto-dynamics hessian data
     */
    HessianOperator(Hessian* data);
    /**
     * @brief Eigen API
     * @return Number of hessian rows
     */
    [[maybe_unused]] Eigen::Index rows() const { return mData->HNC.rows(); }
    /**
     * @brief Eigen API
     * @return Number of hessian columns
     */
    [[maybe_unused]] Eigen::Index cols() const { return mData->HNC.cols(); }
    /**
     * @brief Hessian product Eigen expression
     *
     * @tparam Rhs Eigen type of the right-hand side
     * @param x Right-hand side vector/matrix
     * @return Eigen product expression
     */
    template <class Rhs>
    Eigen::Product<SelfType, Rhs, Eigen::AliasFreeProduct>
    operator*(Eigen::MatrixBase<Rhs> const& x) const
    {
        return Eigen::Product<SelfType, Rhs, Eigen::AliasFreeProduct>(*this, x.derived());
    }

    Hessian* mData; ///< Pointer to the elasto-dynamics hessian data
};

template <class TDerivedF>
inline void Hessian::ImposeDirichletConstraints(Eigen::DenseBase<TDerivedF> const& F)
{
    S   = math::linalg::SelectionMatrix(F, diag.size());
    HNC = S.transpose() * (HNC * S);
    HC  = S.transpose() * (HC * S);
}

template <class TDerivedHCB, class TDerivedHCIB>
inline void Hessian::AddContactHessianBlock(
    Eigen::DenseBase<TDerivedHCB> const& HCB,
    Eigen::DenseBase<TDerivedHCIB> const& HCIB)
{
    pbat::common::ForRange<0, kContactInds>([&]<auto kj>() {
        pbat::common::ForRange<0, kContactInds>([&]<auto ki>() {
            auto const bi     = HCIB(ki);
            auto const bj     = HCIB(kj);
            auto const istart = bi * kDims;
            auto const jstart = bj * kDims;
            auto const HCbibj = HCB.block<kDims, kDims>(ki, kj);
            using IndexType   = std::remove_cvref_t<decltype(std::declval<TripletType>().row())>;
            pbat::common::ForRange<0, kDims>([&]<auto j>() {
                pbat::common::ForRange<0, kDims>([&]<auto i>() {
                    HCij.emplace_back(
                        static_cast<IndexType>(istart + i),
                        static_cast<IndexType>(jstart + j),
                        HCbibj(i, j));
                });
            });
        });
    });
}

template <CElasticPotential TElasticPotential, class TDerivedM>
inline void Hessian::ConstructContactLessHessian(
    Eigen::DenseBase<TDerivedM> const& diagM,
    Scalar bt2,
    TElasticPotential const& U)
{
    U.ToMatrix(HNC);
    HNC *= bt2;
    HNC.coeffs()(diag) += diagM;
}

} // namespace pbat::sim::algorithm::newton

namespace Eigen {
namespace internal {

/**
 * @brief Generic product implementation for the Hessian inverse product
 *
 * @tparam Rhs Right-hand side matrix or vector expression
 * @tparam ProductType Product type
 */
template <typename Rhs, int ProductType>
struct generic_product_impl<
    pbat::sim::algorithm::newton::HessianOperator,
    Rhs,
    SparseShape,
    DenseShape,
    ProductType>
    : generic_product_impl_base<
          pbat::sim::algorithm::newton::HessianOperator,
          Rhs,
          generic_product_impl<pbat::sim::algorithm::newton::HessianOperator, Rhs>>
{
    using HessianOperator = pbat::sim::algorithm::newton::HessianOperator;
    using Scalar          = typename Product<HessianOperator, Rhs>::Scalar;
    using Hessian         = pbat::sim::algorithm::newton::Hessian;
    /**
     * @brief Compute the product of the Hessian operator and a vector/matrix expression
     *
     * @tparam Dst
     * @param dst
     * @param lhs
     * @param rhs
     * @param alpha
     */
    template <typename Dst>
    static void scaleAndAddTo(
        Eigen::MatrixBase<Dst>& dst,
        HessianOperator const& lhs,
        Eigen::MatrixBase<Rhs> const& rhs,
        Scalar const& alpha)
    {
        dst += alpha * (lhs.mData->HNC * rhs + lhs.mData->HC * rhs);
    }
};

} // namespace internal
} // namespace Eigen

#endif // PBAT_SIM_ALGORITHM_NEWTON_HESSIANPRODUCT_H
