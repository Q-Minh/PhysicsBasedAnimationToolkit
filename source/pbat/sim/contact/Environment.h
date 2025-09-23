/**
 * @file Environment.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Static environment geometry for contact simulation
 * @version 0.1
 * @date 2025-09-23
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_SIM_CONTACT_ENVIRONMENT_H
#define PBAT_SIM_CONTACT_ENVIRONMENT_H

#include "MultibodyTetrahedralMeshSystem.h"
#include "pbat/Aliases.h"
#include "pbat/common/Concepts.h"
#include "pbat/geometry/sdf/Forest.h"
#include "pbat/io/Archive.h"

#include <Eigen/Core>

namespace pbat::sim::contact {

/**
 * @brief Static environment geometry for contact simulation
 * @tparam TScalar Scalar type
 */
template <common::CFloatingPoint TScalar = Scalar>
class Environment
{
  public:
    using ScalarType = TScalar; ///< Scalar type
    using SdfType    = pbat::geometry::sdf::Forest<ScalarType>;

    /**
     * @brief Construct a new Environment object
     */
    Environment() = default;
    /**
     * @brief Construct a new Environment object
     * @param sdf Signed distance field representing the environment
     * @param nPrimitives Number of primitives to reserve space for
     */
    Environment(SdfType sdf, Eigen::Index nPrimitives = 0);
    /**
     * @brief Set the signed distance field representing the environment
     * @param sdf Signed distance field representing the environment
     */
    void SetSdf(SdfType sdf);
    /**
     * @brief Reserve space for contact primitives
     * @param nPrimitives Number of primitives to reserve space for
     */
    void Reserve(Eigen::Index nPrimitives);
    /**
     * @brief
     * @return
     */
    template <common::CIndex TIndex = Index>
    void DetectContactCandidates(MultibodyTetrahedralMeshSystem<TIndex, ScalarType> const& meshes);
    /**
     * @brief Get the number of contact candidates
     * @return The number of contact candidates
     */
    Eigen::Index NumContactCandidates() const { return nCandidates; }
    /**
     * @brief Check if a primitive is a contact candidate
     * @param i Index of the primitive in [0, |# primitives|)
     * @return True if the primitive is a contact candidate
     */
    bool IsContactCandidate(Eigen::Index i) const { return mIsCandidate[i]; }
    /**
     * @brief Get the contact candidates
     * @return `|# candidates| x 1` indices of contact candidate primitives in [0, |# primitives|)
     */
    auto ContactCandidates() const { return mCandidates.head(nCandidates); }
    /**
     * @brief Serialize the environment to an archive
     * @param archive Archive to serialize to
     */
    void Serialize(io::Archive& archive) const;
    /**
     * @brief Deserialize the environment from an archive
     * @param archive Archive to deserialize from
     */
    void Deserialize(io::Archive const& archive);

    Eigen::Vector<bool, Eigen::Dynamic>
        mIsCandidate; ///< `|# primitives|` whether each point is in contact
    Eigen::Vector<Eigen::Index, Eigen::Dynamic>
        mCandidates;          ///< `|# primitives|` indices of active primitives in [0, nCandidates)
    Eigen::Index nCandidates; ///< Number of active primitives
    Eigen::Vector<ScalarType, Eigen::Dynamic>
        mPhi; ///< `|# primitives|` signed distance values at each contact point (negative inside,
              ///< positive outside)
    Eigen::Matrix<ScalarType, 3, Eigen::Dynamic>
        mPoint; ///< `3 x |# primitives|` contact point on the surface of each primitive (undefined
                ///< if phi > 0)
    Eigen::Matrix<ScalarType, 3 * 3, Eigen::Dynamic>
        mBasis; ///< `3*3 x |# primitives|` contact basis \f$ \begin{bmatrix} \mathbf{n} &
                ///< \mathbf{t} & \mathbf{b} \end{bmatrix} \f$ at each contact point (undefined if
                ///< phi > 0), such that mBasis.col(i).reshaped<3,3>() is the basis for primitive i
    pbat::geometry::sdf::Forest<ScalarType>
        mSdf; ///< Signed distance field representing the environment
};

} // namespace pbat::sim::contact

#endif // PBAT_SIM_CONTACT_ENVIRONMENT_H
