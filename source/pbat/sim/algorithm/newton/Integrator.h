/**
 * @file Integrator.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Header file for the Newton integrator.
 * @date 2025-05-05
 * @copyright Copyright (c) 2025
 */

#ifndef PBAT_SIM_ALGORITHM_NEWTON_INTEGRATOR_H
#define PBAT_SIM_ALGORITHM_NEWTON_INTEGRATOR_H

#include "Config.h"
#include "pbat/Aliases.h"
#include "pbat/fem/Tetrahedron.h"
#include "pbat/io/Archive.h"
#include "pbat/math/optimization/LineSearch.h"
#include "pbat/math/optimization/Newton.h"
#include "pbat/physics/StableNeoHookeanEnergy.h"
#include "pbat/sim/contact/MultibodyTetrahedralMeshSystem.h"
#include "pbat/sim/dynamics/FemElastoDynamics.h"

#ifdef PBAT_USE_SUITESPARSE
    #include <Eigen/CholmodSupport>
#else
    #include <Eigen/SparseCholesky>
#endif // PBAT_USE_SUITESPARSE
#include <memory>
#include <optional>
#include <vector>

namespace pbat::sim::algorithm::newton {

/**
 * @brief Newton dynamics integrator.
 */
class Integrator
{
  public:
    static auto constexpr kDims  = 3; ///< Number of spatial dimensions
    static auto constexpr kOrder = 1; ///< Shape function order

    using ScalarType     = Scalar; ///< Floating point scalar type
    using IndexType      = Index;  ///< Integer index type
    using MeshSystemType = contact::MultibodyTetrahedralMeshSystem<IndexType>; ///< Mesh system type
    using ElastoDynamicsType = dynamics::FemElastoDynamics<
        fem::Tetrahedron<1>,
        kDims,
        physics::StableNeoHookeanEnergy<kDims>,
        ScalarType,
        IndexType>; ///< Elasto-dynamics type
    using HessianMatrixType =
        Eigen::SparseMatrix<ScalarType, Eigen::ColMajor, IndexType>; ///< Hessian matrix type
#ifdef PBAT_USE_SUITESPARSE
    using DecompositionType =
        Eigen::CholmodDecomposition<HessianMatrixType, Eigen::Upper>; ///< Hessian decomposition
                                                                      ///< type
#else
    using DecompositionType =
        Eigen::SimplicialLDLT<HessianMatrixType, Eigen::Upper>; ///< Hessian decomposition type
#endif // PBAT_USE_SUITESPARSE
    /**
     * @brief Copy constructor
     * @param other Other integrator to copy from
     */
    Integrator(Integrator const& other);
    /**
     * @brief Move constructor
     * @param other Other integrator to move from
     */
    Integrator(Integrator&& other) noexcept = default;
    /**
     * @brief Copy assignment operator
     * @param other Other integrator to copy from
     * @return Reference to this integrator
     */
    Integrator& operator=(Integrator const& other);
    /**
     * @brief Move assignment operator
     * @param other Other integrator to move from
     * @return Reference to this integrator
     */
    Integrator& operator=(Integrator&& other) noexcept = default;
    /**
     * @brief Construct a new Newton integrator.
     * @param config Configuration for the Newton integrator.
     * @param elastoDynamics Elasto-dynamics object.
     */
    Integrator(Config config, MeshSystemType meshSystem, ElastoDynamicsType elastoDynamics);
    /**
     * @brief Perform a single time step of the Newton integrator.
     * @param archive Optional archive to save the state after the step.
     */
    void Step(std::optional<io::Archive> archive = std::nullopt);
    /**
     * @brief Get the Elasto Dynamics object
     * @return ElastoDynamicsType const&
     */
    [[maybe_unused]] auto GetElastoDynamics() const -> ElastoDynamicsType const&
    {
        return mElastoDynamics;
    }
    /**
     * @brief Get the Elasto Dynamics object
     * @return ElastoDynamicsType&
     */
    [[maybe_unused]] auto GetElastoDynamics() -> ElastoDynamicsType& { return mElastoDynamics; }
    /**
     * @brief Get the Mesh System object
     * @return MeshSystemType const&
     */
    [[maybe_unused]] auto GetMeshSystem() const -> MeshSystemType const& { return mMeshes; }
    /**
     * @brief Get the Mesh System object
     * @return MeshSystemType&
     */
    [[maybe_unused]] auto GetMeshSystem() -> MeshSystemType& { return mMeshes; }
    /**
     * @brief Get the Config object
     * @return Config const&
     */
    [[maybe_unused]] auto GetConfig() const -> Config const& { return mConfig; }
    /**
     * @brief Get the Config object
     * @return Config&
     */
    [[maybe_unused]] auto GetConfig() -> Config& { return mConfig; }

  protected:
    /**
     * @brief Assemble the Hessian matrix.
     */
    void AssembleHessian(ScalarType bt2);
    /**
     * @brief Apply the current configuration to the integrator.
     */
    void ApplyConfig();

  private:
    ElastoDynamicsType mElastoDynamics; ///< Hyper elasticity dynamics
    contact::MultibodyTetrahedralMeshSystem<IndexType>
        mMeshes;                                    ///< Multibody tetrahedral mesh system
    Config mConfig;                                 ///< Configuration for the Newton integrator
    math::optimization::Newton<ScalarType> mNewton; ///< Newton optimization solver
    math::optimization::BackTrackingLineSearch<ScalarType> mLineSearch; ///< Line searcher
    std::vector<Eigen::Triplet<ScalarType, IndexType>>
        mTriplets;                                      ///< Triplets for assembling the Hessian
    std::unique_ptr<DecompositionType> mInverseHessian; ///< Inverse hessian
    HessianMatrixType mHessian;                         ///< Time integration optimization hessian
    Eigen::Vector<ScalarType, Eigen::Dynamic> mGrad;    ///< Time integration optimization gradient
};

} // namespace pbat::sim::algorithm::newton

#endif // PBAT_SIM_ALGORITHM_NEWTON_INTEGRATOR_H
