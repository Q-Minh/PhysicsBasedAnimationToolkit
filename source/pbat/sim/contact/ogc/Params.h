/**
 * @file Params.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Parameters for Offset Geometry Contact (OGC) algorithm.
 * @version 0.1
 * @date 2025-12-10
 * @copyright Copyright (c) 2025
 */
#ifndef PBAT_SIM_CONTACT_OGC_PARAMS_H
#define PBAT_SIM_CONTACT_OGC_PARAMS_H

#include "Enums.h"
#include "PhysicsBasedAnimationToolkitExport.h"
#include "pbat/Aliases.h"
#include "pbat/common/Concepts.h"
#include "pbat/io/Archive.h"

namespace pbat::sim::contact::ogc {

/**
 * @brief Parameters for Offset Geometry Contact (OGC).
 *
 * This bundles algorithm configuration knobs and capacities. Mesh data and device are
 * intentionally not stored here per design; this type is for configurable parameters only.
 */
template <common::CFloatingPoint TScalar>
struct Params
{
    using ScalarType = TScalar;

    ScalarType r{ScalarType(0)};  ///< Contact radius
    ScalarType rq{ScalarType(0)}; ///< Query inflation radius

    ESceneFeatures eSceneFeatures{ESceneFeatures::None};       ///< Scene features
    EBuildQuality eDynamicSceneBvhQuality{EBuildQuality::Low}; ///< Scene BVH build quality
    EBuildQuality eDynamicMeshBvhQuality{EBuildQuality::Low};  ///< Mesh BVH build quality
    EBuildQuality eStaticSceneBvhQuality{EBuildQuality::High}; ///< Scene BVH build quality
    EBuildQuality eStaticMeshBvhQuality{EBuildQuality::High};  ///< Mesh BVH build quality

    int nMaxVertexFaceContactsEstimate{
        16}; ///< Max vertex-face contacts estimate for memory pre-allocation
    int nMaxFaceVertexContactsEstimate{
        16}; ///< Max face-vertex contacts estimate for memory pre-allocation
    int nMaxEdgeFaceContactsEstimate{
        16}; ///< Max edge-face contacts estimate for memory pre-allocation

    ScalarType gammap{0.45}; ///< Relaxation parameter for vertex displacement bound, must satisfy
                             ///< `0 < gammap < 0.5`
    ScalarType gammae{
        0.1}; ///< Proportion of bounds-violating vertices to trigger collision detection

  public:
    /**
     * @brief Set contact and query radii.
     * @param _r Contact radius
     * @param _rq Query radius
     * @return Reference to this
     */
    PBAT_API Params& WithRadii(ScalarType _r, ScalarType _rq);
    /**
     * @brief Set scene features.
     * @param features Scene features
     * @return Reference to this
     */
    PBAT_API Params& WithSceneFeatures(ESceneFeatures features);
    /**
     * @brief Set both dynamic scene and mesh BVH build quality.
     * @param scene Build quality for dynamic scene
     * @param mesh Build quality for dynamic mesh
     * @return Reference to this
     */
    PBAT_API Params& WithDynamicBuildQuality(EBuildQuality scene, EBuildQuality mesh);
    /**
     * @brief Set both static scene and mesh BVH build quality.
     * @param scene Build quality for static scene
     * @param mesh Build quality for static mesh
     * @return Reference to this
     */
    PBAT_API Params& WithStaticBuildQuality(EBuildQuality scene, EBuildQuality mesh);
    /**
     * @brief Set maximum number of contacts.
     * @param nvf Max vertex-face contacts estimate
     * @param nfv Max face-vertex contacts estimate
     * @param nef Max edge-face contacts estimate
     * @return Reference to this
     */
    PBAT_API Params& WithMaxContactEstimates(int nvf, int nfv, int nef);
    /**
     * @brief Set displacement bound parameters.
     * @param gammap Relaxation parameter for vertex displacement bound, must satisfy `0 < gammap <
     * 0.5`
     * @param gammae Proportion of bounds-violating vertices to trigger collision detection
     * @return Reference to this
     */
    PBAT_API Params& WithDisplacementBoundConfig(ScalarType _gammap, ScalarType _gammae);
    /**
     * @brief Validate and construct the parameters.
     * @param bValidate Whether to validate parameters
     * @return Reference to this
     */
    PBAT_API Params& Construct(bool bValidate = true);
    /**
     * @brief Serialize the parameters to an archive.
     * @param archive Archive to serialize to
     */
    PBAT_API void Serialize(io::Archive& archive) const;
    /**
     * @brief Deserialize the parameters from an archive.
     * @param archive Archive to deserialize from
     */
    PBAT_API void Deserialize(io::Archive const& archive);
};

template <common::CFloatingPoint TScalar>
Params<TScalar>& Params<TScalar>::WithRadii(ScalarType _r, ScalarType _rq)
{
    r  = _r;
    rq = _rq;
    return *this;
}

template <common::CFloatingPoint TScalar>
Params<TScalar>& Params<TScalar>::WithSceneFeatures(ESceneFeatures features)
{
    eSceneFeatures = features;
    return *this;
}

template <common::CFloatingPoint TScalar>
Params<TScalar>& Params<TScalar>::WithDynamicBuildQuality(EBuildQuality scene, EBuildQuality mesh)
{
    eDynamicSceneBvhQuality = scene;
    eDynamicMeshBvhQuality  = mesh;
    return *this;
}

template <common::CFloatingPoint TScalar>
Params<TScalar>& Params<TScalar>::WithStaticBuildQuality(EBuildQuality scene, EBuildQuality mesh)
{
    eStaticSceneBvhQuality = scene;
    eStaticMeshBvhQuality  = mesh;
    return *this;
}

template <common::CFloatingPoint TScalar>
Params<TScalar>& Params<TScalar>::WithMaxContactEstimates(int nvf, int nfv, int nef)
{
    nMaxVertexFaceContactsEstimate = nvf;
    nMaxFaceVertexContactsEstimate = nfv;
    nMaxEdgeFaceContactsEstimate   = nef;
    return *this;
}

template <common::CFloatingPoint TScalar>
Params<TScalar>&
Params<TScalar>::WithDisplacementBoundConfig(ScalarType _gammap, ScalarType _gammae)
{
    gammap = _gammap;
    gammae = _gammae;
    return *this;
}

template <common::CFloatingPoint TScalar>
Params<TScalar>& Params<TScalar>::Construct(bool bValidate)
{
    if (bValidate)
    {
        if (r < Scalar(0) or rq < Scalar(0) or rq < r)
        {
            throw std::invalid_argument("Params: rq >= r >= 0 required");
        }
        if (nMaxVertexFaceContactsEstimate < 0 or nMaxFaceVertexContactsEstimate < 0 or
            nMaxEdgeFaceContactsEstimate < 0)
        {
            throw std::invalid_argument("Params: contact capacities must be non-negative");
        }
        if (gammap <= Scalar(0) or gammap >= Scalar(0.5))
        {
            throw std::invalid_argument(
                "Params: 0 < gammap < 0.5 required for displacement bound config");
        }
        if (gammae < Scalar(0) or gammae >= Scalar(1))
        {
            throw std::invalid_argument(
                "Params: 0 <= gammae < 1 required for displacement bound config");
        }
    }
    return *this;
}

template <common::CFloatingPoint TScalar>
void Params<TScalar>::Serialize(io::Archive& archive) const
{
    archive.WriteMetaData("r", r);
    archive.WriteMetaData("rq", rq);
    archive.WriteMetaData("eSceneFeatures", static_cast<int>(eSceneFeatures));
    archive.WriteMetaData("eDynamicSceneBvhQuality", static_cast<int>(eDynamicSceneBvhQuality));
    archive.WriteMetaData("eDynamicMeshBvhQuality", static_cast<int>(eDynamicMeshBvhQuality));
    archive.WriteMetaData("eStaticSceneBvhQuality", static_cast<int>(eStaticSceneBvhQuality));
    archive.WriteMetaData("eStaticMeshBvhQuality", static_cast<int>(eStaticMeshBvhQuality));
    archive.WriteMetaData("nMaxVertexFaceContactsEstimate", nMaxVertexFaceContactsEstimate);
    archive.WriteMetaData("nMaxFaceVertexContactsEstimate", nMaxFaceVertexContactsEstimate);
    archive.WriteMetaData("nMaxEdgeFaceContactsEstimate", nMaxEdgeFaceContactsEstimate);
}

template <common::CFloatingPoint TScalar>
void Params<TScalar>::Deserialize(io::Archive const& archive)
{
    if (archive.HasMetaData("r"))
        r = archive.ReadMetaData<Scalar>("r");
    if (archive.HasMetaData("rq"))
        rq = archive.ReadMetaData<Scalar>("rq");
    if (archive.HasMetaData("eSceneFeatures"))
        eSceneFeatures = static_cast<ESceneFeatures>(archive.ReadMetaData<int>("eSceneFeatures"));
    if (archive.HasMetaData("eDynamicSceneBvhQuality"))
        eDynamicSceneBvhQuality =
            static_cast<EBuildQuality>(archive.ReadMetaData<int>("eDynamicSceneBvhQuality"));
    if (archive.HasMetaData("eDynamicMeshBvhQuality"))
        eDynamicMeshBvhQuality =
            static_cast<EBuildQuality>(archive.ReadMetaData<int>("eDynamicMeshBvhQuality"));
    if (archive.HasMetaData("eStaticSceneBvhQuality"))
        eStaticSceneBvhQuality =
            static_cast<EBuildQuality>(archive.ReadMetaData<int>("eStaticSceneBvhQuality"));
    if (archive.HasMetaData("eStaticMeshBvhQuality"))
        eStaticMeshBvhQuality =
            static_cast<EBuildQuality>(archive.ReadMetaData<int>("eStaticMeshBvhQuality"));
    if (archive.HasMetaData("nMaxVertexFaceContactsEstimate"))
        nMaxVertexFaceContactsEstimate =
            archive.ReadMetaData<int>("nMaxVertexFaceContactsEstimate");
    if (archive.HasMetaData("nMaxFaceVertexContactsEstimate"))
        nMaxFaceVertexContactsEstimate =
            archive.ReadMetaData<int>("nMaxFaceVertexContactsEstimate");
    if (archive.HasMetaData("nMaxEdgeFaceContactsEstimate"))
        nMaxEdgeFaceContactsEstimate = archive.ReadMetaData<int>("nMaxEdgeFaceContactsEstimate");
}

} // namespace pbat::sim::contact::ogc

#endif // PBAT_SIM_CONTACT_OGC_PARAMS_H
