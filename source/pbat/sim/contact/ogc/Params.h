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
    using ScalarType = TScalar; ///< Scalar type

    ScalarType r{ScalarType(1e-3)};  ///< Contact radius
    ScalarType rq{ScalarType(1e-2)}; ///< Query radius

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
     * @param _gammap Relaxation parameter for vertex displacement bound, must satisfy `0 < gammap <
     * 0.5`
     * @param _gammae Proportion of bounds-violating vertices to trigger collision detection
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
        if (r < Scalar(0) or rq < Scalar(0))
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
    auto grp = archive.GetOrCreateGroup("pbat.sim.contact.ogc.Params");
    grp.WriteMetaData("r", r);
    grp.WriteMetaData("rq", rq);
    grp.WriteMetaData("gammap", gammap);
    grp.WriteMetaData("gammae", gammae);
    grp.WriteMetaData("eSceneFeatures", static_cast<int>(eSceneFeatures));
    grp.WriteMetaData("eDynamicSceneBvhQuality", static_cast<int>(eDynamicSceneBvhQuality));
    grp.WriteMetaData("eDynamicMeshBvhQuality", static_cast<int>(eDynamicMeshBvhQuality));
    grp.WriteMetaData("eStaticSceneBvhQuality", static_cast<int>(eStaticSceneBvhQuality));
    grp.WriteMetaData("eStaticMeshBvhQuality", static_cast<int>(eStaticMeshBvhQuality));
    grp.WriteMetaData("nMaxVertexFaceContactsEstimate", nMaxVertexFaceContactsEstimate);
    grp.WriteMetaData("nMaxFaceVertexContactsEstimate", nMaxFaceVertexContactsEstimate);
    grp.WriteMetaData("nMaxEdgeFaceContactsEstimate", nMaxEdgeFaceContactsEstimate);
}

template <common::CFloatingPoint TScalar>
void Params<TScalar>::Deserialize(io::Archive const& archive)
{
    auto grp = archive["pbat.sim.contact.ogc.Params"];
    if (grp.HasMetaData("r"))
        r = grp.ReadMetaData<Scalar>("r");
    if (grp.HasMetaData("rq"))
        rq = grp.ReadMetaData<Scalar>("rq");
    if (grp.HasMetaData("gammap"))
        gammap = grp.ReadMetaData<Scalar>("gammap");
    if (grp.HasMetaData("gammae"))
        gammae = grp.ReadMetaData<Scalar>("gammae");
    if (grp.HasMetaData("eSceneFeatures"))
        eSceneFeatures = static_cast<ESceneFeatures>(grp.ReadMetaData<int>("eSceneFeatures"));
    if (grp.HasMetaData("eDynamicSceneBvhQuality"))
        eDynamicSceneBvhQuality =
            static_cast<EBuildQuality>(grp.ReadMetaData<int>("eDynamicSceneBvhQuality"));
    if (grp.HasMetaData("eDynamicMeshBvhQuality"))
        eDynamicMeshBvhQuality =
            static_cast<EBuildQuality>(grp.ReadMetaData<int>("eDynamicMeshBvhQuality"));
    if (grp.HasMetaData("eStaticSceneBvhQuality"))
        eStaticSceneBvhQuality =
            static_cast<EBuildQuality>(grp.ReadMetaData<int>("eStaticSceneBvhQuality"));
    if (grp.HasMetaData("eStaticMeshBvhQuality"))
        eStaticMeshBvhQuality =
            static_cast<EBuildQuality>(grp.ReadMetaData<int>("eStaticMeshBvhQuality"));
    if (grp.HasMetaData("nMaxVertexFaceContactsEstimate"))
        nMaxVertexFaceContactsEstimate = grp.ReadMetaData<int>("nMaxVertexFaceContactsEstimate");
    if (grp.HasMetaData("nMaxFaceVertexContactsEstimate"))
        nMaxFaceVertexContactsEstimate = grp.ReadMetaData<int>("nMaxFaceVertexContactsEstimate");
    if (grp.HasMetaData("nMaxEdgeFaceContactsEstimate"))
        nMaxEdgeFaceContactsEstimate = grp.ReadMetaData<int>("nMaxEdgeFaceContactsEstimate");
}

} // namespace pbat::sim::contact::ogc

#endif // PBAT_SIM_CONTACT_OGC_PARAMS_H
