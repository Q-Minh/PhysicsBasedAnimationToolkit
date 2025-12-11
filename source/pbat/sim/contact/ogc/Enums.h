/**
 * @file Enums.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Enumerations for Offset Geometry Contact (OGC) algorithm.
 * @version 0.1
 * @date 2025-12-10
 * @copyright Copyright (c) 2025
 */
#ifndef PBAT_SIM_CONTACT_OGC_ENUMS_H
#define PBAT_SIM_CONTACT_OGC_ENUMS_H

namespace pbat::sim::contact::ogc {

/**
 * @brief BVH build quality options. See enum RTCBuildQuality in rtcore_common.h of embree.
 */
enum class EBuildQuality {
    Low    = 0, ///< Low build quality (fast build time)
    Medium = 1, ///< Medium build quality (balanced)
    High   = 2, ///< High build quality (slow build time)
    Refit  = 3  ///< Refit
};

/**
 * @brief Scene construction features. See enum RTCSceneFlags in rtcore_scene.h of embree.
 */
enum class ESceneFeatures {
    None    = 0,        ///< No special features
    Dynamic = (1 << 0), ///< Dynamic scene
    Compact = (1 << 1), ///< Compact representation
    Robust  = (1 << 2)  ///< Robust representation
};

} // namespace pbat::sim::contact::ogc

#endif // PBAT_SIM_CONTACT_OGC_ENUMS_H
