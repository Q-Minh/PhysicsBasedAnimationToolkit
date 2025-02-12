/**
 * @file DoctestLoadDLL.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief No-op function to force load PhysicsBasedAnimationToolkit DLL
 * @date 2025-02-11
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_DOCTEST_LOAD_DLL_H
#define PBAT_DOCTEST_LOAD_DLL_H

#include "PhysicsBasedAnimationToolkitExport.h"

namespace pbat {

/**
 * @brief No-op function to force load PhysicsBasedAnimationToolkit DLL
 */
PBAT_API void ForceLoadDLL();

} // namespace pbat

#endif // PBAT_DOCTEST_LOAD_DLL_H