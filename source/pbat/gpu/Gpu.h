/**
 * @file Gpu.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief This file includes all the public GPU related headers
 * @date 2025-02-11
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_GPU_GPU_H
#define PBAT_GPU_GPU_H

/**
 * @note
 * I write a general notes here that applies to some GPU sources in the pbat/gpu directory.
 * - nvcc doesn't allow declaring extended \_\_device\_\_ lambdas in doctest's TEST_CASE clauses
 * directly, so we need to modularize test cases (and subcases) in separate functions in our GPU
 * sources in some cases.
 *
 */

/**
 * @namespace pbat::gpu 
 * @brief GPU related public functionality.
 */
namespace pbat::gpu {
} // namespace pbat::gpu

#include "common/Common.h"
#include "contact/Contact.h"
#include "geometry/Geometry.h"
#include "vbd/Vbd.h"
#include "xpbd/Xpbd.h"

#endif // PBAT_GPU_GPU_H
