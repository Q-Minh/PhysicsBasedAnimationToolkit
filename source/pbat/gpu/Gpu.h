#ifndef PBAT_GPU_GPU_H
#define PBAT_GPU_GPU_H

/**
 * NOTE:
 * I write a general notes here that applies to some GPU sources in the pbat/gpu directory.
 *
 * - nvcc doesn't allow declaring extended __device__ lambdas in doctest's TEST_CASE clauses
 * directly, so we need to modularize test cases (and subcases) in separate functions in our GPU
 * sources in some cases.
 */

#include "common/Common.h"
#include "contact/Contact.h"
#include "geometry/Geometry.h"
#include "vbd/Vbd.h"
#include "xpbd/Xpbd.h"

#endif // PBAT_GPU_GPU_H
