/**
 * @file BinaryNode.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief This file defines binary nodes for a SDF compositions
 * @version 0.1
 * @date 2025-09-16
 * @details Credits go to https://iquilezles.org/articles/distfunctions/, thank you Inigo Quilez!
 * @copyright Copyright (c) 2025
 *
 */
#ifndef PBAT_GEOMETRY_SDF_BINARYNODE_H
#define PBAT_GEOMETRY_SDF_BINARYNODE_H

#include "pbat/HostDevice.h"
#include "pbat/common/Concepts.h"

#include <algorithm>
#include <cmath>

namespace pbat::geometry::sdf {

/**
 * @brief Base struct for all binary nodes
 */
struct BinaryNode
{
};

/**
 * @brief Union operation
 * @tparam TScalar Scalar type
 */
template <common::CArithmetic TScalar>
struct Union : public BinaryNode
{
    using ScalarType = TScalar; ///< Scalar type
    /**
     * @brief Evaluate the signed distance function of the union of two shapes
     * @note The union (as minimum) of 2 SDFs is not necessarily an SDF, especially as we go deeper
     * in the interior of the union.
     * @param sd1 Signed distance to the first shape
     * @param sd2 Signed distance to the second shape
     * @return Signed distance to the union of two shapes
     */
    PBAT_HOST_DEVICE ScalarType Eval(ScalarType sd1, ScalarType sd2) const
    {
        using namespace std;
        return min(sd1, sd2);
    }
};

/**
 * @brief Difference operation
 * @tparam TScalar Scalar type
 */
template <common::CArithmetic TScalar>
struct Difference : public BinaryNode
{
    using ScalarType = TScalar; ///< Scalar type
    /**
     * @brief Evaluate the signed distance function of the difference of two shapes
     * @param sd1 Signed distance to the first shape
     * @param sd2 Signed distance to the second shape
     * @return Signed distance to the difference of two shapes
     */
    PBAT_HOST_DEVICE ScalarType Eval(ScalarType sd1, ScalarType sd2) const
    {
        using namespace std;
        return max(-sd1, sd2);
    }
};

/**
 * @brief Intersection operation
 * @tparam TScalar Scalar type
 */
template <common::CArithmetic TScalar>
struct Intersection : public BinaryNode
{
    using ScalarType = TScalar; ///< Scalar type
    /**
     * @brief Evaluate the signed distance function of the intersection of two shapes
     * @param sd1 Signed distance to the first shape
     * @param sd2 Signed distance to the second shape
     * @return Signed distance to the intersection of two shapes
     */
    PBAT_HOST_DEVICE ScalarType Eval(ScalarType sd1, ScalarType sd2) const
    {
        using namespace std;
        return max(sd1, sd2);
    }
};

/**
 * @brief Exclusive or operation
 * @tparam TScalar Scalar type
 */
template <common::CArithmetic TScalar>
struct ExclusiveOr : public BinaryNode
{
    using ScalarType = TScalar; ///< Scalar type
    /**
     * @brief Evaluate the signed distance function of the exclusive or of two shapes
     * @param sd1 Signed distance to the first shape
     * @param sd2 Signed distance to the second shape
     * @return Signed distance to the exclusive or of two shapes
     */
    PBAT_HOST_DEVICE ScalarType Eval(ScalarType sd1, ScalarType sd2) const
    {
        using namespace std;
        return max(min(sd1, sd2), -max(sd1, sd2));
    }
};

/**
 * @brief Smooth union operation
 * @tparam TScalar Scalar type
 */
template <common::CArithmetic TScalar>
struct SmoothUnion : public BinaryNode
{
    using ScalarType = TScalar; ///< Scalar type
    SmoothUnion()    = default;
    /**
     * @brief Construct a new Smooth Union object
     * @param k_ Smoothness factor
     */
    explicit SmoothUnion(ScalarType k_) : k(k_) {}
    ScalarType k; ///< Smoothness factor
    /**
     * @brief Evaluate the signed distance function of the smooth union of two shapes
     * @param sd1 Signed distance to the first shape
     * @param sd2 Signed distance to the second shape
     * @return Signed distance to the smooth union of two shapes
     */
    PBAT_HOST_DEVICE ScalarType Eval(ScalarType sd1, ScalarType sd2) const
    {
        using namespace std;
        ScalarType kk = 4 * k;
        ScalarType h  = max(kk - abs(sd1 - sd2), ScalarType(0));
        return min(sd1, sd2) - h * h * ScalarType(0.25) / kk;
    }
};

/**
 * @brief Smooth difference operation
 * @tparam TScalar Scalar type
 */
template <common::CArithmetic TScalar>
struct SmoothDifference : public BinaryNode
{
    using ScalarType   = TScalar; ///< Scalar type
    SmoothDifference() = default;
    /**
     * @brief Construct a new Smooth Difference object
     * @param k_ Smoothness factor
     */
    explicit SmoothDifference(ScalarType k_) : k(k_) {}
    ScalarType k; ///< Smoothness factor
    /**
     * @brief Evaluate the signed distance function of the smooth difference of two shapes
     * @param sd1 Signed distance to the first shape
     * @param sd2 Signed distance to the second shape
     * @return Signed distance to the smooth difference of two shapes
     */
    PBAT_HOST_DEVICE ScalarType Eval(ScalarType sd1, ScalarType sd2) const
    {
        SmoothUnion<TScalar> U{};
        U.k = k;
        return -U.Eval(sd1, -sd2);
    }
};

/**
 * @brief Smooth intersection operation
 * @tparam TScalar Scalar type
 */
template <common::CArithmetic TScalar>
struct SmoothIntersection : public BinaryNode
{
    using ScalarType     = TScalar; ///< Scalar type
    SmoothIntersection() = default;
    /**
     * @brief Construct a new Smooth Intersection object
     * @param k_ Smoothness factor
     */
    explicit SmoothIntersection(ScalarType k_) : k(k_) {}
    ScalarType k; ///< Smoothness factor
    /**
     * @brief Evaluate the signed distance function of the smooth intersection of two shapes
     * @param sd1 Signed distance to the first shape
     * @param sd2 Signed distance to the second shape
     * @return Signed distance to the smooth intersection of two shapes
     */
    PBAT_HOST_DEVICE ScalarType Eval(ScalarType sd1, ScalarType sd2) const
    {
        SmoothUnion<TScalar> U{};
        U.k = k;
        return -U.Eval(-sd1, -sd2);
    }
};

} // namespace pbat::geometry::sdf

#endif // PBAT_GEOMETRY_SDF_BINARYNODE_H
