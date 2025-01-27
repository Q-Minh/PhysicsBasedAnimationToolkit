#ifndef PBAT_MATH_OPTIMIZATION_BRANCHANDBOUND_H
#define PBAT_MATH_OPTIMIZATION_BRANCHANDBOUND_H

#include <concepts>
#include <type_traits>

namespace pbat::math::optimization {

/**
 * @brief
 *
 * @tparam FLowerBound f(GpuIndex, I const&). Obtains lower bound on f(x) \forall x \in internal
 * node
 * @tparam FUpperBound f(GpuIndex, I const&). Obtains upper bound on f(x) \forall x \in internal
 * node
 * @tparam FObjective Scalar f(GpuIndex, T const&). Evaluates the objective function.
 * @tparam FIsLeaf f(GpuIndex)
 * @tparam FIsInternal f(GpuIndex)
 * @tparam FGetInternalNode I f(GpuIndex)
 * @tparam FGetLeafNodeObject T f(GpuIndex)
 * @tparam FGetChildren std::array<GpuIndex, 2>
 * @tparam T Type of image of f
 * @tparam U Leaf object type
 * @tparam I Internal node type
 */
template <
    class FLowerBound,
    class FUpperBound,
    class FObjective,
    class FIsLeaf,
    class FIsInternal,
    class FGetInternalNode,
    class FGetLeafNodeObject,
    class FGetChildren,
    class T,
    class U,
    class I>
void BranchAndBound()
{
}

} // namespace pbat::math::optimization

#endif // PBAT_MATH_OPTIMIZATION_BRANCHANDBOUND_H
