#ifndef PBAT_GPU_GEOMETRY_PRIMITIVES_H
#define PBAT_GPU_GEOMETRY_PRIMITIVES_H

#include "PhysicsBasedAnimationToolkitExport.h"
#include "pbat/gpu/Aliases.h"

namespace pbat {
namespace gpu {
namespace geometry {
namespace impl {

struct Points;
struct Simplices;
class Bodies;

} // namespace impl

class Points
{
  public:
    /**
     * @brief
     * @param V
     */
    PBAT_API Points(Eigen::Ref<GpuMatrixX const> const& V);
    /**
     * @brief
     * @param
     */
    Points(Points const&) = delete;
    /**
     * @brief
     * @param
     * @return
     */
    Points& operator=(Points const&) = delete;
    /**
     * @brief
     * @param
     */
    PBAT_API Points(Points&&) noexcept;
    /**
     * @brief
     * @param
     * @return
     */
    PBAT_API Points& operator=(Points&&) noexcept;
    /**
     * @brief
     * @param V
     */
    PBAT_API void Update(Eigen::Ref<GpuMatrixX const> const& V);
    /**
     * @brief
     * @return
     */
    PBAT_API impl::Points* Impl();
    /**
     * @brief
     * @return
     */
    PBAT_API impl::Points const* Impl() const;
    /**
     * @brief
     * @return
     */
    PBAT_API GpuMatrixX Get() const;
    /**
     * @brief
     */
    PBAT_API ~Points();

  private:
    impl::Points* mImpl;
};

class Simplices
{
  public:
    enum class ESimplexType : int { Vertex = 1, Edge = 2, Triangle = 3, Tetrahedron = 4 };

    /**
     * @brief
     * @param C
     */
    PBAT_API Simplices(Eigen::Ref<GpuIndexMatrixX const> const& C);
    /**
     * @brief
     * @param
     */
    Simplices(Simplices const&) = delete;
    /**
     * @brief
     * @param
     * @return
     */
    Simplices& operator=(Simplices const&) = delete;
    /**
     * @brief
     * @param
     */
    PBAT_API Simplices(Simplices&&) noexcept;
    /**
     * @brief
     * @param
     * @return
     */
    PBAT_API Simplices& operator=(Simplices&&) noexcept;
    /**
     * @brief
     * @return
     */
    PBAT_API GpuIndexMatrixX Get() const;
    /**
     * @brief
     * @return
     */
    PBAT_API ESimplexType Type() const;
    /**
     * @brief
     * @return
     */
    PBAT_API impl::Simplices* Impl();
    /**
     * @brief
     * @return
     */
    PBAT_API impl::Simplices const* Impl() const;
    /**
     * @brief
     */
    PBAT_API ~Simplices();

  private:
    impl::Simplices* mImpl;
};

class Bodies
{
  public:
    /**
     * @brief
     * @param B
     */
    PBAT_API Bodies(Eigen::Ref<GpuIndexVectorX const> const& B);
    /**
     * @brief
     * @param
     */
    Bodies(Bodies const&) = delete;
    /**
     * @brief
     * @param
     * @return
     */
    Bodies& operator=(Bodies const&) = delete;
    /**
     * @brief
     * @param
     */
    PBAT_API Bodies(Bodies&&) noexcept;
    /**
     * @brief
     * @param
     * @return
     */
    PBAT_API Bodies& operator=(Bodies&&) noexcept;
    /**
     * @brief
     * @return
     */
    PBAT_API GpuIndexMatrixX Get() const;
    /**
     * @brief
     * @return
     */
    PBAT_API std::size_t NumberOfBodies() const;
    /**
     * @brief
     * @return
     */
    PBAT_API impl::Bodies* Impl();
    /**
     * @brief
     * @return
     */
    PBAT_API impl::Bodies const* Impl() const;
    /**
     * @brief
     */
    PBAT_API ~Bodies();

  private:
    impl::Bodies* mImpl;
};

} // namespace geometry
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_GEOMETRY_PRIMITIVES_H
