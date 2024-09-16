#ifndef PBAT_GPU_GEOMETRY_PRIMITIVES_H
#define PBAT_GPU_GEOMETRY_PRIMITIVES_H

#include "pbat/gpu/Aliases.h"

namespace pbat {
namespace gpu {
namespace geometry {

struct PointsImpl;
struct SimplicesImpl;
class BodiesImpl;

class Points
{
  public:
    /**
     * @brief
     * @param V
     */
    Points(Eigen::Ref<GpuMatrixX const> const& V);
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
    Points(Points&&) noexcept;
    /**
     * @brief
     * @param
     * @return
     */
    Points& operator=(Points&&) noexcept;
    /**
     * @brief
     * @param V
     */
    void Update(Eigen::Ref<GpuMatrixX const> const& V);
    /**
     * @brief
     * @return
     */
    PointsImpl* Impl();
    /**
     * @brief
     * @return
     */
    PointsImpl const* Impl() const;
    /**
     * @brief
     * @return
     */
    GpuMatrixX Get() const;
    /**
     * @brief
     */
    ~Points();

  private:
    PointsImpl* mImpl;
};

class Simplices
{
  public:
    enum class ESimplexType : int { Vertex = 1, Edge = 2, Triangle = 3, Tetrahedron = 4 };

    /**
     * @brief
     * @param C
     */
    Simplices(Eigen::Ref<GpuIndexMatrixX const> const& C);
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
    Simplices(Simplices&&) noexcept;
    /**
     * @brief
     * @param
     * @return
     */
    Simplices& operator=(Simplices&&) noexcept;
    /**
     * @brief
     * @return
     */
    GpuIndexMatrixX Get() const;
    /**
     * @brief
     * @return
     */
    ESimplexType Type() const;
    /**
     * @brief
     * @return
     */
    SimplicesImpl* Impl();
    /**
     * @brief
     * @return
     */
    SimplicesImpl const* Impl() const;
    /**
     * @brief
     */
    ~Simplices();

  private:
    SimplicesImpl* mImpl;
};

class Bodies
{
  public:
    /**
     * @brief
     * @param B
     */
    Bodies(Eigen::Ref<GpuIndexVectorX const> const& B);
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
    Bodies(Bodies&&) noexcept;
    /**
     * @brief
     * @param
     * @return
     */
    Bodies& operator=(Bodies&&) noexcept;
    /**
     * @brief
     * @return
     */
    GpuIndexMatrixX Get() const;
    /**
     * @brief
     * @return
     */
    std::size_t NumberOfBodies() const;
    /**
     * @brief
     * @return
     */
    BodiesImpl* Impl();
    /**
     * @brief
     * @return
     */
    BodiesImpl const* Impl() const;
    /**
     * @brief
     */
    ~Bodies();

  private:
    BodiesImpl* mImpl;
};

} // namespace geometry
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_GEOMETRY_PRIMITIVES_H