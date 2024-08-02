#ifndef PBAT_GPU_GEOMETRY_PRIMITIVES_H
#define PBAT_GPU_GEOMETRY_PRIMITIVES_H

#define EIGEN_NO_CUDA
#include "pbat/Aliases.h"
#undef EIGEN_NO_CUDA

namespace pbat {
namespace gpu {
namespace geometry {

struct PointsImpl;
struct SimplicesImpl;

class Points
{
  public:
    /**
     * @brief
     * @param V
     */
    Points(Eigen::Ref<MatrixX const> const& V);
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
    void Update(Eigen::Ref<MatrixX const> const& V);
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
    MatrixX Get() const;
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
     * @param V
     */
    Simplices(Eigen::Ref<IndexMatrixX const> const& C);
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
    IndexMatrixX Get() const;
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

} // namespace geometry
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_GEOMETRY_PRIMITIVES_H