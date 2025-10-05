/**
 * @file Buffer.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief This file contains the Buffer class for 1- or 2-dimensional GPU buffers of numeric types
 * @date 2025-03-25
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#ifndef PBAT_GPU_COMMON_BUFFER_H
#define PBAT_GPU_COMMON_BUFFER_H

#include "PhysicsBasedAnimationToolkitExport.h"
#include "pbat/gpu/Aliases.h"

#include <cstddef>

namespace pbat::gpu::common {

/**
 * @brief 1- or 2-dimensional GPU buffer of numeric types
 */
class Buffer
{
  public:
    /**
     * @brief Type of the buffer elements
     */
    enum class EType { uint8, uint16, uint32, uint64, int8, int16, int32, int64, float32, float64 };
    static auto constexpr kMaxDims = 4; ///< Maximum number of dimensions

    template <class T>
    using Data =
        Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> const>; ///< Input data type
    /**
     * @brief Construct a new Buffer object of dims rows, n columns, and element type type
     *
     * @param dims Number of rows
     * @param n Number of columns
     * @param type Type of the elements
     */
    PBAT_API Buffer(GpuIndex dims, GpuIndex n, EType type);
    Buffer(const Buffer& other)            = delete;
    Buffer& operator=(const Buffer& other) = delete;
    /**
     * @brief Move constructor
     * @param other Buffer to move
     */
    PBAT_API Buffer(Buffer&& other) noexcept;
    /**
     * @brief Move assignment operator
     * @param other Buffer to move
     * @return Reference to this buffer
     */
    PBAT_API Buffer& operator=(Buffer&& other) noexcept;
    /**
     * @brief Construct a new Buffer object from input data
     * @param data Input data
     */
    PBAT_API Buffer(Data<std::uint8_t> const& data);
    /**
     * @brief Construct a new Buffer object from input data
     * @param data Input data
     */
    PBAT_API Buffer(Data<std::uint16_t> const& data);
    /**
     * @brief Construct a new Buffer object from input data
     * @param data Input data
     */
    PBAT_API Buffer(Data<std::uint32_t> const& data);
    /**
     * @brief Construct a new Buffer object from input data
     * @param data Input data
     */
    PBAT_API Buffer(Data<std::uint64_t> const& data);
    /**
     * @brief Construct a new Buffer object from input data
     * @param data Input data
     */
    PBAT_API Buffer(Data<std::int8_t> const& data);
    /**
     * @brief Construct a new Buffer object from input data
     * @param data Input data
     */
    PBAT_API Buffer(Data<std::int16_t> const& data);
    /**
     * @brief Construct a new Buffer object from input data
     * @param data Input data
     */
    PBAT_API Buffer(Data<std::int32_t> const& data);
    /**
     * @brief Construct a new Buffer object from input data
     * @param data Input data
     */
    PBAT_API Buffer(Data<std::int64_t> const& data);
    /**
     * @brief Construct a new Buffer object from input data
     * @param data Input data
     */
    PBAT_API Buffer(Data<float> const& data);
    /**
     * @brief Construct a new Buffer object from input data
     * @param data Input data
     */
    PBAT_API Buffer(Data<double> const& data);
    /**
     * @brief Copy assignment operator
     * @param data Input data to copy
     * @return Reference to this buffer
     */
    PBAT_API Buffer& operator=(Data<std::uint8_t> const& data);
    /**
     * @brief Copy assignment operator
     * @param data Input data to copy
     * @return Reference to this buffer
     */
    PBAT_API Buffer& operator=(Data<std::uint16_t> const& data);
    /**
     * @brief Copy assignment operator
     * @param data Input data to copy
     * @return Reference to this buffer
     */
    PBAT_API Buffer& operator=(Data<std::uint32_t> const& data);
    /**
     * @brief Copy assignment operator
     * @param data Input data to copy
     * @return Reference to this buffer
     */
    PBAT_API Buffer& operator=(Data<std::uint64_t> const& data);
    /**
     * @brief Copy assignment operator
     * @param data Input data to copy
     * @return Reference to this buffer
     */
    PBAT_API Buffer& operator=(Data<std::int8_t> const& data);
    /**
     * @brief Copy assignment operator
     * @param data Input data to copy
     * @return Reference to this buffer
     */
    PBAT_API Buffer& operator=(Data<std::int16_t> const& data);
    /**
     * @brief Copy assignment operator
     * @param data Input data to copy
     * @return Reference to this buffer
     */
    PBAT_API Buffer& operator=(Data<std::int32_t> const& data);
    /**
     * @brief Copy assignment operator
     * @param data Input data to copy
     * @return Reference to this buffer
     */
    PBAT_API Buffer& operator=(Data<std::int64_t> const& data);
    /**
     * @brief Copy assignment operator
     * @param data Input data to copy
     * @return Reference to this buffer
     */
    PBAT_API Buffer& operator=(Data<float> const& data);
    /**
     * @brief Copy assignment operator
     * @param data Input data to copy
     * @return Reference to this buffer
     */
    PBAT_API Buffer& operator=(Data<double> const& data);
    /**
     * @brief Get the number of dimensions (i.e. rows) of the buffer
     * @return Number of dimensions
     */
    [[maybe_unused]] GpuIndex Dims() const { return mDims; }
    /**
     * @brief Get the type of the buffer elements
     * @return Type of the buffer elements
     */
    PBAT_API EType Type() const;
    /**
     * @brief Get the number of elements per dimension in the buffer (i.e. columns)
     * @return Number of elements per dimension
     */
    PBAT_API std::size_t Size() const;
    /**
     * @brief Resize the buffer to n elements per dimension
     * @param n Number of elements per dimension
     */
    PBAT_API void Resize(GpuIndex n);
    /**
     * @brief Resize the buffer to dims rows and n columns
     * @param dims Number of rows
     * @param n Number of columns
     */
    PBAT_API void Resize(GpuIndex dims, GpuIndex n);
    /**
     * @brief Get handle to the buffer implementation
     * @return Pointer to Buffer implementation
     */
    PBAT_API void* Impl();
    /**
     * @brief Get handle to the buffer implementation
     * @return Pointer to Buffer implementation
     */
    PBAT_API void const* Impl() const;

    PBAT_API ~Buffer();

  private:
    void Deallocate();

    GpuIndex mDims;
    std::string mType;
    void* mImpl; ///< impl::common::Buffer<ValueType, Dims>*
};

} // namespace pbat::gpu::common

#endif // PBAT_GPU_COMMON_BUFFER_H
