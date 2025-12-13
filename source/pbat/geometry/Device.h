/**
 * @file Device.h
 * @brief RAII wrapper over a native spatial-acceleration device (Embree RTCDevice under the hood).
 */

#ifndef PBAT_GEOMETRY_DEVICE_H
#define PBAT_GEOMETRY_DEVICE_H

#include "PhysicsBasedAnimationToolkitExport.h"
#include "pbat/io/Archive.h"

#include <string>
#include <string_view>

namespace pbat {
namespace geometry {

/**
 * @brief Configuration parameters for the device.
 */
struct DeviceConfig
{
    int threads{-1};            ///< Number of build threads (0 = all cores, -1 = default).
    int userThreads{-1};        ///< Number of user threads used to join and participate in a scene
                                ///< commit (-1 = unspecified)
    int setAffinity{-1};        ///< Pin threads to cores (0/1, -1 = unspecified)
    int startThreads{-1};       ///< Start threads at device creation (0/1, -1 = unspecified)
    std::string isa;            ///< Instruction set architecture to use (e.g. "sse2", "sse4.2",
                                ///< "avx", "avx2", "avx512", "" = default)
    std::string maxIsa;         ///< Maximum instruction set architecture to use (e.g. "sse2",
                                ///< "sse4.2", "avx", "avx2", "avx512", "" = default)
    int verbose{-1};            ///< Verbosity level (0..N, -1 = unspecified)
    std::string frequencyLevel; ///< Frequency level the application wants to run on (e.g.
                                ///< "simd128", "simd256", "simd512", "" = default)

    /**
     * @brief Set threading parameters.
     * @param nThreads Number of build threads (0 = all cores, -1 = default).
     * @param nUserThreads Number of user threads used to join and participate in a scene commit (-1
     * = unspecified).
     * @param affinity Pin threads to cores (0/1, -1 = unspecified).
     * @return Reference to this.
     */
    DeviceConfig& WithThreading(int nThreads, int nUserThreads, int affinity);
    /**
     * @brief Set instruction set parameters.
     * @param isa Instruction set architecture to use (e.g. "sse2", "sse4.2", "avx", "avx2",
     * "avx512", "" = default).
     * @param maxIsa Maximum instruction set architecture to use (e.g. "sse2", "sse4.2", "avx",
     * "avx2", "avx512", "" = default).
     * @return Reference to this.
     */
    DeviceConfig& WithInstructionSet(std::string_view isa, std::string_view maxIsa);
    /**
     * @brief Set verbosity level.
     * @param verbose Verbosity level (0..N, -1 = unspecified).
     * @return Reference to this.
     */
    DeviceConfig& WithVerbosity(int verbose);
    /**
     * @brief Set frequency level.
     * @param frequencyLevel Frequency level the application wants to run on (e.g. "simd128",
     * "simd256", "simd512", "" = default).
     * @return Reference to this.
     */
    DeviceConfig& WithFrequencyLevel(std::string_view frequencyLevel);
    /**
     * @brief Serialize the configuration to an archive.
     * @param archive The archive to serialize to.
     */
    void Serialize(io::Archive& archive) const;
    /**
     * @brief Deserialize the configuration from an archive.
     * @param archive The archive to deserialize from.
     */
    void Deserialize(io::Archive const& archive);
    /**
     * @brief Convert the configuration to a string representation.
     * @return The configuration string.
     */
    std::string ToString() const;
};

/**
 * @brief Lightweight value-type handle that manages the lifetime of a native device.
 *
 * This class is implemented against Embree's RTCDevice in the .cpp, but the header
 * does not expose Embree types. Copying retains a reference, destruction releases it.
 */
class Device
{
  public:
    using NativeHandle = void*; ///< Opaque pointer to the underlying native device

    /**
     * @brief Construct a new native device with explicit configuration parameters.
     * @throws std::runtime_error if device creation fails.
     */
    PBAT_API explicit Device(DeviceConfig const& cfg = DeviceConfig{});
    /**
     * @brief Construct a new Device from an existing native handle.
     * @param handle The native device handle to manage.
     */
    PBAT_API explicit Device(NativeHandle handle) noexcept;
    /**
     * @brief Construct a new device by copying an existing one.
     * @param other The device to copy from.
     */
    PBAT_API Device(Device const& other) noexcept;
    /**
     * @brief Construct a new device by moving an existing one.
     * @param other The device to move from.
     */
    PBAT_API Device(Device&& other) noexcept;
    /**
     * @brief
     *
     * @param other
     * @return PBAT_API&
     */
    PBAT_API Device& operator=(Device const& other) noexcept;
    /**
     * @brief
     *
     * @param other
     * @return PBAT_API&
     */
    PBAT_API Device& operator=(Device&& other) noexcept;

    /**
     * @brief Releases the managed reference if any.
     */
    PBAT_API ~Device();

    /**
     * @brief Checks if the device is valid.
     * @return true if managing a valid native device.
     * @return false if not.
     */
    explicit operator bool() const noexcept { return mHandle != nullptr; }
    /**
     * @brief Get the raw native device handle.
     * @return NativeHandle
     */
    NativeHandle Raw() const noexcept { return mHandle; }

  private:
    NativeHandle mHandle{nullptr}; ///< Opaque handle to the native device
};

} // namespace geometry
} // namespace pbat

#endif // PBAT_GEOMETRY_DEVICE_H
