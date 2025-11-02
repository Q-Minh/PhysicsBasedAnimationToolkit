/**
 * @file Device.cpp
 */

#include "pbat/geometry/Device.h"

#include <embree4/rtcore.h>
#include <fmt/core.h>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
// Helper to convert opaque handle to Embree RTCDevice
inline RTCDevice toRtc(pbat::geometry::Device::NativeHandle h) noexcept
{
    return static_cast<RTCDevice>(h);
}
} // namespace

namespace pbat {
namespace geometry {

Device::Device(Config const& cfg)
{
    // Build configuration string
    auto constexpr nParts = 8;
    std::vector<std::string> configParts;
    configParts.reserve(nParts);
    if (cfg.threads >= 0)
        configParts.push_back(fmt::format("threads={}", cfg.threads));
    if (cfg.userThreads >= 0)
        configParts.push_back(fmt::format("user_threads={}", cfg.userThreads));
    if (cfg.setAffinity >= 0)
        configParts.push_back(fmt::format("set_affinity={}", cfg.setAffinity));
    if (cfg.startThreads >= 0)
        configParts.push_back(fmt::format("start_threads={}", cfg.startThreads));
    if (!cfg.isa.empty())
        configParts.push_back(fmt::format("isa={}", cfg.isa));
    if (!cfg.maxIsa.empty())
        configParts.push_back(fmt::format("max_isa={}", cfg.maxIsa));
    if (cfg.verbose >= 0)
        configParts.push_back(fmt::format("verbose={}", cfg.verbose));
    if (!cfg.frequencyLevel.empty())
        configParts.push_back(fmt::format("frequency_level={}", cfg.frequencyLevel));
    std::size_t nCharacters = std::accumulate(
        configParts.begin(),
        configParts.end(),
        std::size_t{0},
        [](std::size_t sum, std::string const& part) { return sum + part.size(); });
    std::size_t nCommas = std::max(std::size_t{0}, configParts.size() - 1);
    std::string config{};
    config.reserve(nCharacters + nCommas);
    for (std::size_t i = 0; i < configParts.size(); ++i)
    {
        if (i > 0)
            config.push_back(',');
        config += configParts[i];
    }
    // Create the device
    RTCDevice dev = rtcNewDevice(config.c_str());
    if (!dev)
    {
        throw std::runtime_error("Failed to create spatial device");
    }
    mHandle = dev;
}

Device::Device(Device const& other) noexcept : mHandle(other.mHandle)
{
    if (mHandle)
    {
        rtcRetainDevice(toRtc(mHandle));
    }
}

Device::Device(Device&& other) noexcept : mHandle(std::exchange(other.mHandle, nullptr)) {}

Device::~Device()
{
    if (mHandle)
    {
        rtcReleaseDevice(toRtc(mHandle));
    }
}

} // namespace geometry
} // namespace pbat
