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

// clang-format off
#include "xmmintrin.h"
#include "pmmintrin.h"
// clang-format on

namespace pbat {
namespace geometry {

static void EmbreeDeviceErrorFunction(void* userPtr, enum RTCError code, const char* str)
{
    throw std::runtime_error(
        fmt::format("Embree Device Error (code {}): {}\n", static_cast<int>(code), str));
}

DeviceConfig& DeviceConfig::WithThreading(int nThreads, int nUserThreads, int affinity)
{
    this->threads     = nThreads;
    this->userThreads = nUserThreads;
    this->setAffinity = affinity;
    return *this;
}

DeviceConfig& DeviceConfig::WithInstructionSet(std::string_view isa, std::string_view maxIsa)
{
    this->isa    = isa;
    this->maxIsa = maxIsa;
    return *this;
}

DeviceConfig& DeviceConfig::WithVerbosity(int verbose)
{
    this->verbose = verbose;
    return *this;
}

DeviceConfig& DeviceConfig::WithFrequencyLevel(std::string_view frequencyLevel)
{
    this->frequencyLevel = frequencyLevel;
    return *this;
}

void DeviceConfig::Serialize(io::Archive& archive) const
{
    auto grp = archive["pbat.geometry.DeviceConfig"];
    grp.WriteMetaData("threads", threads);
    grp.WriteMetaData("userThreads", userThreads);
    grp.WriteMetaData("setAffinity", setAffinity);
    grp.WriteMetaData("startThreads", startThreads);
    grp.WriteMetaData("isa", isa);
    grp.WriteMetaData("maxIsa", maxIsa);
    grp.WriteMetaData("verbose", verbose);
    grp.WriteMetaData("frequencyLevel", frequencyLevel);
}

void DeviceConfig::Deserialize(io::Archive const& archive)
{
    auto grp = archive["pbat.geometry.DeviceConfig"];
    if (grp.HasMetaData("threads"))
        threads = grp.ReadMetaData<int>("threads");
    if (grp.HasMetaData("userThreads"))
        userThreads = grp.ReadMetaData<int>("userThreads");
    if (grp.HasMetaData("setAffinity"))
        setAffinity = grp.ReadMetaData<int>("setAffinity");
    if (grp.HasMetaData("startThreads"))
        startThreads = grp.ReadMetaData<int>("startThreads");
    if (grp.HasMetaData("isa"))
        isa = grp.ReadMetaData<std::string>("isa");
    if (grp.HasMetaData("maxIsa"))
        maxIsa = grp.ReadMetaData<std::string>("maxIsa");
    if (grp.HasMetaData("verbose"))
        verbose = grp.ReadMetaData<int>("verbose");
    if (grp.HasMetaData("frequencyLevel"))
        frequencyLevel = grp.ReadMetaData<std::string>("frequencyLevel");
}

std::string DeviceConfig::ToString() const
{
    auto constexpr nParts = 8;
    std::vector<std::string> configParts;
    configParts.reserve(nParts);
    if (this->threads >= 0)
        configParts.push_back(fmt::format("threads={}", this->threads));
    if (this->userThreads >= 0)
        configParts.push_back(fmt::format("user_threads={}", this->userThreads));
    if (this->setAffinity >= 0)
        configParts.push_back(fmt::format("set_affinity={}", this->setAffinity));
    if (this->startThreads >= 0)
        configParts.push_back(fmt::format("start_threads={}", this->startThreads));
    if (!this->isa.empty())
        configParts.push_back(fmt::format("isa={}", this->isa));
    if (!this->maxIsa.empty())
        configParts.push_back(fmt::format("max_isa={}", this->maxIsa));
    if (this->verbose >= 0)
        configParts.push_back(fmt::format("verbose={}", this->verbose));
    if (!this->frequencyLevel.empty())
        configParts.push_back(fmt::format("frequency_level={}", this->frequencyLevel));
    std::size_t nCharacters = std::accumulate(
        configParts.begin(),
        configParts.end(),
        std::size_t{0},
        [](std::size_t sum, std::string const& part) { return sum + part.size(); });
    std::size_t nCommas = std::max(int(0), static_cast<int>(configParts.size()) - 1);
    std::string config{};
    config.reserve(nCharacters + nCommas);
    for (std::size_t i = 0; i < configParts.size(); ++i)
    {
        if (i > 0)
            config.push_back(',');
        config += configParts[i];
    }
    return config;
}

Device::Device(DeviceConfig const& cfg)
{
    // NOTE: These should be called before the creation of the tbb::task_scheduler_init object.
    // Hopefully, if it's called after, we only have a slight performance hit.
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    std::string const config = cfg.ToString();
    RTCDevice dev            = rtcNewDevice(config.c_str());
    if (!dev)
    {
        throw std::runtime_error("Failed to create spatial device");
    }
    mHandle = dev;
    rtcSetDeviceErrorFunction(static_cast<RTCDevice>(mHandle), &EmbreeDeviceErrorFunction, nullptr);
}

Device::Device(NativeHandle handle) noexcept
{
    mHandle = handle;
}

Device::Device(Device const& other) noexcept : mHandle(other.mHandle)
{
    if (mHandle)
    {
        rtcRetainDevice(static_cast<RTCDevice>(mHandle));
    }
}

Device::Device(Device&& other) noexcept : mHandle(std::exchange(other.mHandle, nullptr)) {}

Device& Device::operator=(Device const& other) noexcept
{
    if (this != &other)
    {
        if (mHandle)
        {
            rtcReleaseDevice(static_cast<RTCDevice>(mHandle));
        }
        mHandle = other.mHandle;
        if (mHandle)
        {
            rtcRetainDevice(static_cast<RTCDevice>(mHandle));
        }
    }
    return *this;
}

Device& Device::operator=(Device&& other) noexcept
{
    if (this != &other)
    {
        if (mHandle)
        {
            rtcReleaseDevice(static_cast<RTCDevice>(mHandle));
        }
        mHandle = std::exchange(other.mHandle, nullptr);
    }
    return *this;
}

Device::~Device()
{
    if (mHandle)
    {
        rtcReleaseDevice(static_cast<RTCDevice>(mHandle));
    }
}

} // namespace geometry
} // namespace pbat

#include "Device.h"

#include <doctest/doctest.h>

TEST_CASE("[geometry] Device::Config")
{
    pbat::geometry::DeviceConfig cfg;
    cfg.threads      = 4;
    cfg.userThreads  = 2;
    cfg.setAffinity  = 1;
    cfg.startThreads = 1;
    SUBCASE("Sparse config")
    {
        CHECK(cfg.ToString() == "threads=4,user_threads=2,set_affinity=1,start_threads=1");
    }
    SUBCASE("Full config")
    {
        cfg.isa            = "AVX2";
        cfg.maxIsa         = "AVX512";
        cfg.verbose        = 1;
        cfg.frequencyLevel = "high";
        CHECK(
            cfg.ToString() ==
            "threads=4,user_threads=2,set_affinity=1,start_threads=1,isa=AVX2,max_isa=AVX512,"
            "verbose=1,frequency_level=high");
    }
}