#include "Profiling.h"

#include <array>
#include <cstring>

namespace pbat {
namespace profiling {
namespace detail {

std::array<char, 256>& buffer()
{
    static std::array<char, 256> buf{};
    return buf;
}

} // namespace detail

void BeginFrame(std::string_view name)
{
#if defined(PBAT_HAS_TRACY_PROFILER)
    auto& buf       = detail::buffer();
    auto const size = std::min(buf.size() - 1, name.size());
    std::memcpy(buf.data(), name.data(), size);
    buf[size] = '\0';
    FrameMarkStart(buf.data());
#endif // PBAT_HAS_TRACY_PROFILER
}

void EndFrame(std::string_view name)
{
#if defined(PBAT_HAS_TRACY_PROFILER)
    auto& buf       = detail::buffer();
    auto const size = std::min(buf.size() - 1, name.size());
    std::memcpy(buf.data(), name.data(), std::min(buf.size(), name.size()));
    buf[size] = '\0';
    FrameMarkEnd(buf.data());
#endif // PBAT_HAS_TRACY_PROFILER
}

bool IsConnectedToServer()
{
#if defined(PBAT_CAN_USE_TRACY_CPP)
    return TracyIsConnected;
#else
    return false;
#endif // defined(PBAT_CAN_USE_TRACY_CPP)
}

} // namespace profiling
} // namespace pbat