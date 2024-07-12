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
#ifdef PBAT_HAS_TRACY_PROFILER
    auto& buf       = detail::buffer();
    auto const size = std::min(buf.size(), name.size());
    std::memcpy(buf.data(), name.data(), size);
    buf[size - 1] = '\0';
    FrameMarkStart(buf.data());
#endif // PBAT_HAS_TRACY_PROFILER
}

void EndFrame(std::string_view name)
{
#ifdef PBAT_HAS_TRACY_PROFILER
    auto& buf       = detail::buffer();
    auto const size = std::min(buf.size(), name.size());
    std::memcpy(buf.data(), name.data(), std::min(buf.size(), name.size()));
    buf[size - 1] = '\0';
    FrameMarkEnd(buf.data());
#endif // PBAT_HAS_TRACY_PROFILER
}

bool IsConnectedToServer()
{
#ifdef PBAT_HAS_TRACY_PROFILER
    return TracyIsConnected;
#else
    return false;
#endif // PBAT_HAS_TRACY_PROFILER
}

} // namespace profiling
} // namespace pbat