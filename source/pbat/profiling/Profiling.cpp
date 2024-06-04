#include "Profiling.h"

#include <array>
#include <cstring>

namespace pbat {
namespace profiling {
namespace detail {

std::array<char, 64>& buffer()
{
    static std::array<char, 64> buf{};
    return buf;
}

} // namespace detail

void BeginFrame(std::string_view name)
{
    auto& buf = detail::buffer();
    std::memcpy(buf.data(), name.data(), std::min(buf.size(), name.size()));
    FrameMarkStart(buf.data());
}

void EndFrame(std::string_view name)
{
    auto& buf = detail::buffer();
    std::memcpy(buf.data(), name.data(), std::min(buf.size(), name.size()));
    FrameMarkEnd(buf.data());
}

bool IsConnectedToServer()
{
    return TracyIsConnected;
}

} // namespace profiling
} // namespace pbat