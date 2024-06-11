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
    auto& buf = detail::buffer();
    auto const size = std::min(buf.size(), name.size());
    std::memcpy(buf.data(), name.data(), size);
    buf[size - 1] = '\0';
    FrameMarkStart(buf.data());
}

void EndFrame(std::string_view name)
{
    auto& buf = detail::buffer();
    auto const size = std::min(buf.size(), name.size());
    std::memcpy(buf.data(), name.data(), std::min(buf.size(), name.size()));
    buf[size - 1] = '\0';
    FrameMarkEnd(buf.data());
}

bool IsConnectedToServer()
{
    return TracyIsConnected;
}

} // namespace profiling
} // namespace pbat