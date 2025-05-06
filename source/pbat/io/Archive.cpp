#include "Archive.h"

#include "pbat/common/Overloaded.h"

#include <fmt/format.h>

namespace pbat::io {

Archive::Archive(std::filesystem::path const& filepath, HighFive::File::AccessMode flags)
    : mHdf5Object(HighFive::File(filepath.string(), flags))
{
}

bool Archive::IsUsable() const
{
    bool bIsUsable{false};
    std::visit(
        [&](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (not std::is_same_v<T, std::monostate>)
            {
                bIsUsable = arg.isValid();
            }
        },
        mHdf5Object);
    return bIsUsable;
}

Archive Archive::operator[](std::string const& path)
{
    Object obj;
    std::visit(
        [&](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, HighFive::File> or std::is_same_v<T, HighFive::Group>)
            {
                if (arg.exist(path))
                {
                    if (arg.getObjectType(path) == HighFive::ObjectType::Group)
                    {
                        obj = arg.getGroup(path);
                    }
                    else
                    {
                        throw std::runtime_error(
                            fmt::format(
                                "Attempted to access group {}/{} but a non-group object of the "
                                "same path already exists",
                                arg.getPath(),
                                path));
                    }
                }
                else
                {
                    obj = arg.createGroup(path);
                }
            }
        },
        mHdf5Object);
    return Archive(std::move(obj));
}

Archive Archive::operator[](std::string const& path) const
{
    Object obj;
    std::visit(
        [&](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, HighFive::File> or std::is_same_v<T, HighFive::Group>)
            {
                obj = arg.getGroup(path);
            }
        },
        mHdf5Object);
    return Archive(std::move(obj));
}

Archive::Archive(Object obj) : mHdf5Object(std::move(obj)) {}

} // namespace pbat::io