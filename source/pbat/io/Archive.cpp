#include "Archive.h"

#include "pbat/common/Overloaded.h"

namespace pbat::io {

Archive::Archive(std::filesystem::path const& filepath, HighFive::File::AccessMode flags)
    : mHdf5Object(HighFive::File(filepath.string(), flags))
{
}

Archive::Archive(Object obj) : mHdf5Object(std::move(obj)) {}

} // namespace pbat::io