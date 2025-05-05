/**
 * @file Archive.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief (De)serializer
 * @date 2025-05-05
 * @copyright Copyright (c) 2025
 */

#ifndef PBAT_IO_ARCHIVE_H
#define PBAT_IO_ARCHIVE_H

#include <filesystem>
#include <highfive/H5Attribute.hpp>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5File.hpp>
#include <highfive/H5Group.hpp>
#include <highfive/eigen.hpp>
#include <string>
#include <variant>

namespace pbat::io {

class Archive
{
  public:
    Archive()                          = default;
    Archive(const Archive&)            = delete;
    Archive(Archive&&)                 = default;
    Archive& operator=(const Archive&) = delete;
    Archive& operator=(Archive&&)      = default;

    Archive(
        std::filesystem::path const& filepath,
        HighFive::File::AccessMode flags = HighFive::File::ReadWrite);

    Archive Attribute(std::string const& name) const;
    Archive Group(std::string const& name) const;
    Archive DataSet(std::string const& name) const;

    template <class T>
    void Write(T const& data);

    template <class T>
    T Read() const;

    ~Archive() = default;

  protected:
    using Object = std::variant<
        std::monostate,
        HighFive::File,
        HighFive::Group,
        HighFive::DataSet,
        HighFive::Attribute>;

    Archive(Object obj);

  private:
    Object mHdf5Object;
};

} // namespace pbat::io

#endif // PBAT_IO_ARCHIVE_H
