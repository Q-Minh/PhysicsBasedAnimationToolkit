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
#include <highfive/H5File.hpp>
#include <highfive/H5Group.hpp>
#include <highfive/eigen.hpp>
#include <string>
#include <type_traits>
#include <variant>

namespace pbat::io {

/**
 * @brief Archive class for reading and writing data to HDF5 files.
 */
class Archive
{
  public:
    Archive(const Archive&)            = delete;
    Archive(Archive&&)                 = default;
    Archive& operator=(const Archive&) = delete;
    Archive& operator=(Archive&&)      = default;

    /**
     * @brief Construct a new Archive object from a filepath to an HDF5 file.
     *
     * @param filepath Path to the HDF5 file.
     * @param flags Access mode for the file (default is OpenOrCreate).
     */
    Archive(
        std::filesystem::path const& filepath,
        HighFive::File::AccessMode flags = HighFive::File::OpenOrCreate);
    /**
     * @brief Check if the archive is usable.
     * @return true if the archive is usable, false otherwise.
     */
    bool IsUsable() const;
    /**
     * @brief Get a group or create it if it does not exist.
     * @param path Path to the group.
     * @return Archive object representing the group.
     * @throw std::runtime_error if a non-group object of the same name already exists.
     * @throw HighFive::GroupException if the group cannot be created or fetched.
     */
    Archive operator[](std::string const& path);
    /**
     * @brief Get a group if it exists.
     * @param path Path to the group.
     * @return Archive object representing the group.
     * @throw HighFive::GroupException if the group does not exist.
     */
    Archive operator[](std::string const& path) const;
    /**
     * @brief Write data to the archive.
     * @tparam T Type of the data to write.
     * @param path Path to the dataset.
     * @param data Data to write.
     * @throw HighFive::DataSetException if the dataset cannot be created or written to.
     */
    template <class T>
    void WriteData(std::string const& path, T const& data);
    /**
     * @brief Write metadata to the archive.
     * @tparam T Type of the metadata to write.
     * @param key Name the attribute.
     * @param value Metadata to write.
     * @throw HighFive::AttributeException if the attribute cannot be created or written to.
     */
    template <class T>
    void WriteMetaData(std::string const& key, T const& value);
    /**
     * @brief Read data from the archive.
     * @tparam T Type of the data to read.
     * @param path Path to the dataset.
     * @return Data read from the dataset.
     * @throw HighFive::DataSetException if the dataset cannot be read.
     */
    template <class T>
    T ReadData(std::string const& path) const;
    /**
     * @brief Read metadata from the archive.
     * @tparam T Type of the metadata to read.
     * @param key Name of the attribute.
     * @return Metadata read from the attribute.
     * @throw HighFive::AttributeException if the attribute cannot be read.
     */
    template <class T>
    T ReadMetaData(std::string const& key) const;
    /**
     * @brief Unlink a dataset or group from the archive.
     * @param path Path to the dataset or group.
     * @throw HighFive::GroupException if the dataset or group cannot be unlinked.
     */
    void Unlink(std::string const& path);

    ~Archive() = default;

  protected:
    using Object =
        std::variant<std::monostate, HighFive::File, HighFive::Group>; ///< HDF5 object type

    /**
     * @brief Construct a new Archive object from an HDF5 object.
     * @param obj HDF5 object (file or group).
     */
    Archive(Object obj);

  private:
    Object mHdf5Object; ///< HDF5 object (file or group)
};

template <class T>
inline void Archive::WriteData(std::string const& path, T const& data)
{
    std::visit(
        [&](auto&& arg) {
            using U = std::decay_t<decltype(arg)>;
            if constexpr (not std::is_same_v<U, std::monostate>)
            {
                if (arg.exist(path) and (arg.getObjectType(path) == HighFive::ObjectType::Dataset))
                {
                    arg.getDataSet(path).write(data);
                }
                else
                {
                    arg.createDataSet(path, data);
                }
            }
        },
        mHdf5Object);
}

template <class T>
inline void Archive::WriteMetaData(std::string const& key, T const& value)
{
    std::visit(
        [&](auto&& arg) {
            using U = std::decay_t<decltype(arg)>;
            if constexpr (not std::is_same_v<U, std::monostate>)
            {
                if (arg.hasAttribute(key))
                {
                    arg.getAttribute(key).write(value);
                }
                else
                {
                    arg.createAttribute(key, value);
                }
            }
        },
        mHdf5Object);
}

template <class T>
inline T Archive::ReadData(std::string const& path) const
{
    T data;
    std::visit(
        [&](auto&& arg) {
            using U = std::decay_t<decltype(arg)>;
            if constexpr (not std::is_same_v<U, std::monostate>)
            {
                data = arg.getDataSet(path).read<T>();
            }
        },
        mHdf5Object);
    return std::move(data);
}

template <class T>
inline T Archive::ReadMetaData(std::string const& key) const
{
    T data;
    std::visit(
        [&](auto&& arg) {
            using U = std::decay_t<decltype(arg)>;
            if constexpr (not std::is_same_v<U, std::monostate>)
            {
                data = arg.getAttribute(key).read<T>();
            }
        },
        mHdf5Object);
    return data;
}

} // namespace pbat::io

#endif // PBAT_IO_ARCHIVE_H
