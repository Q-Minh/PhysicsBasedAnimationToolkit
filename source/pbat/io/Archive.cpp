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
                if (arg.exist(path) and arg.getObjectType(path) == HighFive::ObjectType::Group)
                {
                    obj = arg.getGroup(path);
                }
            }
        },
        mHdf5Object);
    return Archive(std::move(obj));
}

void Archive::Unlink(std::string const& path) 
{
    std::visit(
        [&](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (not std::is_same_v<T, std::monostate>)
            {
                if (arg.exist(path)) {
                    arg.unlink(path);
                }
            }
        },
        mHdf5Object);
}

Archive::Archive(Object obj) : mHdf5Object(std::move(obj)) {}

} // namespace pbat::io

#include <doctest/doctest.h>
#include <vector>

TEST_CASE("[io] Archive")
{
    using namespace pbat::io;

    SUBCASE("Create and open archive")
    {
        std::filesystem::path tempFile = std::filesystem::temp_directory_path() / "test_archive.h5";
        {
            Archive archive(tempFile, HighFive::File::Overwrite);
            CHECK(archive.IsUsable());
        }
        {
            Archive archive(tempFile, HighFive::File::ReadOnly);
            CHECK(archive.IsUsable());
        }
        std::filesystem::remove(tempFile);
    }
    SUBCASE("Create groups and datasets")
    {
        std::filesystem::path tempFile = std::filesystem::temp_directory_path() / "test_archive.h5";
        {
            Archive archive(tempFile, HighFive::File::Overwrite);
            CHECK(archive.IsUsable());

            auto group1 = archive["group1"];
            CHECK(group1.IsUsable());

            auto group2 = group1["group2"];
            CHECK(group2.IsUsable());

            group2.WriteData("dataset1", std::vector<int>{1, 2, 3, 4, 5});
            auto data = group2.ReadData<std::vector<int>>("dataset1");
            CHECK(data == std::vector<int>({1, 2, 3, 4, 5}));
        }
        {
            Archive archive(tempFile, HighFive::File::ReadOnly);
            CHECK(archive.IsUsable());

            auto group1 = archive["group1"];
            CHECK(group1.IsUsable());

            auto group2 = group1["group2"];
            CHECK(group2.IsUsable());

            auto data = group2.ReadData<std::vector<int>>("dataset1");
            CHECK(data == std::vector<int>({1, 2, 3, 4, 5}));

            auto data2 = archive["group1"]["group2"].ReadData<std::vector<int>>("dataset1");
            CHECK(data2 == std::vector<int>({1, 2, 3, 4, 5}));

            auto data3 = archive["group1/group2"].ReadData<std::vector<int>>("dataset1");
            CHECK(data3 == std::vector<int>({1, 2, 3, 4, 5}));
        }
        {
            Archive archive(tempFile, HighFive::File::ReadWrite);
            CHECK(archive.IsUsable());
            archive["group1/group2"].WriteData("dataset2", std::vector<double>{1.1, 2.2, 3.3});
            archive["group1/group2"].Unlink("dataset1");
            archive["group1/group2"].WriteData("dataset1", std::vector<int>{6, 7, 8});
            auto data2 = archive["group1/group2"].ReadData<std::vector<double>>("dataset2");
            CHECK(data2 == std::vector<double>({1.1, 2.2, 3.3}));
            auto data1 = archive["group1/group2"].ReadData<std::vector<int>>("dataset1");
            CHECK(data1 == std::vector<int>({6, 7, 8}));
        }
        std::filesystem::remove(tempFile);
    }
}