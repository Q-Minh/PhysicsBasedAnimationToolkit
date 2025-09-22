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

Archive Archive::GetOrCreateGroup(std::string const& path)
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

Archive Archive::operator[](std::string const& path)
{
    return GetOrCreateGroup(path);
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

bool Archive::HasGroup(std::string const& path) const
{
    bool bExists{false};
    std::visit(
        [&](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, HighFive::File> or std::is_same_v<T, HighFive::Group>)
            {
                bExists =
                    arg.exist(path) and (arg.getObjectType(path) == HighFive::ObjectType::Group);
            }
        },
        mHdf5Object);
    return bExists;
}

void Archive::Unlink(std::string const& path)
{
    std::visit(
        [&](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (not std::is_same_v<T, std::monostate>)
            {
                if (arg.exist(path))
                {
                    arg.unlink(path);
                }
            }
        },
        mHdf5Object);
}

std::optional<std::string> Archive::GetPath() const
{
    std::optional<std::string> path;
    std::visit(
        [&](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (not std::is_same_v<T, std::monostate>)
            {
                path = arg.getPath();
            }
        },
        mHdf5Object);
    return path;
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
        {
            Archive archive(tempFile, HighFive::File::ReadWrite);
            CHECK(archive.IsUsable());

            // Write Eigen matrix
            Eigen::MatrixXd matrix(2, 2);
            matrix << 1.1, 2.2, 3.3, 4.4;
            archive["group/eigen"].WriteData("matrix", matrix);

            // Read Eigen matrix
            auto readMatrix = archive["group/eigen"].ReadData<Eigen::MatrixXd>("matrix");
            CHECK(readMatrix.rows() == 2);
            CHECK(readMatrix.cols() == 2);
            CHECK(readMatrix(0, 0) == doctest::Approx(1.1));
            CHECK(readMatrix(0, 1) == doctest::Approx(2.2));
            CHECK(readMatrix(1, 0) == doctest::Approx(3.3));
            CHECK(readMatrix(1, 1) == doctest::Approx(4.4));

            // Write Eigen vector
            Eigen::VectorXd vector(3);
            vector << 5.5, 6.6, 7.7;
            archive["group/eigen"].WriteData("vector", vector);

            // Read Eigen vector
            auto readVector = archive["group/eigen"].ReadData<Eigen::VectorXd>("vector");
            CHECK(readVector.size() == 3);
            CHECK(readVector(0) == doctest::Approx(5.5));
            CHECK(readVector(1) == doctest::Approx(6.6));
            CHECK(readVector(2) == doctest::Approx(7.7));
        }
        std::filesystem::remove(tempFile);
    }
}