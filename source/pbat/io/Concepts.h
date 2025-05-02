/**
 * @file Concepts.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Header file for the IO module's concepts.
 * @date 2025-05-02
 * @copyright Copyright (c) 2025
 */

#ifndef PBAT_IO_CONCEPTS_H
#define PBAT_IO_CONCEPTS_H

#include <concepts>
#include <highfive/H5Object.hpp>
#include <highfive/bits/H5Annotate_traits.hpp>
#include <highfive/bits/H5Node_traits.hpp>

namespace pbat::io {

/**
 * @brief Concept for an HDF5 group.
 *
 * [HighFive](https://highfive-devs.github.io/highfive/index.html) makes a distinction between a
 * [`HighFive::File`](https://highfive-devs.github.io/highfive/class_high_five_1_1_file.html) and a
 * [`HighFive::Group`](https://highfive-devs.github.io/highfive/class_high_five_1_1_group.html), but
 * a file is really the root group of the HDF5 file. So, we can use the same concept for both.
 * @tparam T The type to check.
 */
template <class T>
concept CGroup =
    std::derived_from<T, HighFive::NodeTraits<T>> and
    std::derived_from<T, HighFive::AnnotateTraits<T>> and std::derived_from<T, HighFive::Object>;

} // namespace pbat::io

#endif // PBAT_IO_CONCEPTS_H
