/**
 * @file Concepts.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Header file for the IO module's concepts.
 * @date 2025-05-02
 * @copyright Copyright (c) 2025
 */

#ifndef PBAT_IO_CONCEPTS_H
#define PBAT_IO_CONCEPTS_H

#include "Archive.h"

#include <concepts>
#include <utility>

namespace pbat::io {

/**
 * @brief Concept for a serializable object.
 * @tparam T The type to check.
 */
template <class T>
concept CSerializable = requires(T t)
{
    {t.Serialize(std::declval<Archive&>())};
};

/**
 * @brief Concept for a deserializable object.
 * @tparam T The type to check.
 */
template <class T>
concept CDeserializable = requires(T t)
{
    {t.Deserialize(std::declval<Archive const&>())};
};

} // namespace pbat::io

#endif // PBAT_IO_CONCEPTS_H
