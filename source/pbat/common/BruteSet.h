/**
 * @file BruteSet.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Fixed-size brute-force set implementation usable in both host and device code, suitable
 * for small sets.
 * @date 2025-02-10
 *
 * @copyright Copyright (c) 2025
 */

#ifndef PBAT_COMMON_BRUTESET_H
#define PBAT_COMMON_BRUTESET_H

#include <cassert>
#include <utility>

namespace pbat::common {

/**
 * @brief Fixed-size brute-force set implementation
 * @tparam T Type of the elements in the set
 * @param kCapacity Maximum number of elements in the set
 * @details This set is suitable for small sets where the maximum number of elements is known at
 * compile time. It uses a simple array to store the elements and provides basic operations such as
 * insertion, removal, and membership testing.
 */
template <class T, auto kCapacity = 32>
class BruteSet
{
  public:
    /**
     * @brief Construct a new BruteSet object
     */
    BruteSet() : size(0) {}
    /**
     * @brief Begin iterator
     * @return Pointer to the beginning of the set
     */
    T* begin() { return set; }
    /**
     * @brief End iterator
     * @return Pointer to the end of the set
     */
    T* end() { return set + size; }
    /**
     * @brief Const begin iterator
     * @return Pointer to the beginning of the set
     */
    T const* begin() const { return set; }
    /**
     * @brief Const end iterator
     * @return Pointer to the end of the set
     */
    T const* end() const { return set + size; }
    /**
     * @brief Insert an element into the set
     * @param value Element to insert
     * @return True if the element was inserted, false if it was already present
     * @note Does not check overflow, so ensure that the set does not exceed kCapacity before
     * inserting.
     */
    bool Insert(T const& value)
    {
        assert(size < kCapacity);
        if (not Contains(value))
        {
            set[size++] = value;
            return true;
        }
        return false;
    }
    /**
     * @brief Check if the set contains an element
     * @param value Element to check
     * @return True if the element is in the set, false otherwise
     */
    bool Contains(T const& value) const
    {
        for (int i = 0; i < size; ++i)
            if (set[i] == value)
                return true;
        return false;
    }
    /**
     * @brief Remove an element from the set
     * @param value Element to remove
     * @return True if the element was removed, false if it was not present
     */
    bool Remove(T const& value)
    {
        for (int i = 0; i < size; ++i)
        {
            if (set[i] == value)
            {
                set[i] = set[--size];
                return true;
            }
        }
        return false;
    }
    /**
     * @brief Get the size of the set
     * @return Number of elements in the set
     */
    int Size() const { return size; }
    /**
     * @brief Check if the set is full
     * @return True if the set is full, false otherwise
     */
    bool IsFull() const { return size == kCapacity; }
    /**
     * @brief Check if the set is empty
     * @return True if the set is empty, false otherwise
     */
    bool IsEmpty() const { return size == 0; }
    /**
     * @brief Clear the set
     * @details Removes all elements from the set, resetting its size to zero.
     */
    void Clear() { size = 0; }

  private:
    T set[kCapacity];
    int size;
};

} // namespace pbat::common

#endif // PBAT_COMMON_BRUTESET_H