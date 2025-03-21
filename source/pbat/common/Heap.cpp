#include "Heap.h"

#include <doctest/doctest.h>

TEST_CASE("[common] Heap")
{
    pbat::common::Heap<int> heap;

    SUBCASE("Insert elements into the heap")
    {
        heap.Push(10);
        heap.Push(20);
        heap.Push(5);

        CHECK_EQ(heap.Size(), 3);
        CHECK_EQ(heap.Top(), 20);
    }

    SUBCASE("Extract the top element")
    {
        heap.Push(15);
        heap.Push(30);
        heap.Push(10);

        CHECK_EQ(heap.Pop(), 30);
        CHECK_EQ(heap.Size(), 2);
        CHECK_EQ(heap.Top(), 15);
    }

    SUBCASE("Heap is empty after all elements are extracted")
    {
        heap.Push(25);
        heap.Push(35);
        heap.Pop();
        heap.Pop();
        CHECK(heap.IsEmpty());
    }

    SUBCASE("Heap property is maintained after multiple operations")
    {
        heap.Push(40);
        heap.Push(50);
        heap.Push(30);
        heap.Push(20);

        CHECK_EQ(heap.Pop(), 50);
        CHECK_EQ(heap.Pop(), 40);
        CHECK_EQ(heap.Pop(), 30);
        CHECK_EQ(heap.Pop(), 20);
        CHECK(heap.IsEmpty());
    }
}