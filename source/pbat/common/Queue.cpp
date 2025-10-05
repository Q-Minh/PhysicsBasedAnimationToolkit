#include "Queue.h"

#include <doctest/doctest.h>

TEST_CASE("[common] Queue")
{
    pbat::common::Queue<int> queue;

    SUBCASE("Queue is initially empty")
    {
        CHECK(queue.IsEmpty());
        CHECK_EQ(queue.Size(), 0);
    }

    SUBCASE("Push elements")
    {
        queue.Push(1);
        queue.Push(2);
        queue.Push(3);

        CHECK_FALSE(queue.IsEmpty());
        CHECK_EQ(queue.Size(), 3);
    }

    SUBCASE("Pop elements")
    {
        queue.Push(1);
        queue.Push(2);
        queue.Push(3);

        CHECK_EQ(queue.Top(), 1);
        queue.Pop();
        CHECK_EQ(queue.Top(), 2);
        queue.Pop();
        CHECK_EQ(queue.Top(), 3);
        queue.Pop();
        CHECK(queue.IsEmpty());
    }

    SUBCASE("Peek at the front element")
    {
        queue.Push(42);
        CHECK_EQ(queue.Top(), 42);
        CHECK_FALSE(queue.IsEmpty());
    }
}