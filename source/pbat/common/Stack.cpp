#include "Stack.h"

#include <doctest/doctest.h>

TEST_CASE("[common] Stack")
{
    pbat::common::Stack<int> stack;

    SUBCASE("New stack is empty")
    {
        CHECK(stack.IsEmpty());
        CHECK_EQ(stack.Size(), 0);
    }

    SUBCASE("Push elements onto the stack")
    {
        stack.Push(10);
        stack.Push(20);
        stack.Push(30);

        CHECK_FALSE(stack.IsEmpty());
        CHECK_EQ(stack.Size(), 3);
    }

    SUBCASE("Pop elements from the stack")
    {
        stack.Push(10);
        stack.Push(20);
        stack.Push(30);

        CHECK_EQ(stack.Pop(), 30);
        CHECK_EQ(stack.Pop(), 20);
        CHECK_EQ(stack.Pop(), 10);
        CHECK(stack.IsEmpty());
    }

    SUBCASE("Peek at the top element")
    {
        stack.Push(10);
        stack.Push(20);

        CHECK_EQ(stack.Top(), 20);
        CHECK_EQ(stack.Size(), 2);
    }
}