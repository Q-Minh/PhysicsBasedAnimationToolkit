#include "Var.cuh"
#include "pbat/gpu/Aliases.h"

#include <doctest/doctest.h>

TEST_CASE("[gpu][common] Var")
{
    using pbat::gpu::common::Var;
    using ValueType = pbat::GpuScalar;
    ValueType const varValueExpected = 3.f;
    Var<ValueType> var{};
    var                      = varValueExpected;
    ValueType const varValue = var.Get();
    CHECK_EQ(varValue, varValueExpected);
}