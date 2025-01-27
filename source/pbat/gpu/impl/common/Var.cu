// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "Var.cuh"
#include "pbat/gpu/Aliases.h"

#include <doctest/doctest.h>

TEST_CASE("[gpu][impl][common] Var")
{
    using pbat::gpu::impl::common::Var;
    using ValueType                  = pbat::GpuScalar;
    ValueType const varValueExpected = 3.f;
    Var<ValueType> var{};
    var                      = varValueExpected;
    ValueType const varValue = var.Get();
    CHECK_EQ(varValue, varValueExpected);
}