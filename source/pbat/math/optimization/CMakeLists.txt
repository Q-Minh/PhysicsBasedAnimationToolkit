target_sources(
    PUBLIC
    FILE_SET api
    FILES
    "Optimization.h"
    "BranchAndBound.h"
)
target_sources(
    PRIVATE
    "BranchAndBound.cpp"
)