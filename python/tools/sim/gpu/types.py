import warp as wp

vec3i = wp.types.vector(length=3, dtype=wp.int32)
vec4i = wp.types.vector(length=4, dtype=wp.int32)
mat43f = wp.types.matrix(shape=(4, 3), dtype=wp.float32)
mat99f = wp.types.matrix(shape=(9, 9), dtype=wp.float32)
vec9f = wp.types.vector(length=9, dtype=wp.float32)
