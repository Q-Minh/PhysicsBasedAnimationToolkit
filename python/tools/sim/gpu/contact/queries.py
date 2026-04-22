import warp as wp


@wp.func
def closest_point_triangle(
    P: wp.vec3f, A: wp.vec3f, B: wp.vec3f, C: wp.vec3f
) -> wp.vec3f:
    """Compute the closest point on the triangle defined by (A, B, C) to the point P."""
    AB = B - A  # pyright: ignore[reportOperatorIssue]
    AC = C - A  # pyright: ignore[reportOperatorIssue]
    AP = P - A  # pyright: ignore[reportOperatorIssue]
    d1 = wp.dot(AB, AP)
    d2 = wp.dot(AC, AP)
    zero = wp.float32(0)
    one = wp.float32(1)
    if d1 <= zero and d2 <= zero:
        return wp.vec3f(one, zero, zero)  # barycentric coordinates (1,0,0)

    # Check if P in vertex region outside B
    BP = P - B  # pyright: ignore[reportOperatorIssue]
    d3 = wp.dot(AB, BP)
    d4 = wp.dot(AC, BP)
    if d3 >= zero and d4 <= d3:
        return wp.vec3f(zero, one, zero)  # barycentric coordinates (0,1,0)

    # Check if P in edge region of AB, if so return projection of P onto AB
    vc = d1 * d4 - d3 * d2
    if vc <= zero and d1 >= zero and d3 <= zero:
        v = d1 / (d1 - d3)
        return wp.vec3f(one - v, v, zero)  # barycentric coordinates (1-v,v,0)

    # Check if P in vertex region outside C
    CP = P - C  # pyright: ignore[reportOperatorIssue]
    d5 = wp.dot(AB, CP)
    d6 = wp.dot(AC, CP)
    if d6 >= zero and d5 <= d6:
        return wp.vec3f(zero, zero, one)  # barycentric coordinates (0,0,1)

    # Check if P in edge region of AC, if so return projection of P onto AC
    vb = d5 * d2 - d1 * d6
    if vb <= zero and d2 >= zero and d6 <= zero:
        w = d2 / (d2 - d6)
        return wp.vec3f(one - w, zero, w)  # barycentric coordinates (1-w,0,w)
    # Check if P in edge region of BC, if so return projection of P onto BC
    va = d3 * d6 - d5 * d4
    if va <= zero and (d4 - d3) >= zero and (d5 - d6) >= zero:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return wp.vec3f(zero, one - w, w)  # barycentric coordinates (0,1-w,w)
    # P inside face region. Compute Q through its barycentric coordinates (u,v,w)
    denom = one / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return wp.vec3f(one - v - w, v, w)  # barycentric coordinates (1-v-w, v, w)


@wp.func
def closest_points_line_segments(
    P1: wp.vec3f,
    Q1: wp.vec3f,
    P2: wp.vec3f,
    Q2: wp.vec3f,
    eps: wp.float32 = wp.float32(1e-5),  # pyright: ignore[reportArgumentType]
) -> wp.vec2f:
    zero = wp.float32(0)
    one = wp.float32(1)
    half = wp.float32(0.5)  # pyright: ignore[reportArgumentType]
    d1 = Q1 - P1  # pyright: ignore[reportOperatorIssue]
    d2 = Q2 - P2  # pyright: ignore[reportOperatorIssue]
    r = P1 - P2  # pyright: ignore[reportOperatorIssue]
    a = wp.dot(d1, d1)  # Squared length of segment S1, always nonnegative
    e = wp.dot(d2, d2)  # Squared length of segment S2, always nonnegative
    f = wp.dot(d2, r)  # Check if either or both segments degenerate into points
    eps2 = eps * eps
    if a <= eps2 and e <= eps2:
        # Both segments degenerate into points
        return wp.vec2f(half, half)  # pyright: ignore[reportArgumentType]
    elif a <= eps2:
        # First segment degenerates into a point
        return wp.vec2f(
            zero, f / e
        )  # s = 0 => t = (b*s + f) / e = f / et = Clamp(t, 0.0f, 1.0f);
    else:
        c = wp.dot(d1, r)
        if e <= eps2:
            # Second segment degenerates into a point
            # t = 0 => s = (b*t - c) / a = -c / a
            return wp.vec2f(wp.clamp(-c / a, zero, one), zero)
        else:
            # The general nondegenerate case starts here
            b = wp.dot(d1, d2)
            denom = a * e - b * b  # Always nonnegative
            # If segments not parallel, compute closest point on L1 to L2 and
            # clamp to segment S1. Else pick arbitrary s (here 0.5)
            st = wp.vec2f()
            if denom < eps * a * e:
                st[0] = half  # pyright: ignore[reportIndexIssue]
            else:
                st[0] = wp.clamp(  # pyright: ignore[reportIndexIssue]
                    (b * f - c * e) / denom, zero, one
                )
            # Compute point on L2 closest to S1(s) using
            # t = Dot((P1 + D1*s) - P2,D2) / Dot(D2,D2) = (b*s + f) / e
            st[1] = (b * st[0] + f) / e  # pyright: ignore[reportIndexIssue]

            # If t in
            # [0,1] done. Else clamp t, recompute s for the new value of t using s = Dot((P2 +
            # D2*t) - P1,D1) / Dot(D1,D1)= (t*b - c) / a and clamp s to [0, 1]
            if st[1] < zero:  # pyright: ignore[reportIndexIssue]
                st[1] = zero  # pyright: ignore[reportIndexIssue]
                st[0] = wp.clamp(-c / a, zero, one)  # pyright: ignore[reportIndexIssue]
            elif st[1] > one:  # pyright: ignore[reportIndexIssue]
                st[1] = one  # pyright: ignore[reportIndexIssue]
                st[0] = wp.clamp(  # pyright: ignore[reportIndexIssue]
                    (b - c) / a, zero, one
                )  # pyright: ignore[reportIndexIssue]
            return st
