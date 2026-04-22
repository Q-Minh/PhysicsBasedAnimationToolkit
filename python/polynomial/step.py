# type: ignore
# Smooth step approximation via cubic polynomial

import sympy as sp


def main():
    sp.init_printing(pretty_print=True)
    t, K, d = sp.symbols("t K \\delta", real=True)
    m = K - d
    tt = (t - m) / d
    # Cubic polynomial P(t) interpolating the step function { 1 if t < K, 0 if t >= K }
    # except in transition region [K-d,K]
    P = 2 * tt**3 - 3 * tt**2 + 1
    # A is antiderivative of P/t in the interval [K-d,K]
    A = sp.integrate(P / t, t)
    # In the interval (0, K-d), P = 1 -> P/t = 1/t -> antiderivative of 1/t = log(t) + C0
    C0 = A.subs(t, m) - sp.log(m)
    # In the interval t > K, P = 0 -> P/t = 0 -> antiderivative of 0 = C1
    C1 = A.subs(t, K)
    sp.pprint(A)


if __name__ == "__main__":
    main()
