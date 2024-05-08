import sympy as sp
from sympy.printing.cxx import CXX17CodePrinter
from sympy.codegen.ast import Assignment
from sympy.core import S


class CXXPrinter(CXX17CodePrinter):

    def __init__(self):
        from sympy.codegen.cfunctions import log2, Sqrt, log10
        from sympy.functions.elementary.exponential import log
        from sympy.functions.elementary.miscellaneous import sqrt

        # In the C89CodePrinter class, we transform the following
        # expressions into macros that major compilers typically expose.
        # However, since C++20, these numeric constants are accessible
        # through the <numbers> header, but the highest C++ standard
        # that sympy supports is C++17 through the CXX17CodePrinter.
        self.math_macros = {
            S.Exp1: "std::numbers::e_v<Scalar>",
            log2(S.Exp1): "std::numbers::log2e_v<Scalar>",
            log10(S.Exp1): "std::numbers::log10e_v<Scalar>",
            S.Pi: "std::pi_v<Scalar>",
            1/S.Pi: "std::numbers::inv_pi_v<Scalar>",
            1/sqrt(S.Pi): "std::numbers::inv_sqrtpi_v<Scalar>",
            1/Sqrt(S.Pi): "std::numbers::inv_sqrtpi_v<Scalar>",
            log(2): "std::numbers::ln2_v<Scalar>",
            log(10): "std::numbers::ln10_v<Scalar>",
            sqrt(2): "std::numbers::sqrt2_v<Scalar>",
            Sqrt(2): "std::numbers::sqrt2_v<Scalar>",
            sqrt(3): "std::numbers::sqrt3_v<Scalar>",
            Sqrt(3): "std::numbers::sqrt3_v<Scalar>",
            1/sqrt(3): "std::numbers::inv_sqrt3_v<Scalar>",
            1/Sqrt(3): "std::numbers::inv_sqrt3_v<Scalar>"
        }
        super().__init__()

    def _print_Pow(self, expr):
        if expr.exp.is_integer and expr.exp > 0 and expr.exp <= 4:
            return "*".join([self._print(expr.base) for i in range(expr.exp)])
        else:
            return super()._print_Pow(expr)


def codegen(exprs, lhs=None, use_cse=True, csesymbol="a"):
    cppgen = CXXPrinter()
    if use_cse:
        subexprs, exprs = sp.cse(
            exprs, symbols=sp.numbered_symbols(
                csesymbol)
        )
        lines = []
        for var, subexpr in subexprs:
            lines.append(
                "Scalar const " + cppgen.doprint(Assignment(var, subexpr)))
        vars = "\n".join(lines)
        outputs = cppgen.doprint(exprs if len(
            exprs) > 1 else exprs[0], assign_to=lhs)
        return "\n".join([vars, outputs]) if len(lines) > 0 else outputs

    return cppgen.doprint(exprs, assign_to=lhs)


def tabulate(code, spaces=8):
    prefix = "".join(
        [" " for _ in range(spaces)])
    code = code.split("\n")
    code = ["{}{}".format(
        prefix, statement) for statement in code]
    return "\n".join(code)


if __name__ == "__main__":
    pass
