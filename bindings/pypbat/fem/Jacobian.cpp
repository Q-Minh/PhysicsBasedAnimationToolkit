#include "Jacobian.h"

#include "For.h"
#include "Mesh.h"

#include <exception>
#include <format>
#include <pbat/common/ConstexprFor.h>
#include <pbat/fem/Jacobian.h>
#include <pybind11/eigen.h>
#include <string>

namespace pbat {
namespace py {
namespace fem {

namespace pyb = pybind11;

void BindJacobian(pyb::module& m)
{
    ForMeshTypes([&]<class MeshType>() {
        auto constexpr kMaxQuadratureOrder = 6u;
        auto const throw_bad_quad_order    = [](int qorder) {
            std::string const what = std::format(
                "Invalid quadrature order={}, supported orders are [1,{}]",
                qorder,
                kMaxQuadratureOrder);
            throw std::invalid_argument(what);
        };
        std::string const meshTypeName             = MeshTypeName<MeshType>();
        std::string const jacobianDeterminantsName = "jacobian_determinants_" + meshTypeName;
        m.def(
            jacobianDeterminantsName.data(),
            [&](MeshType const& mesh, int qorder) -> MatrixX {
                MatrixX R;
                pbat::common::ForRange<1, kMaxQuadratureOrder + 1>([&]<auto QuadratureOrder>() {
                    if (qorder == QuadratureOrder)
                    {
                        R = pbat::fem::DeterminantOfJacobian<QuadratureOrder, MeshType>(mesh);
                    }
                });
                if (R.size() == 0)
                    throw_bad_quad_order(qorder);
                return R;
            },
            pyb::arg("mesh"),
            pyb::arg("quadrature_order"));
    });
}

} // namespace fem
} // namespace py
} // namespace pbat
