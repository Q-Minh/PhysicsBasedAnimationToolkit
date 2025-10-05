#include "Sdf.h"

#include "BinaryNode.h"
#include "Composite.h"
#include "Primitive.h"
#include "Transform.h"
#include "UnaryNode.h"

namespace pbat::py::geometry::sdf {

void Bind(nanobind::module_& m)
{
    BindPrimitive(m);
    BindTransform(m);
    BindUnaryNode(m);
    BindBinaryNode(m);
    BindComposite(m);
}

} // namespace pbat::py::geometry::sdf