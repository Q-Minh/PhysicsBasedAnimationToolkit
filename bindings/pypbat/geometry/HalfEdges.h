/**
 * @file HalfEdges.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Python bindings for half-edge style adjacency on triangle meshes.
 * @version 0.1
 * @date 2025-11-06
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#ifndef PYPBAT_GEOMETRY_HALFEDGES_H
#define PYPBAT_GEOMETRY_HALFEDGES_H

#include <nanobind/nanobind.h>

namespace pbat::py::geometry {

void BindHalfEdges(nanobind::module_& m);

} // namespace pbat::py::geometry

#endif // PYPBAT_GEOMETRY_HALFEDGES_H
