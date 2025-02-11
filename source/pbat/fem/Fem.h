/**
 * @defgroup fem FEM
 */

/**
 * @file Fem.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief This file includes all the FEM related headers
 * @date 2025-02-10
 *
 * @copyright Copyright (c) 2025
 * @ingroup fem
 */

#ifndef PBAT_FEM_FEM_H
#define PBAT_FEM_FEM_H

/**
 * @brief
 */
namespace pbat::fem {
} // namespace pbat::fem

#include "Concepts.h"
#include "DeformationGradient.h"
#include "DivergenceVector.h"
#include "Gradient.h"
#include "Hexahedron.h"
#include "HyperElasticPotential.h"
#include "Jacobian.h"
#include "LaplacianMatrix.h"
#include "Line.h"
#include "LoadVector.h"
#include "MassMatrix.h"
#include "Mesh.h"
#include "QuadratureRules.h"
#include "Quadrilateral.h"
#include "ShapeFunctions.h"
#include "Tetrahedron.h"
#include "Triangle.h"

#endif // PBAT_FEM_FEM_H