// ---------------------------------------------------------------------
//
// Copyright (C) 2020 - 2021 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/multigrid/mg_constrained_dofs.h>

#include <deal.II/numerics/vector_tools.h>

#include "../tests.h"


// Test FEFaceEvaluation::read_dof_values() and
// FEFaceEvaluation::gather_evaluate() for ECL for two cells.
//
// @note Since this program assumes that both cells are within the same
//   macro cell, this test is only run if vectorization is enabled.

template <int dim, int fe_degree, int n_points, typename Number>
void
test(const unsigned int n_refinements = 1)
{
  using VectorizedArrayType = VectorizedArray<Number>;

  using VectorType = LinearAlgebra::Vector<Number>;

  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria);

  FE_Q<dim>       fe(fe_degree);
  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  MappingQ<dim> mapping(1);
  QGauss<1>     quad(n_points);

  AffineConstraints<Number> constraint;

  using MF = MatrixFree<dim, Number, VectorizedArrayType>;

  typename MF::AdditionalData additional_data;
  additional_data.mapping_update_flags =
    update_values | update_quadrature_points;

  MF matrix_free;
  matrix_free.reinit(mapping, dof_handler, constraint, quad, additional_data);

  VectorType src, dst;

  matrix_free.initialize_dof_vector(src);
  matrix_free.initialize_dof_vector(dst);


  FEEvaluation<dim, fe_degree, n_points, 1, Number, VectorizedArrayType> phi(
    matrix_free);

  for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
    {
      std::array<unsigned int, VectorizedArrayType::size()> indices;

      indices.fill(numbers::invalid_unsigned_int);
      indices[0] = 0;

      phi.reinit(indices);

      for (unsigned int i = 0; i < phi.n_q_points; ++i)
        deallog << phi.quadrature_point(i) << std::endl;
    }
}

int
main()
{
  initlog();
  test<2, 1, 2, double>(1);
}
