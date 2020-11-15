// ---------------------------------------------------------------------
//
// Copyright (C) 2020 by the deal.II authors
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



// Test MatrixFree for simplices.

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/tria.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/simplex/fe_lib.h>
#include <deal.II/simplex/grid_generator.h>
#include <deal.II/simplex/quadrature_lib.h>

#include "../tests.h"

using namespace dealii;

template <int dim, int spacedim = dim>
void
test(const FiniteElement<dim, spacedim> &fe)
{
  Triangulation<dim, spacedim> tria;
  GridGenerator::subdivided_hyper_cube_with_simplices(tria, 1);

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  MappingFE<dim> mapping(Simplex::FE_P<dim>(1));

  MatrixFree<dim, double> matrix_free;

  AffineConstraints<double> constraints;

  Simplex::QGauss<dim> quadrature(1);

  matrix_free.reinit(mapping, dof_handler, constraints, quadrature);

  Vector<double> dst, src;

  matrix_free.initialize_dof_vector(dst);
  matrix_free.initialize_dof_vector(src);

  matrix_free.template cell_loop<Vector<double>, Vector<double>>(
    [](const auto &data, auto &dst, const auto &src, const auto cells) {
      FEEvaluation<dim, -1, 0, 1, double> phi(data);

      for (unsigned int cell = cells.first; cell < cells.second; ++cell)
        {
          phi.reinit(cell);
          phi.gather_evaluate(src, false, true);

          for (unsigned int q = 0; q < phi.n_q_points; ++q)
            phi.submit_gradient(phi.get_gradient(q), q);

          phi.integrate_scatter(false, true, dst);
        }
    },
    dst,
    src);
}

int
main(int argc, char **argv)
{
  initlog();

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  const unsigned int dim = 2;
  test<dim>(Simplex::FE_P<dim>(2));
}
