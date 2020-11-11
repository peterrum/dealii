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
}

int
main(int argc, char **argv)
{
  initlog();

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  const unsigned int dim = 2;
  test<dim>(Simplex::FE_P<dim>(2));
}
