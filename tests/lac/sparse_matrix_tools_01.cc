// ---------------------------------------------------------------------
//
// Copyright (C) 2022 by the deal.II authors
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

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix_tools.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/numerics/matrix_creator.h>

#include "../tests.h"

template <int dim>
void
test()
{
  const unsigned int fe_degree = 1;

  // create mesh, ...
  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
  GridGenerator::subdivided_hyper_cube(tria, 3);

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(FE_Q<dim>(fe_degree));

  QGauss<dim> quadrature(fe_degree + 1);

  AffineConstraints<double> constraints;
  DoFTools::make_zero_boundary_constraints(dof_handler, constraints);
  constraints.close();

  // create system matrix
  TrilinosWrappers::SparsityPattern sparsity_pattern(
    dof_handler.locally_owned_dofs(), dof_handler.get_communicator());
  DoFTools::make_sparsity_pattern(dof_handler,
                                  sparsity_pattern,
                                  constraints,
                                  false);
  sparsity_pattern.compress();

  TrilinosWrappers::SparseMatrix laplace_matrix;
  laplace_matrix.reinit(sparsity_pattern);

  MatrixCreator::
    create_laplace_matrix<dim, dim, TrilinosWrappers::SparseMatrix>(
      dof_handler, quadrature, laplace_matrix, nullptr, constraints);

  // extract blocks
  std::vector<FullMatrix<double>> blocks;
  SparseMatrixTools::restrict_to_cells(laplace_matrix,
                                       sparsity_pattern,
                                       dof_handler,
                                       blocks);

  for (const auto &block : blocks)
    {
      if (block.m() == 0 && block.m() == 0)
        continue;

      block.print_formatted(deallog.get_file_stream(), 2, false, 8);
      deallog << std::endl;
    }

  const auto test_restrict = [&](const IndexSet &is_0, const IndexSet &is_1) {
    (void)is_1;
    SparsityPattern      serial_sparsity_pattern;
    SparseMatrix<double> serial_sparse_matrix;

    if (is_1.size() == 0)
      SparseMatrixTools::restrict_to_serial_sparse_matrix(
        laplace_matrix,
        sparsity_pattern,
        is_0,
        serial_sparse_matrix,
        serial_sparsity_pattern);
    else
      AssertThrow(false, ExcNotImplemented());

    FullMatrix<double> serial_sparse_matrix_full;
    serial_sparse_matrix_full.copy_from(serial_sparse_matrix);
    serial_sparse_matrix_full.print_formatted(deallog.get_file_stream(),
                                              2,
                                              false,
                                              8);
  };

  test_restrict(dof_handler.locally_owned_dofs(), {});
  test_restrict(DoFTools::extract_locally_active_dofs(dof_handler), {});
}

#include "../tests.h"

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  MPILogInitAll all;

  test<2>();
}
