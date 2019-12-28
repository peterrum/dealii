// ---------------------------------------------------------------------
//
// Copyright (C) 2019 by the deal.II authors
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


// Test p-transfer.

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/mg_transfer_interface_util.h>

#include "test.h"

using namespace dealii;

template <int dim, typename Number>
void
do_test(const FiniteElement<dim> &fe_fine, const FiniteElement<dim> &fe_coarse)
{
  parallel::distributed::Triangulation<dim> tria(
    MPI_COMM_WORLD,
    Triangulation<dim>::limit_level_difference_at_vertices,
    parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy);

  // create grid
  GridGenerator::hyper_cube(tria);
  tria.refine_global();

  for (auto &cell : tria.active_cell_iterators())
    if (cell->active() && cell->center()[0] < 0.5)
      cell->set_refine_flag();
  tria.execute_coarsening_and_refinement();

  // setup dof-handlers
  DoFHandler<dim> dof_handler_fine(tria);
  dof_handler_fine.distribute_dofs(fe_fine);

  DoFHandler<dim> dof_handler_coarse(tria);
  dof_handler_coarse.distribute_dofs(fe_coarse);

  AffineConstraints<Number> constraint_coarse;
  DoFTools::make_hanging_node_constraints(dof_handler_coarse,
                                          constraint_coarse);
  constraint_coarse.close();

  AffineConstraints<Number> constraint_fine;
  DoFTools::make_hanging_node_constraints(dof_handler_fine, constraint_fine);
  constraint_fine.close();

  // setup transfer operator
  Transfer<dim, Number> transfer;
  MGTransferUtil::setup_polynomial_transfer(dof_handler_fine,
                                            dof_handler_coarse,
                                            constraint_fine,
                                            constraint_coarse,
                                            transfer);

  test_transfer_operator(transfer, dof_handler_fine, dof_handler_coarse);
}

template <int dim, typename Number>
void
test(int fe_degree_fine, int fe_degree_coarse)
{
  const auto str_fine   = std::to_string(fe_degree_fine);
  const auto str_coarse = std::to_string(fe_degree_coarse);

  {
    deallog.push("CG<2>(" + str_fine + ")<->CG<2>(" + str_coarse + ")");
    do_test<dim, Number>(FE_Q<dim>(fe_degree_fine),
                         FE_Q<dim>(fe_degree_coarse));
    deallog.pop();
  }

  {
    deallog.push("DG<2>(" + str_fine + ")<->CG<2>(" + str_coarse + ")");
    do_test<dim, Number>(FE_DGQ<dim>(fe_degree_fine),
                         FE_Q<dim>(fe_degree_coarse));
    deallog.pop();
  }

  {
    deallog.push("DG<2>(" + str_fine + ")<->DG<2>(" + str_coarse + ")");
    do_test<dim, Number>(FE_DGQ<dim>(fe_degree_fine),
                         FE_DGQ<dim>(fe_degree_coarse));
    deallog.pop();
  }
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  MPILogInitAll                    all;

  test<2, double>(1, 1);
  test<2, double>(2, 1);
}
