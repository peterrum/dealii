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


/**
 *
 * Example:
 *
 * +---+---+-----+      +-----+-----+
 * | k | k |     |      |     |     |
 * |---+---|  k  |      |  k  |  k  |
 * | k | k |     |  0   |     |     |
 * +---+---+-----+  ->  +-----+-----+
 * | k | k |     |      |     |     |
 * |---+---|  k  |      |  k  |  k  |
 * | k | k |     |      |     |     |
 * +---+---+-----+      +-----+-----+
 *
 *                 ... with fe_degree in the cells
 */

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
  // create coarse grid
  Triangulation<dim> tria_coarse;
  {
    GridGenerator::hyper_cube(tria_coarse);
    tria_coarse.refine_global();
  }

  // create fine grid
  Triangulation<dim> tria_fine;
  {
    GridGenerator::hyper_cube(tria_fine);
    if (true)
      {
        tria_fine.refine_global();

        for (auto &cell : tria_fine.active_cell_iterators())
          if (cell->active() && cell->center()[0] < 0.5)
            cell->set_refine_flag();
        tria_fine.execute_coarsening_and_refinement();
      }
    else
      {
        tria_fine.refine_global(2);
      }
  }

  // setup dof-handlers
  DoFHandler<dim> dof_handler_fine(tria_fine);
  dof_handler_fine.distribute_dofs(fe_fine);

  DoFHandler<dim> dof_handler_coarse(tria_coarse);
  dof_handler_coarse.distribute_dofs(fe_coarse);

  // setup constraint matrix
  AffineConstraints<Number> constraint_coarse;
  DoFTools::make_hanging_node_constraints(dof_handler_coarse,
                                          constraint_coarse);
  constraint_coarse.close();

  AffineConstraints<Number> constraint_fine;
  DoFTools::make_hanging_node_constraints(dof_handler_fine, constraint_fine);
  constraint_coarse.close();

  // setup transfer operator
  Transfer<dim, Number> transfer;
  MGTransferUtil::setup_global_coarsening_transfer(dof_handler_fine,
                                                   dof_handler_coarse,
                                                   constraint_fine,
                                                   constraint_coarse,
                                                   transfer);

  test_transfer_operator(transfer, dof_handler_fine, dof_handler_coarse);
}

template <int dim, typename Number>
void
test(int fe_degree)
{
  const auto str_fine   = std::to_string(fe_degree);
  const auto str_coarse = std::to_string(fe_degree);

  {
    deallog.push("CG<2>(" + str_fine + ")<->CG<2>(" + str_coarse + ")");
    do_test<dim, double>(FE_Q<dim>(fe_degree), FE_Q<dim>(fe_degree));
    deallog.pop();
  }

  {
    deallog.push("DG<2>(" + str_fine + ")<->CG<2>(" + str_coarse + ")");
    do_test<dim, double>(FE_DGQ<dim>(fe_degree), FE_Q<dim>(fe_degree));
    deallog.pop();
  }

  {
    deallog.push("DG<2>(" + str_fine + ")<->DG<2>(" + str_coarse + ")");
    do_test<dim, double>(FE_DGQ<dim>(fe_degree), FE_DGQ<dim>(fe_degree));
    deallog.pop();
  }
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  MPILogInitAll                    all;

  test<2, double>(1);
  test<2, double>(2);
}
