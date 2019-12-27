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

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/hp/dof_handler.h>
#include <deal.II/hp/fe_collection.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/mg_transfer_p.h>

#include "test.h"

using namespace dealii;

template <int dim, int fe_degree_fine, int fe_degree_coarse, typename Number>
void
do_test(const FiniteElement<dim> &fe_fine, const FiniteElement<dim> &fe_coarse)
{
  AssertDimension(fe_fine.degree, fe_degree_fine);
  AssertDimension(fe_coarse.degree, fe_degree_coarse);

  // create triangulation
  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria);
  tria.refine_global();

  // data structures needed on all levels
  hp::DoFHandler<dim>   dof_handler_fine(tria);
  hp::DoFHandler<dim>   dof_handler_coarse(tria);
  hp::FECollection<dim> fe_collection;

  // set FEs on fine level
  {
    unsigned int i = 0;

    for (auto &cell : dof_handler_fine.active_cell_iterators())
      {
        cell->set_active_fe_index(i);
        fe_collection.push_back(FE_Q<dim>(i + 1));
        i++;
      }
  }

  // set FEs on coarse level
  {
    auto cell_other = dof_handler_fine.begin_active();
    for (auto &cell : dof_handler_coarse.active_cell_iterators())
      cell->set_active_fe_index(
        std::max(((cell_other++)->active_fe_index() + 1) / 2 /*bisection*/,
                 1u) -
        1);
  }

  // create dof_handler
  dof_handler_fine.distribute_dofs(fe_collection);
  dof_handler_coarse.distribute_dofs(fe_collection);

  AffineConstraints<Number> constraint_coarse;
  DoFTools::make_hanging_node_constraints(dof_handler_coarse,
                                          constraint_coarse);
  constraint_coarse.close();

  AffineConstraints<Number> constraint_fine;
  DoFTools::make_hanging_node_constraints(dof_handler_fine, constraint_fine);
  constraint_fine.close();

  // setup transfer operator
  TransferP<dim, fe_degree_fine, fe_degree_coarse, Number> transfer;
  transfer.reinit(dof_handler_fine,
                  dof_handler_coarse,
                  constraint_fine,
                  constraint_coarse);

  test_transfer_operator(transfer, dof_handler_fine, dof_handler_coarse);
}

template <int dim, int fe_degree_fine, int fe_degree_coarse, typename Number>
void
test()
{
  const auto str_fine   = std::to_string(fe_degree_fine);
  const auto str_coarse = std::to_string(fe_degree_coarse);

  {
    deallog.push("CG<2>(" + str_fine + ")<->CG<2>(" + str_coarse + ")");
    do_test<dim, fe_degree_fine, fe_degree_coarse, Number>(
      FE_Q<dim>(fe_degree_fine), FE_Q<dim>(fe_degree_coarse));
    deallog.pop();
  }

  {
    deallog.push("DG<2>(" + str_fine + ")<->CG<2>(" + str_coarse + ")");
    do_test<dim, fe_degree_fine, fe_degree_coarse, Number>(
      FE_DGQ<dim>(fe_degree_fine), FE_Q<dim>(fe_degree_coarse));
    deallog.pop();
  }

  {
    deallog.push("DG<2>(" + str_fine + ")<->DG<2>(" + str_coarse + ")");
    do_test<dim, fe_degree_fine, fe_degree_coarse, Number>(
      FE_DGQ<dim>(fe_degree_fine), FE_DGQ<dim>(fe_degree_coarse));
    deallog.pop();
  }
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  MPILogInitAll                    all;

  test<2, 1, 1, double>();
  test<2, 2, 1, double>();
}
