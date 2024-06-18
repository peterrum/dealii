// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2024 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------


// Test AffineConstraints::make_consistent_in_parallel() for hp constraints.
//
// Scenarios like the following for an integer k>1
//   +---+---+---+
//   |p>0|p>k|p=k|
//   |s=0|s=1|s=1|
//   +---+---+---+
// trigger an assertion on process 0 in AffineConstraints::close(), where
// - p denotes the polynomial degree for a Lagrange element Q_p, and
// - s describes the subdomain id.
// p4est partitions the above mesh as shown for two MPI processes.


#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/filtered_iterator.h>

#include <deal.II/hp/fe_collection.h>

#include <deal.II/lac/affine_constraints.h>

#include <deal.II/numerics/data_out.h>

#include "../tests.h"

#include "../test_grids.h"


template <typename Number>
IndexSet
collect_lines(const AffineConstraints<Number> &constraints,
              const unsigned int               size)
{
  IndexSet lines_local(size);
  for (const auto &line : constraints.get_lines())
    lines_local.add_index(line.index);
  return lines_local;
}


template <int dim>
void
test()
{
  const MPI_Comm mpi = MPI_COMM_WORLD;

  // ----- setup -----
  parallel::distributed::Triangulation<dim> tr(mpi);
  TestGrids::hyper_line(tr, 3);

  DoFHandler<dim> dh(tr);
  for (const auto &cell :
       dh.active_cell_iterators() | IteratorFilters::LocallyOwnedCell())
    {
      if (cell->id().to_string() == "0_0:")
        cell->set_active_fe_index(1);
      else if (cell->id().to_string() == "1_0:")
        cell->set_active_fe_index(1);
      else if (cell->id().to_string() == "2_0:")
        cell->set_active_fe_index(0);
      else
        Assert(false, ExcInternalError());
    }

  hp::FECollection<dim> fes;
  for (unsigned int p = 2; p <= 3; ++p)
    fes.push_back(FE_Q<dim>(p));
  dh.distribute_dofs(fes);

  // ----- output -----
  // Kept for future debugging attempts.
#if 0
  Vector<float> fe_degrees(tr.n_active_cells());
  for (const auto &cell :
       dh.active_cell_iterators() | IteratorFilters::LocallyOwnedCell())
    fe_degrees(cell->active_cell_index()) = cell->get_fe().degree;

  Vector<float> subdomain(tr.n_active_cells());
  for (auto &subd : subdomain)
    subd = tr.locally_owned_subdomain();

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dh);
  data_out.add_data_vector(fe_degrees, "fe_degree");
  data_out.add_data_vector(subdomain, "subdomain");
  data_out.build_patches();
  data_out.write_vtu_in_parallel("output-" + Utilities::to_string(dim) +
                                   "d.vtu",
                                 mpi);
#endif

  // ----- constraints -----
  const auto &owned_dofs    = dh.locally_owned_dofs();
  const auto  relevant_dofs = DoFTools::extract_locally_relevant_dofs(dh);

  {
    AffineConstraints<double> ac(owned_dofs, relevant_dofs);
    DoFTools::make_hanging_node_constraints(dh, ac);

    const auto before = collect_lines(ac, dh.n_dofs());
    before.print(deallog.get_file_stream());

    ac.make_consistent_in_parallel(owned_dofs, relevant_dofs, mpi);

    const auto after = collect_lines(ac, dh.n_dofs());
    after.print(deallog.get_file_stream());

    ac.close();
  }

  deallog << "OK" << std::endl;
}


int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  MPILogInitAll                    all;

  deallog.push("2d");
  test<2>();
  deallog.pop();
  deallog.push("3d");
  test<3>();
  deallog.pop();
}
