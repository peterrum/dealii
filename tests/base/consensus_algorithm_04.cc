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


// Test ConsensusAlgorithms::AnonymousProcess.

#include <deal.II/base/mpi_compute_index_owner_internal.h>
#include <deal.II/base/mpi_consensus_algorithms.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <set>

#include "consensus_algorithm_util.h"

template <int dim, int spacedim>
void
test(const MPI_Comm &comm)
{
  Triangulation<dim> basetria;
  GridGenerator::subdivided_hyper_cube(basetria, 4);
  basetria.refine_global(2);

  const auto deterimine_n_coarse_cells = [&comm](auto &tria) {
    types::coarse_cell_id n_coarse_cells = 0;

    for (auto cell : tria.active_cell_iterators())
      if (!cell->is_artificial())
        n_coarse_cells =
          std::max(n_coarse_cells, cell->id().get_coarse_cell_id());

    return Utilities::MPI::max(n_coarse_cells, comm) + 1;
  };

  // create translator: CellID <-> unique ID
  CellIDTranslator<dim> cell_id_translator(deterimine_n_coarse_cells(basetria),
                                           basetria.n_global_levels());


  if (false)
    for (unsigned int i = 0; i < basetria.n_global_levels(); ++i)
      for (auto cell : basetria.cell_iterators_on_level(i))
        {
          const auto id = cell_id_translator.translate(cell);
          deallog << cell->id() << " - " << id << std::endl;
        }

  for (auto cell : basetria.cell_iterators())
    {
      Assert(cell->id() == cell_id_translator.to_cell_id(
                             cell_id_translator.translate(cell)),
             ExcNotImplemented());
    }
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  MPILogInitAll                    all;

  const MPI_Comm comm = MPI_COMM_WORLD;

  test<2, 2>(comm);
}
