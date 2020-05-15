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

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>

#include "consensus_algorithm_util.h"

template <int dim, int spacedim>
void
check(const Triangulation<dim, spacedim> &tria_dst,
      const Triangulation<dim, spacedim> &tria_src,
      const MPI_Comm &                    communicator)
{
  CellIDTranslator<dim> translator_fine(tria_dst.n_cells(0),
                                        tria_dst.n_global_levels());
  IndexSet              is_dst_locally_owned(translator_fine.size());

  for (auto cell : tria_dst.active_cell_iterators())
    is_dst_locally_owned.add_index(translator_fine.translate(cell));

  CellIDTranslator<dim> translator_coarse(tria_dst.n_cells(0),
                                          tria_dst.n_global_levels());

  IndexSet is_dst_remote_potentially_relevant(translator_coarse.size());

  for (auto cell : tria_src.active_cell_iterators())
    {
      is_dst_remote_potentially_relevant.add_index(
        translator_coarse.translate(cell));

      if (cell->level() + 1 == tria_dst.n_global_levels())
        continue;

      for (unsigned int i = 0; i < GeometryInfo<dim>::max_children_per_cell;
           ++i)
        is_dst_remote_potentially_relevant.add_index(
          translator_coarse.translate(cell, i));
    }

  is_dst_remote_potentially_relevant.subtract_set(is_dst_locally_owned);

  std::vector<unsigned int> owning_ranks_of_ghosts(
    is_dst_remote_potentially_relevant.n_elements());

  {
    Utilities::MPI::internal::ComputeIndexOwner::ConsensusAlgorithmsPayload
      process(is_dst_locally_owned,
              is_dst_remote_potentially_relevant,
              communicator,
              owning_ranks_of_ghosts,
              false);

    Utilities::MPI::ConsensusAlgorithms::Selector<
      std::pair<types::global_dof_index, types::global_dof_index>,
      unsigned int>
      consensus_algorithm(process, communicator);
    consensus_algorithm.run();
  }

  IndexSet is_dst_remote(translator_coarse.size());

  for (unsigned i = 0; i < is_dst_remote_potentially_relevant.n_elements(); ++i)
    if (owning_ranks_of_ghosts[i] != numbers::invalid_unsigned_int)
      is_dst_remote.add_index(
        is_dst_remote_potentially_relevant.nth_index_in_set(i));


  is_dst_remote.print(deallog.get_file_stream());

  std::vector<unsigned int> owning_ranks_of_ghosts_clean(
    is_dst_remote.n_elements());

  {
    Utilities::MPI::internal::ComputeIndexOwner::ConsensusAlgorithmsPayload
      process(is_dst_locally_owned,
              is_dst_remote,
              communicator,
              owning_ranks_of_ghosts_clean,
              true);

    Utilities::MPI::ConsensusAlgorithms::Selector<
      std::pair<types::global_dof_index, types::global_dof_index>,
      unsigned int>
      consensus_algorithm(process, communicator);
    consensus_algorithm.run();
  }

  // is_dst_locally_owned & is_dst_remote
}

template <int dim, int spacedim>
void
test(const MPI_Comm &comm)
{
  // create first mesh
  parallel::distributed::Triangulation<dim, spacedim> tria_1(
    comm,
    ::Triangulation<dim, spacedim>::none,
    parallel::distributed::Triangulation<dim, spacedim>::Settings::
      construct_multigrid_hierarchy);
  GridGenerator::hyper_cube(tria_1, -1, +1);
  tria_1.refine_global(3);

  for (unsigned int i = 0; i < 4; ++i)
    {
      for (auto cell : tria_1.active_cell_iterators())

        if (cell->is_locally_owned() && cell->center()[0] < 0.0 &&
            cell->center()[1] < 0.0)
          cell->set_refine_flag();
      tria_1.execute_coarsening_and_refinement();
    }

  // create second mesh
  parallel::distributed::Triangulation<dim, spacedim> tria_2(
    comm,
    ::Triangulation<dim, spacedim>::none,
    parallel::distributed::Triangulation<dim, spacedim>::Settings::
      construct_multigrid_hierarchy);
  GridGenerator::hyper_cube(tria_2, -1, +1);

  tria_1.save("mesh");
  tria_2.load("mesh", false);

  {
    for (auto cell : tria_2.active_cell_iterators())
      if (cell->is_locally_owned())
        cell->set_coarsen_flag();

    tria_2.execute_coarsening_and_refinement();
  }

  check(tria_1, tria_2, comm);
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  MPILogInitAll                    all;

  const MPI_Comm comm = MPI_COMM_WORLD;

  test<2, 2>(comm);
}
