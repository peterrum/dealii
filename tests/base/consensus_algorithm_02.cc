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
check(const DoFHandler<dim, spacedim> &dof_handler_dst,
      const DoFHandler<dim, spacedim> &dof_handler_src,
      const MPI_Comm &                 communicator)
{
  // get reference to triangulations
  const auto &tria_dst = dof_handler_dst.get_triangulation();
  const auto &tria_src = dof_handler_src.get_triangulation();

  const auto deterimine_n_coarse_cells = [&communicator](auto &tria) {
    types::coarse_cell_id n_coarse_cells = 0;

    for (auto cell : tria.active_cell_iterators())
      if (!cell->is_artificial())
        n_coarse_cells =
          std::max(n_coarse_cells, cell->id().get_coarse_cell_id());

    return Utilities::MPI::max(n_coarse_cells, communicator) + 1;
  };

  const auto n_coarse_cells_dst = deterimine_n_coarse_cells(tria_dst);

  AssertDimension(n_coarse_cells_dst, deterimine_n_coarse_cells(tria_src));

  // create translator: CellID <-> unique ID
  CellIDTranslator<dim> cell_id_translator(n_coarse_cells_dst,
                                           tria_dst.n_global_levels());


  // create index sets
  IndexSet is_dst_locally_owned(cell_id_translator.size());
  IndexSet is_dst_remote(cell_id_translator.size());
  IndexSet is_dst_remote_potentially_relevant(cell_id_translator.size());
  IndexSet is_src_locally_owned(cell_id_translator.size());

  for (auto cell : tria_dst.active_cell_iterators())
    if (!cell->is_artificial() && cell->is_locally_owned())
      is_dst_locally_owned.add_index(cell_id_translator.translate(cell));


  for (auto cell : tria_src.active_cell_iterators())
    if (!cell->is_artificial() && cell->is_locally_owned())
      {
        is_src_locally_owned.add_index(cell_id_translator.translate(cell));
        is_dst_remote_potentially_relevant.add_index(
          cell_id_translator.translate(cell));

        if (cell->level() + 1 == tria_dst.n_global_levels())
          continue;

        for (unsigned int i = 0; i < GeometryInfo<dim>::max_children_per_cell;
             ++i)
          is_dst_remote_potentially_relevant.add_index(
            cell_id_translator.translate(cell, i));
      }

  is_dst_remote_potentially_relevant.subtract_set(is_dst_locally_owned);

  {
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

    for (unsigned i = 0; i < is_dst_remote_potentially_relevant.n_elements();
         ++i)
      if (owning_ranks_of_ghosts[i] != numbers::invalid_unsigned_int)
        is_dst_remote.add_index(
          is_dst_remote_potentially_relevant.nth_index_in_set(i));
  }

  // determine owner of remote cells
  std::vector<unsigned int> is_dst_remote_owners(is_dst_remote.n_elements());

  Utilities::MPI::internal::ComputeIndexOwner::ConsensusAlgorithmsPayload
    process(is_dst_locally_owned,
            is_dst_remote,
            communicator,
            is_dst_remote_owners,
            true);

  Utilities::MPI::ConsensusAlgorithms::Selector<
    std::pair<types::global_dof_index, types::global_dof_index>,
    unsigned int>
    consensus_algorithm(process, communicator);
  consensus_algorithm.run();

  {
    deallog << "IS_SRC_LOCALLY_OWNED" << std::endl;
    for (auto i : is_src_locally_owned)
      deallog << cell_id_translator.to_cell_id(i) << std::endl;
    deallog << std::endl << std::endl << std::endl;


    deallog << "IS_DST_LOCALLY_OWNED" << std::endl;
    for (auto i : is_dst_locally_owned)
      deallog << cell_id_translator.to_cell_id(i) << std::endl;
    deallog << std::endl << std::endl << std::endl;


    deallog << "IS_DST_REMOTE" << std::endl;
    for (unsigned int i = 0; i < is_dst_remote.n_elements(); i++)
      {
        deallog << cell_id_translator.to_cell_id(
                     is_dst_remote.nth_index_in_set(i))
                << " -> " << is_dst_remote_owners[i] << std::endl;
      }
  }
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
  tria_1.refine_global(1);

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

  if (Utilities::MPI::n_mpi_processes(comm) == 1)
    {
      for (const auto cell : tria_2.active_cell_iterators())
        {
          deallog << cell->id() << " : ";

          const auto other_cell = cell->id().to_cell(tria_1);

          if (other_cell->has_children())
            {
              for (unsigned int i = 0; i < other_cell->n_children(); ++i)
                deallog << other_cell->child(i)->id() << " ";
            }

          deallog << std::endl;
        }
    }

  FE_Q<dim, spacedim> fe(1);

  DoFHandler<dim, spacedim> dof_handler_1(tria_1);
  dof_handler_1.distribute_dofs(fe);

  DoFHandler<dim, spacedim> dof_handler_2(tria_2);
  dof_handler_2.distribute_dofs(fe);

  check(dof_handler_1, dof_handler_2, comm);
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  MPILogInitAll                    all;

  const MPI_Comm comm = MPI_COMM_WORLD;

  test<2, 2>(comm);
}
