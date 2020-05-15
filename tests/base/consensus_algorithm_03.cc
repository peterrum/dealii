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

#include "../tests.h"

template <int dim>
types::coarse_cell_id
convert_cell_id_binary_type_to_level_coarse_cell_id(
  const typename CellId::binary_type &binary_representation)
{
  // exploiting the structure of CellId::binary_type
  // see also the documentation of CellId

  // actual coarse-grid id
  const unsigned int coarse_cell_id  = binary_representation[0];
  const unsigned int n_child_indices = binary_representation[1] >> 2;

  const unsigned int children_per_value =
    sizeof(CellId::binary_type::value_type) * 8 / dim;
  unsigned int child_level  = 0;
  unsigned int binary_entry = 2;

  // path to the get to the cell
  std::vector<unsigned int> cell_indices;
  while (child_level < n_child_indices)
    {
      Assert(binary_entry < binary_representation.size(), ExcInternalError());

      for (unsigned int j = 0; j < children_per_value; ++j)
        {
          unsigned int cell_index =
            (((binary_representation[binary_entry] >> (j * dim))) &
             (GeometryInfo<dim>::max_children_per_cell - 1));
          cell_indices.push_back(cell_index);
          ++child_level;
          if (child_level == n_child_indices)
            break;
        }
      ++binary_entry;
    }

  // compute new coarse-grid id: c_{i+1} = c_{i}*2^dim + q;
  types::coarse_cell_id level_coarse_cell_id = coarse_cell_id;
  for (auto i : cell_indices)
    level_coarse_cell_id =
      level_coarse_cell_id * GeometryInfo<dim>::max_children_per_cell + i;

  return level_coarse_cell_id;
}


template <int dim>
class CellIDTranslator
{
public:
  CellIDTranslator(const unsigned int n_coarse_cells,
                   const unsigned int n_global_levels)
    : n_coarse_cells(n_coarse_cells)
    , n_global_levels(n_global_levels)
  {}

  unsigned int
  size()
  {
    return n_coarse_cells *
           (Utilities::pow(GeometryInfo<dim>::max_children_per_cell,
                           n_global_levels) -
            1);
  }

  template <typename T>
  unsigned int
  translate(const T &cell)
  {
    return convert_cell_id_binary_type_to_level_coarse_cell_id<dim>(
             cell->id().template to_binary<dim>()) +
           n_coarse_cells *
             (Utilities::pow(GeometryInfo<dim>::max_children_per_cell,
                             cell->level()) -
              1);
  }

  CellId
  to_cell_id(const unsigned int id)
  {
    const std::vector<std::uint8_t> child_indices;
    return CellId(id, child_indices); // TODO
  }

  template <typename T>
  unsigned int
  translate(const T &cell, const unsigned int i)
  {
    return translate(cell) * GeometryInfo<dim>::max_children_per_cell + i;
  }

private:
  const unsigned int n_coarse_cells;
  const unsigned int n_global_levels;
};


class VectorRepartitioner
{
public:
  template <int dim, int spacedim>
  VectorRepartitioner(const DoFHandler<dim, spacedim> &dof_handler_dst,
                      const DoFHandler<dim, spacedim> &dof_handler_src,
                      const MPI_Comm &                 communicator)
    : communicator(communicator)
  {
    const Triangulation<dim, spacedim> &tria_dst =
      dof_handler_dst.get_triangulation();
    const Triangulation<dim, spacedim> &tria_src =
      dof_handler_src.get_triangulation();


    CellIDTranslator<dim> translator_fine(tria_dst.n_cells(0),
                                          tria_dst.n_global_levels());
    IndexSet              is_dst_locally_owned(translator_fine.size());

    for (auto cell : tria_dst.active_cell_iterators())
      if (!cell->is_artificial() && cell->is_locally_owned())
        is_dst_locally_owned.add_index(translator_fine.translate(cell));

    CellIDTranslator<dim> translator_coarse(tria_dst.n_cells(0),
                                            tria_dst.n_global_levels());

    IndexSet is_dst_remote_potentially_relevant(translator_coarse.size());
    IndexSet is_src_locally_owned(translator_coarse.size());

    for (auto cell : tria_src.active_cell_iterators())
      if (!cell->is_artificial() && cell->is_locally_owned())
        {
          is_src_locally_owned.add_index(translator_coarse.translate(cell));
          is_dst_remote_potentially_relevant.add_index(
            translator_coarse.translate(cell));
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

    for (unsigned i = 0; i < is_dst_remote_potentially_relevant.n_elements();
         ++i)
      if (owning_ranks_of_ghosts[i] != numbers::invalid_unsigned_int)
        is_dst_remote.add_index(
          is_dst_remote_potentially_relevant.nth_index_in_set(i));


    is_dst_remote.print(deallog.get_file_stream());

    std::vector<unsigned int> owning_ranks_of_ghosts_clean(
      is_dst_remote.n_elements());

    std::map<unsigned int, IndexSet> targets_with_indexset;
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

      targets_with_indexset = process.get_requesters();
    }

    std::map<unsigned int, std::vector<unsigned int>> indices_to_be_sent;
    std::vector<MPI_Request>                          requests;
    requests.reserve(targets_with_indexset.size());

    {
      std::vector<types::global_dof_index> indices(
        dof_handler_dst.get_fe().dofs_per_cell);

      for (auto i : targets_with_indexset)
        {
          deallog << i.first << ": ";
          i.second.print(deallog.get_file_stream());

          indices_to_be_sent[i.first] = {};
          auto &buffer                = indices_to_be_sent[i.first];

          for (auto cell_id : i.second)
            {
              typename DoFHandler<dim, spacedim>::cell_iterator cell(
                *translator_fine.to_cell_id(cell_id).to_cell(tria_dst),
                &dof_handler_dst);

              cell->get_dof_indices(indices);
              buffer.insert(buffer.end(), indices.begin(), indices.end());

              for (auto i : indices)
                deallog << i << " ";
              deallog << std::endl;
            }

          requests.resize(requests.size() + 1);

          MPI_Isend(buffer.data(),
                    buffer.size(),
                    MPI_UNSIGNED,
                    i.first,
                    11,
                    communicator,
                    &requests.back());
        }
    }

    std::vector<unsigned int> ghost_indices;

    std::map<unsigned int, std::vector<unsigned int>> rank_to_ids;

    this->indices.resize(dof_handler_src.locally_owned_dofs().n_elements());


    // process local cells
    {
      auto is_src_and_dst_locally_owned = is_src_locally_owned;
      is_src_and_dst_locally_owned.subtract_set(is_dst_remote);

      std::vector<types::global_dof_index> indices(
        dof_handler_dst.get_fe().dofs_per_cell);

      std::vector<types::global_dof_index> indices_(
        dof_handler_dst.get_fe().dofs_per_cell);

      for (const auto id : is_src_and_dst_locally_owned)
        {
          const auto cell_id = translator_coarse.to_cell_id(id);

          typename DoFHandler<dim, spacedim>::cell_iterator cell(
            *cell_id.to_cell(tria_src), &dof_handler_src);

          typename DoFHandler<dim, spacedim>::cell_iterator cell_(
            *cell_id.to_cell(tria_dst), &dof_handler_dst);

          cell->get_dof_indices(indices);
          cell_->get_dof_indices(indices_);

          for (unsigned int j = 0; j < dof_handler_dst.get_fe().dofs_per_cell;
               ++j)
            {
              if (dof_handler_src.locally_owned_dofs().is_element(indices[j]))
                this->indices[dof_handler_src.locally_owned_dofs()
                                .index_within_set(indices[j])] = indices_[j];
            }
        }
    }

    {
      std::set<unsigned int> ranks;

      for (auto i : owning_ranks_of_ghosts_clean)
        ranks.insert(i);

      for (auto i : ranks)
        rank_to_ids[i] = {};

      for (unsigned int i = 0; i < owning_ranks_of_ghosts_clean.size(); ++i)
        rank_to_ids[owning_ranks_of_ghosts_clean[i]].push_back(
          is_dst_remote.nth_index_in_set(i));


      for (unsigned int i = 0; i < ranks.size(); ++i)
        {
          MPI_Status status;
          MPI_Probe(MPI_ANY_SOURCE, 11, communicator, &status);

          int message_length;
          MPI_Get_count(&status, MPI_UNSIGNED, &message_length);

          std::vector<unsigned int> buffer(message_length);

          MPI_Recv(buffer.data(),
                   buffer.size(),
                   MPI_UNSIGNED,
                   status.MPI_SOURCE,
                   11,
                   communicator,
                   MPI_STATUS_IGNORE);

          ghost_indices.insert(ghost_indices.end(),
                               buffer.begin(),
                               buffer.end());

          const unsigned int rank = status.MPI_SOURCE;

          const auto ids = rank_to_ids[rank];

          {
            std::vector<types::global_dof_index> indices(
              dof_handler_dst.get_fe().dofs_per_cell);

            for (unsigned int i = 0, k = 0; i < ids.size(); ++i)
              {
                typename DoFHandler<dim, spacedim>::cell_iterator cell(
                  *translator_coarse.to_cell_id(ids[i]).to_cell(tria_src),
                  &dof_handler_src);

                cell->get_dof_indices(indices);

                for (unsigned int j = 0;
                     j < dof_handler_dst.get_fe().dofs_per_cell;
                     ++j, ++k)
                  {
                    if (dof_handler_src.locally_owned_dofs().is_element(
                          indices[j]))
                      this->indices[dof_handler_src.locally_owned_dofs()
                                      .index_within_set(indices[j])] =
                        buffer[k];
                  }
              }
          }
        }

      MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    }

    std::sort(ghost_indices.begin(), ghost_indices.end());
    ghost_indices.erase(std::unique(ghost_indices.begin(), ghost_indices.end()),
                        ghost_indices.end());

    this->is_dst_large = IndexSet(dof_handler_dst.n_dofs());
    is_dst_large.add_indices(ghost_indices.begin(), ghost_indices.end());

    deallog << "AA " << std::endl;
    this->d = dof_handler_dst.locally_owned_dofs();

    IndexSet dd;
    DoFTools::extract_locally_relevant_dofs(dof_handler_dst, dd);

    d.print(deallog.get_file_stream());
    dd.print(deallog.get_file_stream());
    is_dst_large.print(deallog.get_file_stream());

    is_dst_large.add_indices(dd);
    is_dst_large.print(deallog.get_file_stream());

    LinearAlgebra::distributed::Vector<double> vector(d, dd, communicator);

    LinearAlgebra::distributed::Vector<double> vector_extended(d,
                                                               is_dst_large,
                                                               communicator);

    vector_extended.copy_locally_owned_data_from(vector);
    vector_extended.update_ghost_values();

    for (auto i : owning_ranks_of_ghosts_clean)
      deallog << i << " ";
    deallog << std::endl;
  }

  template <typename Number>
  void
  update(LinearAlgebra::distributed::Vector<Number> &      dst,
         const LinearAlgebra::distributed::Vector<Number> &src) const
  {
    // create new source vector with matching ghost values
    LinearAlgebra::distributed::Vector<Number> src_extended(d,
                                                            is_dst_large,
                                                            communicator);

    // copy locally owned values from original source vector
    src_extended.copy_locally_owned_data_from(src);

    // update ghost values
    src_extended.update_ghost_values();

    // copy locally owned values from temporal array to destination vector
    for (unsigned int i = 0; i < indices.size(); ++i)
      dst.local_element(i) = src_extended[indices[i]]; // TODO
  }

private:
  MPI_Comm communicator;
  IndexSet d;
  IndexSet is_dst_large;

  std::vector<unsigned int> indices;
};

template <int dim, int spacedim>
void
test(const MPI_Comm &comm)
{
  Triangulation<dim> basetria;
  GridGenerator::subdivided_hyper_cube(basetria, 4);

  parallel::fullydistributed::Triangulation<dim> tria_1(comm);
  parallel::fullydistributed::Triangulation<dim> tria_2(comm);

  {
    GridTools::partition_triangulation_zorder(
      Utilities::MPI::n_mpi_processes(comm), basetria);

    auto construction_data = TriangulationDescription::Utilities::
      create_description_from_triangulation(basetria, comm);

    tria_1.create_triangulation(construction_data);
  }

  {
    GridTools::partition_triangulation_zorder(
      Utilities::MPI::n_mpi_processes(comm) / 2, basetria);

    auto construction_data = TriangulationDescription::Utilities::
      create_description_from_triangulation(basetria, comm);

    tria_2.create_triangulation(construction_data);
  }

  FE_Q<dim, spacedim> fe(1);

  DoFHandler<dim, spacedim> dof_handler_1(tria_1);
  dof_handler_1.distribute_dofs(fe);

  DoFHandler<dim, spacedim> dof_handler_2(tria_2);
  dof_handler_2.distribute_dofs(fe);

  VectorRepartitioner vr(dof_handler_1, dof_handler_2, comm);

  // setup first vector
  IndexSet d1 = dof_handler_1.locally_owned_dofs();
  IndexSet dd1;
  DoFTools::extract_locally_relevant_dofs(dof_handler_1, dd1);

  LinearAlgebra::distributed::Vector<double> vec1(d1, dd1, comm);

  for (auto i : dof_handler_1.locally_owned_dofs())
    vec1[i] = i;

  // setup second vector
  IndexSet d2 = dof_handler_2.locally_owned_dofs();
  IndexSet dd2;
  DoFTools::extract_locally_relevant_dofs(dof_handler_2, dd2);

  LinearAlgebra::distributed::Vector<double> vec2(d2, dd2, comm);

  vr.update(vec2, vec1);

  vec1.print(deallog.get_file_stream());
  vec2.print(deallog.get_file_stream());

  AssertDimension(vec1.l2_norm(), vec2.l2_norm());
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  MPILogInitAll                    all;

  const MPI_Comm comm = MPI_COMM_WORLD;

  test<2, 2>(comm);
}
