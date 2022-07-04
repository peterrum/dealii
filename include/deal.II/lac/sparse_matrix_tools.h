// ---------------------------------------------------------------------
//
// Copyright (C) 1999 - 2022 by the deal.II authors
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

#ifndef dealii_sparse_matrix_tools_h
#define dealii_sparse_matrix_tools_h

#include <deal.II/base/config.h>

#include <deal.II/base/mpi_compute_index_owner_internal.h>

DEAL_II_NAMESPACE_OPEN

namespace SparseMatrixTools
{
  template <int dim,
            int spacedim,
            typename SparseMatrixType,
            typename SparsityPatternType,
            typename Number>
  void
  restrict_to_cells(const SparseMatrixType &         system_matrix,
                    const SparsityPatternType &      sparsity_pattern,
                    const DoFHandler<dim, spacedim> &dof_handler,
                    std::vector<FullMatrix<Number>> &blocks)
  {
    const unsigned int dofs_per_cell = dof_handler.get_fe().n_dofs_per_cell();

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    const auto locally_owned_dofs = dof_handler.locally_owned_dofs();

    IndexSet locally_active_dofs;
    DoFTools::extract_locally_active_dofs(dof_handler, locally_active_dofs);

    locally_active_dofs.subtract_set(locally_owned_dofs);

    const auto comm = dof_handler.get_communicator();

    std::vector<unsigned int> dummy(locally_active_dofs.n_elements());

    Utilities::MPI::internal::ComputeIndexOwner::ConsensusAlgorithmsPayload
      process(locally_owned_dofs, locally_active_dofs, comm, dummy, true);

    Utilities::MPI::ConsensusAlgorithms::Selector<
      std::vector<std::pair<types::global_dof_index, types::global_dof_index>>,
      std::vector<unsigned int>>
      consensus_algorithm;
    consensus_algorithm.run(process, comm);

    using T1 = std::vector<
      std::pair<types::global_dof_index,
                std::vector<std::pair<types::global_dof_index, Number>>>>;

    auto requesters = process.get_requesters();

    std::vector<std::vector<std::pair<types::global_dof_index, Number>>>
      locally_relevant_matrix_entries(locally_active_dofs.n_elements());


    std::vector<unsigned int> ranks;

    for (const auto &i : requesters)
      ranks.push_back(i.first);

    dealii::Utilities::MPI::ConsensusAlgorithms::selector<T1>(
      ranks,
      [&](const unsigned int other_rank) {
        T1 send_buffer;

        for (auto index : requesters[other_rank])
          {
            std::vector<std::pair<types::global_dof_index, Number>> t;

            for (auto entry = system_matrix.begin(index);
                 entry != system_matrix.end(index);
                 ++entry)
              t.emplace_back(entry->column(), entry->value());

            send_buffer.emplace_back(index, t);
          }

        return send_buffer;
      },
      [&](const unsigned int &, const T1 &buffer_recv) {
        for (const auto &i : buffer_recv)
          {
            auto &dst =
              locally_relevant_matrix_entries[locally_active_dofs
                                                .index_within_set(i.first)];
            dst = i.second;
            std::sort(dst.begin(), dst.end(), [](const auto &a, const auto &b) {
              return a.first < b.first;
            });
          }
      },
      comm);

    blocks.clear();
    blocks.resize(dof_handler.get_triangulation().n_active_cells());

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned() == false)
          continue;

        cell->get_dof_indices(local_dof_indices);

        auto &cell_matrix = blocks[cell->active_cell_index()];

        cell_matrix = FullMatrix<Number>(dofs_per_cell, dofs_per_cell);

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
              if (locally_owned_dofs.is_element(
                    local_dof_indices[i])) // row is local
                {
                  cell_matrix(i, j) =
                    sparsity_pattern.exists(local_dof_indices[i],
                                            local_dof_indices[j]) ?
                      system_matrix(local_dof_indices[i],
                                    local_dof_indices[j]) :
                      0;
                }
              else // row is ghost
                {
                  Assert(locally_active_dofs.is_element(local_dof_indices[i]),
                         ExcInternalError());

                  const auto &row_entries =
                    locally_relevant_matrix_entries[locally_active_dofs
                                                      .index_within_set(
                                                        local_dof_indices[i])];

                  const auto ptr =
                    std::lower_bound(row_entries.begin(),
                                     row_entries.end(),
                                     std::pair<types::global_dof_index, Number>{
                                       local_dof_indices[j], /*dummy*/ 0.0},
                                     [](const auto a, const auto b) {
                                       return a.first < b.first;
                                     });

                  if (ptr != row_entries.end() &&
                      local_dof_indices[j] == ptr->first)
                    cell_matrix(i, j) = ptr->second;
                  else
                    cell_matrix(i, j) = 0.0;
                }
            }
      }
  }

} // namespace SparseMatrixTools

DEAL_II_NAMESPACE_CLOSE

#endif
