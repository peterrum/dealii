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
  {
    tree_sizes.push_back(0);
    for (unsigned int i = 0; i < n_global_levels; ++i)
      tree_sizes.push_back(
        tree_sizes.back() +
        Utilities::pow(GeometryInfo<dim>::max_children_per_cell, i) *
          n_coarse_cells);
  }

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
    unsigned int id = 0;

    id += convert_cell_id_binary_type_to_level_coarse_cell_id<dim>(
      cell->id().template to_binary<dim>());

    id += tree_sizes[cell->level()];

    return id;
  }

  template <typename T>
  unsigned int
  translate(const T &cell, const unsigned int i)
  {
    return translate(cell) * GeometryInfo<dim>::max_children_per_cell + i +
           tree_sizes[cell->level() + 1];
  }

  CellId
  to_cell_id(const unsigned int id)
  {
    std::vector<std::uint8_t> child_indices;

    unsigned int id_temp = id;

    unsigned int level = 0;

    for (; level < n_global_levels; ++level)
      if (id < tree_sizes[level])
        break;
    level -= 1;

    id_temp -= tree_sizes[level];

    for (unsigned int l = 0; l < level; ++l)
      {
        child_indices.push_back(id_temp %
                                GeometryInfo<dim>::max_children_per_cell);
        id_temp /= GeometryInfo<dim>::max_children_per_cell;
      }

    std::reverse(child_indices.begin(), child_indices.end());

    return CellId(id_temp, child_indices); // TODO
  }

private:
  const unsigned int        n_coarse_cells;
  const unsigned int        n_global_levels;
  std::vector<unsigned int> tree_sizes;
};
