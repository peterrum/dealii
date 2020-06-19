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

#ifndef dealii_tet_partition_h
#define dealii_tet_partition_h

#include <deal.II/base/config.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_poly.h>

#include <deal.II/tet/fe_q.h>
#include <deal.II/tet/mapping_q.h>
#include <deal.II/tet/polynomials.h>

DEAL_II_NAMESPACE_OPEN

namespace Tet
{
  template <int dim, int spacedim>
  void
  partition_triangulation(const unsigned                n_part,
                          Triangulation<dim, spacedim> &tria,
                          const bool                    distributed_mesh = true)
  {
    // determine number of cells
    const unsigned int n_cells = tria.n_active_cells();

    // determine number of cells per process
    const unsigned int n_cells_per_proc = (n_cells + n_part - 1) / n_part;

    // partition mesh
    unsigned int counter = 0;
    for (const auto &cell : tria.cell_iterators())
      cell->set_subdomain_id(counter++ / n_cells_per_proc);

    if (distributed_mesh == false)
      return;

    // collect vertices of locally owned cells
    std::vector<bool> vertex_of_own_cell(tria.n_vertices(), false);
    for (const auto &cell : tria.active_cell_iterators())
      if (cell->is_locally_owned())
        for (const auto v : cell->vertex_indices())
          vertex_of_own_cell[cell->vertex_index(v)] = true;

    // clear artificial cells
    for (const auto &cell : tria.cell_iterators())
      {
        const auto temp = cell->subdomain_id();
        cell->set_subdomain_id(numbers::artificial_subdomain_id);

        for (const auto v : cell->vertex_indices())
          if (vertex_of_own_cell[cell->vertex_index(v)])
            {
              cell->set_subdomain_id(temp);
              break;
            }
      }
  }

} // namespace Tet

DEAL_II_NAMESPACE_CLOSE

#endif
