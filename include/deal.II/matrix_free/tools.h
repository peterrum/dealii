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

#ifndef dealii_matrix_free_tools_h
#define dealii_matrix_free_tools_h

#include <deal.II/base/config.h>

#include <deal.II/matrix_free/matrix_free.h>



DEAL_II_NAMESPACE_OPEN

/**
 * A namespace for utility functions in the context of matrix-free operator
 * evaluation.
 */
namespace MatrixFreeTools
{
  /**
   * Modify @p additional_data so that cells are categorized
   * according to their boundary ids, making face integrals in the case of
   * ECL simpler.
   */
  template <int dim, typename AdditionalData>
  void
  categorize_accoring_boundary_ids_for_ecl(
    const Triangulation<dim> &tria,
    AdditionalData &          additional_data,
    const unsigned int        level = numbers::invalid_unsigned_int)
  {
    bool is_mg = (level != numbers::invalid_unsigned_int);

    // ... create list for the category of each cell
    if (is_mg)
      additional_data.cell_vectorization_category.resize(
        std::distance(tria.begin(level), tria.end(level)));
    else
      additional_data.cell_vectorization_category.resize(tria.n_active_cells());

    // ... setup scaling factor
    std::vector<unsigned int> factors(dim * 2);

    std::map<unsigned int, unsigned int> bid_map;
    for (unsigned int i = 0; i < tria.get_boundary_ids().size(); i++)
      bid_map[tria.get_boundary_ids()[i]] = i + 1;

    {
      unsigned int bids   = tria.get_boundary_ids().size() + 1;
      int          offset = 1;
      for (unsigned int i = 0; i < dim * 2; i++, offset = offset * bids)
        factors[i] = offset;
    }

    auto to_category = [&](auto &cell) {
      unsigned int c_num = 0;
      for (unsigned int i = 0; i < dim * 2; i++)
        {
          auto &face = *cell->face(i);
          if (face.at_boundary())
            c_num += factors[i] * bid_map[face.boundary_id()];
        }
      return c_num;
    };

    if (!is_mg)
      {
        for (auto cell = tria.begin_active(); cell != tria.end(); ++cell)
          {
            if (cell->is_locally_owned())
              additional_data
                .cell_vectorization_category[cell->active_cell_index()] =
                to_category(cell);
          }
      }
    else
      {
        for (auto cell = tria.begin(level); cell != tria.end(level); ++cell)
          {
            if (cell->is_locally_owned_on_level())
              additional_data.cell_vectorization_category[cell->index()] =
                to_category(cell);
          }
      }

    // ... finalize setup of matrix_free
    additional_data.hold_all_faces_to_owned_cells        = true;
    additional_data.cell_vectorization_categories_strict = true;
    additional_data.mapping_update_flags_faces_by_cells =
      additional_data.mapping_update_flags_inner_faces |
      additional_data.mapping_update_flags_boundary_faces;
  }

} // namespace MatrixFreeTools


DEAL_II_NAMESPACE_CLOSE


#endif
