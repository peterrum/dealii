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

#include <deal.II/grid/tria.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/vector_access_internal.h>


DEAL_II_NAMESPACE_OPEN

/**
 * A namespace for utility functions in the context of matrix-free operator
 * evaluation.
 */
namespace MatrixFreeTools
{
  /**
   * Modify @p additional_data so that cells are categorized
   * according to their boundary IDs, making face integrals in the case of
   * cell-centric loop simpler.
   */
  template <int dim, typename AdditionalData>
  void
  categorize_by_boundary_ids(
    const Triangulation<dim> &tria,
    AdditionalData &          additional_data,
    const unsigned int        level = numbers::invalid_unsigned_int);

  /**
   * Compute the diagonal of a linear operator (@p diagonal_global), given
   * @p matrix_free, @p phi, and the local cell integral operation @p local_vmult.
   */
  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components,
            typename Number,
            typename VectorizedArrayType>
  void
  compute_diagonal(
    const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
    FEEvaluation<dim,
                 fe_degree,
                 n_q_points_1d,
                 n_components,
                 Number,
                 VectorizedArrayType> &                 phi,
    LinearAlgebra::distributed::Vector<Number> &        diagonal_global,
    const std::function<void()> &                       local_vmult);


  // implementations



  template <int dim, typename AdditionalData>
  void
  categorize_by_boundary_ids(const Triangulation<dim> &tria,
                             AdditionalData &          additional_data,
                             const unsigned int        level)
  {
    // ... determine if we are on an active or a multigrid level
    const bool is_mg = (level != numbers::invalid_unsigned_int);

    // ... create empty list for the category of each cell
    if (is_mg)
      additional_data.cell_vectorization_category.resize(
        std::distance(tria.begin(level), tria.end(level)));
    else
      additional_data.cell_vectorization_category.resize(tria.n_active_cells());

    // ... set up scaling factor
    std::vector<unsigned int> factors(GeometryInfo<dim>::faces_per_cell);

    auto bids = tria.get_boundary_ids();
    std::sort(bids.begin(), bids.end());

    {
      unsigned int n_bids = bids.size() + 1;
      int          offset = 1;
      for (unsigned int i = 0; i < GeometryInfo<dim>::faces_per_cell;
           i++, offset = offset * n_bids)
        factors[i] = offset;
    }

    const auto to_category = [&](const auto &cell) {
      unsigned int c_num = 0;
      for (unsigned int i = 0; i < GeometryInfo<dim>::faces_per_cell; i++)
        {
          auto &face = *cell->face(i);
          if (face.at_boundary() && !cell->has_periodic_neighbor(i))
            c_num +=
              factors[i] * (1 + std::distance(bids.begin(),
                                              std::find(bids.begin(),
                                                        bids.end(),
                                                        face.boundary_id())));
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

    // ... finalize set up of matrix_free
    additional_data.hold_all_faces_to_owned_cells        = true;
    additional_data.cell_vectorization_categories_strict = true;
    additional_data.mapping_update_flags_faces_by_cells =
      additional_data.mapping_update_flags_inner_faces |
      additional_data.mapping_update_flags_boundary_faces;
  }

  namespace internal
  {
    template <typename Number>
    struct LocalCSR
    {
      std::vector<unsigned int> row_lid_to_gid;
      std::vector<unsigned int> row{0};
      std::vector<unsigned int> col;
      std::vector<Number>       val;
    };
  } // namespace internal

  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components,
            typename Number,
            typename VectorizedArrayType>
  void
  compute_diagonal(
    const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
    FEEvaluation<dim,
                 fe_degree,
                 n_q_points_1d,
                 n_components,
                 Number,
                 VectorizedArrayType> &                 phi,
    LinearAlgebra::distributed::Vector<Number> &        diagonal_global,
    const std::function<void()> &                       local_vmult)
  {
    matrix_free.initialize_dof_vector(diagonal_global);

    LinearAlgebra::distributed::Vector<Number> dummy;
    matrix_free.initialize_dof_vector(dummy);

    matrix_free.template cell_loop<LinearAlgebra::distributed::Vector<Number>,
                                   LinearAlgebra::distributed::Vector<Number>>(
      [&phi, &local_vmult](
        const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
        LinearAlgebra::distributed::Vector<Number> &        diagonal_global,
        const LinearAlgebra::distributed::Vector<Number> &,
        const std::pair<unsigned int, unsigned int> &range) mutable {
        for (unsigned int cell = range.first; cell < range.second; cell++)
          {
            const auto &       dof_info = matrix_free.get_dof_info();
            const unsigned int n_fe_components =
              dof_info.start_components.back();
            const unsigned int n_vectorization =
              VectorizedArrayType::n_array_elements;
            const unsigned int first_selected_component =
              0; // TODO: where should this come from?
            const unsigned int dofs_per_component = phi.dofs_per_component;

            const unsigned int *dof_indices[n_vectorization];
            const unsigned int *plain_dof_indices[n_vectorization];

            std::vector<std::tuple<unsigned int, unsigned int, Number>>
              constraint_list;


            const unsigned int n_vectorization_actual =
              matrix_free.n_active_entries_per_cell_batch(cell);

            {
              const unsigned int n_components_read =
                n_fe_components > 1 ? n_components : 1;
              for (unsigned int v = 0; v < n_vectorization_actual; ++v)
                {
                  dof_indices[v] = dof_info.dof_indices.data() +
                                   dof_info
                                     .row_starts[(cell * n_vectorization + v) *
                                                   n_fe_components +
                                                 first_selected_component]
                                     .first;

                  plain_dof_indices[v] =
                    dof_info.plain_dof_indices.data() +
                    dof_info
                      .row_starts_plain_indices[(cell * n_vectorization + v) *
                                                  n_fe_components +
                                                first_selected_component];
                }
              for (unsigned int v = n_vectorization_actual; v < n_vectorization;
                   ++v)
                {
                  dof_indices[v]       = nullptr;
                  plain_dof_indices[v] = nullptr;
                }
            }

            std::array<internal::LocalCSR<Number>,
                       VectorizedArrayType::n_array_elements>
              c_pools;

            for (unsigned int v = 0; v < n_vectorization_actual; ++v)
              {
                unsigned int       index_indicators, next_index_indicators;
                const unsigned int n_components_read =
                  n_fe_components > 1 ? n_components : 1;

                index_indicators =
                  dof_info
                    .row_starts[(cell * n_vectorization + v) * n_fe_components +
                                first_selected_component]
                    .second;
                next_index_indicators =
                  dof_info
                    .row_starts[(cell * n_vectorization + v) * n_fe_components +
                                first_selected_component + 1]
                    .second;

                std::vector<std::tuple<unsigned int, unsigned int, Number>>
                  locally_relevant_constrains; // (constrained local index,
                                               // global index of dof which
                                               // constrains, weight)

                if (n_components == 1 || n_fe_components == 1)
                  {
                    AssertDimension(
                      n_components,
                      1); // TODO: currently no block vector supported

                    unsigned int ind_local = 0;
                    for (; index_indicators != next_index_indicators;
                         ++index_indicators)
                      {
                        const std::pair<unsigned short, unsigned short>
                          indicator =
                            dof_info.constraint_indicator[index_indicators];

                        for (unsigned int j = 0; j < indicator.first;
                             ++j, ++ind_local)
                          locally_relevant_constrains.emplace_back(
                            ind_local, dof_indices[v][j], 1.0);

                        dof_indices[v] += indicator.first;

                        const Number *data_val =
                          matrix_free.constraint_pool_begin(indicator.second);
                        const Number *end_pool =
                          matrix_free.constraint_pool_end(indicator.second);

                        for (; data_val != end_pool;
                             ++data_val, ++dof_indices[v])
                          locally_relevant_constrains.emplace_back(
                            ind_local, *dof_indices[v], *data_val);

                        ind_local++;
                      }

                    AssertIndexRange(ind_local, dofs_per_component + 1);

                    for (; ind_local < dofs_per_component;
                         ++dof_indices[v], ++ind_local)
                      locally_relevant_constrains.emplace_back(ind_local,
                                                               *dof_indices[v],
                                                               1.0);

                    plain_dof_indices[v] += dofs_per_component;
                  }
                else
                  {
                    // case with vector-valued finite elements where all
                    // components are included in one single vector. Assumption:
                    // first come all entries to the first component, then all
                    // entries to the second one, and so on. This is ensured by
                    // the way MatrixFree reads out the indices.
                    for (unsigned int comp = 0; comp < n_components; ++comp)
                      {
                        unsigned int ind_local = 0;

                        // check whether there is any constraint on the current
                        // cell
                        for (; index_indicators != next_index_indicators;
                             ++index_indicators)
                          {
                            const std::pair<unsigned short, unsigned short>
                              indicator =
                                dof_info.constraint_indicator[index_indicators];

                            // run through values up to next constraint
                            for (unsigned int j = 0; j < indicator.first;
                                 ++j, ++ind_local)
                              locally_relevant_constrains.emplace_back(
                                comp * dofs_per_component + ind_local,
                                dof_indices[v][j],
                                1.0);
                            dof_indices[v] += indicator.first;

                            const Number *data_val =
                              matrix_free.constraint_pool_begin(
                                indicator.second);
                            const Number *end_pool =
                              matrix_free.constraint_pool_end(indicator.second);

                            for (; data_val != end_pool;
                                 ++data_val, ++dof_indices[v])
                              locally_relevant_constrains.emplace_back(
                                comp * dofs_per_component + ind_local,
                                *dof_indices[v],
                                *data_val);

                            ind_local++;
                          }

                        AssertIndexRange(ind_local, dofs_per_component + 1);

                        // get the dof values past the last constraint
                        for (; ind_local < dofs_per_component;
                             ++dof_indices[v], ++ind_local)
                          locally_relevant_constrains.emplace_back(
                            comp * dofs_per_component + ind_local,
                            *dof_indices[v],
                            1.0);

                        if (comp + 1 < n_components)
                          {
                            next_index_indicators =
                              dof_info
                                .row_starts[(cell * n_vectorization + v) *
                                              n_fe_components +
                                            first_selected_component + comp + 2]
                                .second;
                          }
                      }
                  }


                // presort vector for transposed access
                std::sort(locally_relevant_constrains.begin(),
                          locally_relevant_constrains.end(),
                          [](const auto &a, const auto &b) {
                            if (std::get<1>(a) < std::get<1>(b))
                              return true;
                            return (std::get<1>(a) == std::get<1>(b)) &&
                                   (std::get<0>(a) < std::get<0>(b));
                          });

                // make sure that all entries are unique
                locally_relevant_constrains.erase(
                  unique(locally_relevant_constrains.begin(),
                         locally_relevant_constrains.end(),
                         [](const auto &a, const auto &b) {
                           return (std::get<1>(a) == std::get<1>(b)) &&
                                  (std::get<0>(a) == std::get<0>(b));
                         }),
                  locally_relevant_constrains.end());

                /*****************************************************************
                 * STEP 2: setup CSR storage of transposed locally relevant
                 *constraint matrix
                 ****************************************************************/
                auto &c_pool = c_pools[v];

                {
                  if (locally_relevant_constrains.size() > 0)
                    c_pool.row_lid_to_gid.emplace_back(
                      std::get<1>(locally_relevant_constrains.front()));
                  for (const auto &j : locally_relevant_constrains)
                    {
                      if (c_pool.row_lid_to_gid.back() != std::get<1>(j))
                        {
                          c_pool.row_lid_to_gid.push_back(std::get<1>(j));
                          c_pool.row.push_back(c_pool.val.size());
                        }

                      c_pool.col.emplace_back(std::get<0>(j));
                      c_pool.val.emplace_back(std::get<2>(j));
                    }

                  if (c_pool.val.size() > 0)
                    c_pool.row.push_back(c_pool.val.size());
                }
              }



            /*********************************************************************
             * STEP 3: compute element matrix element-by-element and apply
             *constraints
             ********************************************************************/

            // local storage: buffer so that we access the global vector once
            // note: may be larger then dofs_per_cell in the presence of
            // constraints!
            std::array<std::vector<Number>,
                       VectorizedArrayType::n_array_elements>
              diagonals_local_constrained;

            for (unsigned int v = 0; v < n_vectorization_actual; v++)
              diagonals_local_constrained[v].resize(
                c_pools[v].row_lid_to_gid.size(), Number(0.0));

            phi.reinit(cell);

            // loop over all columns of element stiffness matrix
            for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
              {
                // compute i-th column of element stiffness matrix:
                // this could be simply performed as done at the moment with
                // matrix-free operator evaluation applied to a ith-basis vector

                for (unsigned int j = 0; j < phi.dofs_per_cell; ++j)
                  phi.begin_dof_values()[j] = static_cast<Number>(i == j);

                local_vmult();

                const auto ith_column = phi.begin_dof_values();

                // apply local constraint matrix from left and from right:
                // loop over all rows of transposed constrained matrix
                for (unsigned int v = 0; v < n_vectorization_actual; v++)
                  {
                    const auto &c_pool = c_pools[v];

                    for (unsigned int j = 0; j < c_pool.row.size() - 1; j++)
                      {
                        // check if the result will be zero, so that we can skip
                        // the following computations -> binary search
                        const auto scale_iterator =
                          std::lower_bound(c_pool.col.begin() + c_pool.row[j],
                                           c_pool.col.begin() +
                                             c_pool.row[j + 1],
                                           i);

                        if (scale_iterator ==
                            c_pool.col.begin() + c_pool.row[j + 1])
                          continue;

                        if (*scale_iterator != i)
                          continue;

                        // apply constraint matrix from the left
                        Number temp = 0.0;
                        for (unsigned int k = c_pool.row[j];
                             k < c_pool.row[j + 1];
                             k++)
                          temp += c_pool.val[k] * ith_column[c_pool.col[k]][v];

                        // apply constraint matrix from the right
                        diagonals_local_constrained[v][j] +=
                          temp * c_pool.val[std::distance(c_pool.col.begin(),
                                                          scale_iterator)];
                      }
                  }
              }

            // assembly results: add into global vector
            for (unsigned int v = 0; v < n_vectorization_actual; v++)
              for (unsigned int j = 0; j < c_pools[v].row.size() - 1; j++)
                ::dealii::internal::vector_access_add(
                  diagonal_global,
                  c_pools[v].row_lid_to_gid[j],
                  diagonals_local_constrained[v][j]);
          }
      },
      diagonal_global,
      dummy,
      false);
  }

} // namespace MatrixFreeTools

DEAL_II_NAMESPACE_CLOSE


#endif
