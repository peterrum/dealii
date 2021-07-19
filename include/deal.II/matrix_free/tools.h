// ---------------------------------------------------------------------
//
// Copyright (C) 2020 - 2021 by the deal.II authors
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
  categorize_by_boundary_ids(const Triangulation<dim> &tria,
                             AdditionalData &          additional_data);

  /**
   * Compute the diagonal of a linear operator (@p diagonal_global), given
   * @p matrix_free and the local cell integral operation @p local_vmult. The
   * vector is initialized to the right size in the function.
   *
   * The parameters @p dof_no, @p quad_no, and @p first_selected_component are
   * passed to the constructor of the FEEvaluation that is internally set up.
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
    LinearAlgebra::distributed::Vector<Number> &        diagonal_global,
    const std::function<void(FEEvaluation<dim,
                                          fe_degree,
                                          n_q_points_1d,
                                          n_components,
                                          Number,
                                          VectorizedArrayType> &)> &local_vmult,
    const unsigned int                                              dof_no  = 0,
    const unsigned int                                              quad_no = 0,
    const unsigned int first_selected_component = 0);

  /**
   * Same as above but with a class and a function pointer.
   */
  template <typename CLASS,
            int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components,
            typename Number,
            typename VectorizedArrayType>
  void
  compute_diagonal(
    const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
    LinearAlgebra::distributed::Vector<Number> &        diagonal_global,
    void (CLASS::*cell_operation)(FEEvaluation<dim,
                                               fe_degree,
                                               n_q_points_1d,
                                               n_components,
                                               Number,
                                               VectorizedArrayType> &) const,
    const CLASS *      owning_class,
    const unsigned int dof_no                   = 0,
    const unsigned int quad_no                  = 0,
    const unsigned int first_selected_component = 0);


  /**
   * Compute the matrix representation of a linear operator (@p matrix), given
   * @p matrix_free and the local cell integral operation @p local_vmult.
   * Constrained entries on the diagonal are set to one.
   *
   * The parameters @p dof_no, @p quad_no, and @p first_selected_component are
   * passed to the constructor of the FEEvaluation that is internally set up.
   */
  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components,
            typename Number,
            typename VectorizedArrayType,
            typename MatrixType>
  void
  compute_matrix(
    const MatrixFree<dim, Number, VectorizedArrayType> &            matrix_free,
    const AffineConstraints<Number> &                               constraints,
    MatrixType &                                                    matrix,
    const std::function<void(FEEvaluation<dim,
                                          fe_degree,
                                          n_q_points_1d,
                                          n_components,
                                          Number,
                                          VectorizedArrayType> &)> &local_vmult,
    const unsigned int                                              dof_no  = 0,
    const unsigned int                                              quad_no = 0,
    const unsigned int first_selected_component = 0);

  /**
   * Same as above but with a class and a function pointer.
   */
  template <typename CLASS,
            int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components,
            typename Number,
            typename VectorizedArrayType,
            typename MatrixType>
  void
  compute_matrix(
    const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
    const AffineConstraints<Number> &                   constraints,
    MatrixType &                                        matrix,
    void (CLASS::*cell_operation)(FEEvaluation<dim,
                                               fe_degree,
                                               n_q_points_1d,
                                               n_components,
                                               Number,
                                               VectorizedArrayType> &) const,
    const CLASS *      owning_class,
    const unsigned int dof_no                   = 0,
    const unsigned int quad_no                  = 0,
    const unsigned int first_selected_component = 0);


  // implementations

#ifndef DOXYGEN

  template <int dim, typename AdditionalData>
  void
  categorize_by_boundary_ids(const Triangulation<dim> &tria,
                             AdditionalData &          additional_data)
  {
    // ... determine if we are on an active or a multigrid level
    const unsigned int level = additional_data.mg_level;
    const bool         is_mg = (level != numbers::invalid_unsigned_int);

    // ... create empty list for the category of each cell
    if (is_mg)
      additional_data.cell_vectorization_category.assign(
        std::distance(tria.begin(level), tria.end(level)), 0);
    else
      additional_data.cell_vectorization_category.assign(tria.n_active_cells(),
                                                         0);

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
      LocalCSR()
        : row{0}
      {}

      std::vector<unsigned int> row_lid_to_gid;
      std::vector<unsigned int> row;
      std::vector<unsigned int> col;
      std::vector<Number>       val;
    };

    template <int dim,
              int fe_degree,
              int n_q_points_1d,
              int n_components,
              typename Number,
              typename VectorizedArrayType>
    class ComputeDiagonalHelper
    {
    public:
      static const unsigned int n_lanes = VectorizedArrayType::size();

      ComputeDiagonalHelper(FEEvaluation<dim,
                                         fe_degree,
                                         n_q_points_1d,
                                         n_components,
                                         Number,
                                         VectorizedArrayType> &phi)
        : phi(phi)
      {}

      void
      reinit(const unsigned int cell)
      {
        this->phi.reinit(cell);
        // STEP 1: get relevant information from FEEvaluation
        const unsigned int first_selected_component =
          phi.get_first_selected_component();
        const auto &       dof_info        = phi.get_dof_info();
        const unsigned int n_fe_components = dof_info.start_components.back();
        const unsigned int dofs_per_component = phi.dofs_per_component;
        const auto &       matrix_free        = phi.get_matrix_free();

        const unsigned int n_lanes_filled =
          matrix_free.n_active_entries_per_cell_batch(cell);

        std::array<const unsigned int *, n_lanes> dof_indices{};
        {
          for (unsigned int v = 0; v < n_lanes_filled; ++v)
            dof_indices[v] =
              dof_info.dof_indices.data() +
              dof_info
                .row_starts[(cell * n_lanes + v) * n_fe_components +
                            first_selected_component]
                .first;
        }

        // STEP 2: setup CSR storage of transposed locally-relevant
        //   constraint matrix
        c_pools = std::array<internal::LocalCSR<Number>, n_lanes>();

        for (unsigned int v = 0; v < n_lanes_filled; ++v)
          {
            unsigned int index_indicators, next_index_indicators;

            index_indicators =
              dof_info
                .row_starts[(cell * n_lanes + v) * n_fe_components +
                            first_selected_component]
                .second;
            next_index_indicators =
              dof_info
                .row_starts[(cell * n_lanes + v) * n_fe_components +
                            first_selected_component + 1]
                .second;

            // STEP 2a: setup locally-relevant constraint matrix in a
            //   coordinate list (COO)
            std::vector<std::tuple<unsigned int, unsigned int, Number>>
              locally_relevant_constrains; // (constrained local index,
                                           // global index of dof which
                                           // constrains, weight)

            if (n_components == 1 || n_fe_components == 1)
              {
                AssertDimension(n_components,
                                1); // TODO: currently no block vector supported

                unsigned int ind_local = 0;
                for (; index_indicators != next_index_indicators;
                     ++index_indicators, ++ind_local)
                  {
                    const std::pair<unsigned short, unsigned short> indicator =
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

                    for (; data_val != end_pool; ++data_val, ++dof_indices[v])
                      locally_relevant_constrains.emplace_back(ind_local,
                                                               *dof_indices[v],
                                                               *data_val);
                  }

                AssertIndexRange(ind_local, dofs_per_component + 1);

                for (; ind_local < dofs_per_component;
                     ++dof_indices[v], ++ind_local)
                  locally_relevant_constrains.emplace_back(ind_local,
                                                           *dof_indices[v],
                                                           1.0);
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
                         ++index_indicators, ++ind_local)
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
                          matrix_free.constraint_pool_begin(indicator.second);
                        const Number *end_pool =
                          matrix_free.constraint_pool_end(indicator.second);

                        for (; data_val != end_pool;
                             ++data_val, ++dof_indices[v])
                          locally_relevant_constrains.emplace_back(
                            comp * dofs_per_component + ind_local,
                            *dof_indices[v],
                            *data_val);
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
                            .row_starts[(cell * n_lanes + v) * n_fe_components +
                                        first_selected_component + comp + 2]
                            .second;
                      }
                  }
              }

            // STEP 2b: sort and make unique

            // sort vector
            std::sort(locally_relevant_constrains.begin(),
                      locally_relevant_constrains.end(),
                      [](const auto &a, const auto &b) {
                        if (std::get<0>(a) < std::get<0>(b))
                          return true;
                        return (std::get<0>(a) == std::get<0>(b)) &&
                               (std::get<1>(a) < std::get<1>(b));
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

            // STEP 2b: apply hanging-node constraints
            if (dof_info.component_masks.size() > 0)
              {
#  if false
                std::cout << "A" << std::endl;
                for (auto i : locally_relevant_constrains)
                  std::cout << std::get<0>(i) << " " << std::get<1>(i) << " "
                            << std::get<2>(i) << " " << std::endl;
                std::cout << std::endl << std::endl << std::endl;
#  endif

                const auto mask =
                  dof_info
                    .component_masks[(cell * n_lanes + v) * n_fe_components +
                                     first_selected_component];

                if (mask != 0)
                  {
                    const unsigned int n_points_1d =
                      this->phi.get_shape_info().data.front().n_q_points_1d;
                    const auto &weight = this->phi.get_shape_info()
                                           .data.front()
                                           .subface_interpolation_matrix;

                    const auto is_set = [](const unsigned int a,
                                           const unsigned int b) {
                      return (a & b) == b;
                    };

                    const auto not_set = [](const unsigned int a,
                                            const unsigned int b) {
                      return (a & b) == 0;
                    };

                    (void)not_set;

                    std::vector<std::tuple<unsigned int, unsigned int, Number>>
                      locally_relevant_constrains_hn;

                    // 1) collect hanging-node constraints for cell assuming
                    // scalar finite element
                    if (dim == 2)
                      {
                        // helper function to process faces
                        const auto process = [&](const auto face,
                                                 const auto type) {
                          const std::array<
                            std::array<std::pair<unsigned int, unsigned int>,
                                       2>,
                            2>
                            start_and_strides{
                              {{{{0, n_points_1d},
                                 {n_points_1d - 1, n_points_1d}}},
                               {{{0, 1},
                                 {n_points_1d * n_points_1d - n_points_1d,
                                  1}}}}};


                          const auto ss = start_and_strides[face / 2][face % 2];

                          for (unsigned int h = 0; h < n_points_1d; ++h)
                            for (unsigned int k = 0; k < n_points_1d; ++k)
                              {
                                const unsigned int index_h =
                                  ss.first + ss.second * h;
                                const unsigned int index_k =
                                  ss.first + ss.second * k;
                                const unsigned int index_w =
                                  n_points_1d *
                                    (type ? h : (n_points_1d - 1 - h)) +
                                  (type ? k : (n_points_1d - 1 - k));

                                if (0.0 < weight[index_w][v] &&
                                    weight[index_w][v] < 1.0)
                                  locally_relevant_constrains_hn.emplace_back(
                                    index_h, index_k, weight[index_w][v]);
                              }
                        };

                        // direction 0:
                        if (mask & dealii::internal::constr_face_y)
                          {
                            const bool not_flipped =
                              mask & dealii::internal::constr_type_x;
                            if (is_set(mask, dealii::internal::constr_type_y))
                              process(2, not_flipped);
                            else
                              process(3, not_flipped);
                          }

                        // direction 1:
                        if (mask & dealii::internal::constr_face_x)
                          {
                            const bool not_flipped =
                              mask & dealii::internal::constr_type_y;
                            if (is_set(mask, dealii::internal::constr_type_x))
                              process(0, not_flipped);
                            else
                              process(1, not_flipped);
                          }
                      }
                    else if (dim == 3)
                      {
                        AlignedVector<VectorizedArrayType> values_dofs(
                          dofs_per_component);

                        std::array<unsigned int, VectorizedArrayType::size()>
                          constraint_mask;
                        constraint_mask[0] = mask;

                        for (unsigned int i = 0; i < dofs_per_component; ++i)
                          {
                            for (unsigned int j = 0; j < dofs_per_component;
                                 ++j)
                              values_dofs[j] = static_cast<Number>(i == j);

                            dealii::internal::FEEvaluationHangingNodesFactory<
                              dim,
                              Number,
                              VectorizedArrayType>::apply(1,
                                                          phi.get_shape_info()
                                                            .data.front()
                                                            .fe_degree,
                                                          phi,
                                                          false,
                                                          constraint_mask,
                                                          values_dofs.data());

                            for (unsigned int j = 0; j < dofs_per_component;
                                 ++j)
                              if (0.0 < values_dofs[j][0] &&
                                  values_dofs[j][0] < 1.0)
                                locally_relevant_constrains_hn.emplace_back(
                                  j, i, values_dofs[j][0]);
                          }
                      }
                    else
                      {
                        const unsigned int p0 = 0;
                        const unsigned int p1 = n_points_1d - 1;
                        const unsigned int p2 =
                          n_points_1d * n_points_1d - n_points_1d;
                        const unsigned int p3 = n_points_1d * n_points_1d - 1;
                        const unsigned int p4 =
                          n_points_1d * n_points_1d * n_points_1d -
                          n_points_1d * n_points_1d;
                        const unsigned int p5 =
                          n_points_1d * n_points_1d * n_points_1d -
                          n_points_1d * n_points_1d + n_points_1d - 1;
                        const unsigned int p6 =
                          n_points_1d * n_points_1d * n_points_1d - n_points_1d;

                        const auto process_edge = [&](const auto face,
                                                      const auto type) {
                          std::array<std::pair<unsigned int, unsigned int>, 12>
                            start_and_strides{{
                              {p0, n_points_1d},               // 0
                              {p2, n_points_1d},               // 1
                              {p4, 1},                         // 2
                              {p6, 1},                         // 3
                              {p0, n_points_1d},               // 4
                              {p1, n_points_1d},               // 5
                              {p4, 1},                         // 6
                              {p5, 1},                         // 7
                              {p0, n_points_1d * n_points_1d}, // 8
                              {p1, n_points_1d * n_points_1d}, // 9
                              {p2, n_points_1d * n_points_1d}, // 10
                              {p3, n_points_1d * n_points_1d}  // 11
                            }};


                          const auto ss = start_and_strides[face];

                          for (unsigned int h = 0; h < n_points_1d; ++h)
                            for (unsigned int k = 0; k < n_points_1d; ++k)
                              {
                                const unsigned int index_h =
                                  ss.first + ss.second * h;
                                const unsigned int index_k =
                                  ss.first + ss.second * k;
                                const unsigned int index_w =
                                  n_points_1d *
                                    (type ? h : (n_points_1d - 1 - h)) +
                                  (type ? k : (n_points_1d - 1 - k));

                                if (0.0 < weight[index_w][v] &&
                                    weight[index_w][v] < 1.0)
                                  locally_relevant_constrains_hn.emplace_back(
                                    index_h, index_k, weight[index_w][v]);
                              }
                        };

                        const auto process_face = [&](const auto face,
                                                      const auto type_1,
                                                      const auto type_2) {
                          std::array<std::array<unsigned int, 3>, 12>
                            start_and_strides{{
                              {{p0,
                                n_points_1d,
                                n_points_1d * n_points_1d}}, // 0
                              {{p1,
                                n_points_1d,
                                n_points_1d * n_points_1d}},        // 1
                              {{p0, 1, n_points_1d}},               // 2
                              {{p2, 1, n_points_1d}},               // 3
                              {{p0, 1, n_points_1d * n_points_1d}}, // 4
                              {{p4, 1, n_points_1d * n_points_1d}}  // 5
                            }};


                          const auto ss = start_and_strides[face];

                          for (unsigned int h1 = 1; h1 < n_points_1d - 1; ++h1)
                            for (unsigned int h2 = 1; h2 < n_points_1d - 1;
                                 ++h2)
                              for (unsigned int k1 = 0; k1 < n_points_1d; ++k1)
                                for (unsigned int k2 = 0; k2 < n_points_1d;
                                     ++k2)
                                  {
                                    const unsigned int index_h =
                                      ss[0] + ss[2] * h1 + ss[2] * h2;
                                    const unsigned int index_k =
                                      ss[0] + ss[1] * k1 + ss[2] * k2;
                                    const unsigned int index_w1 =
                                      n_points_1d *
                                        (type_1 ? h1 : (n_points_1d - 1 - h1)) +
                                      (type_1 ? k1 : (n_points_1d - 1 - k1));
                                    const unsigned int index_w2 =
                                      n_points_1d *
                                        (type_2 ? h2 : (n_points_1d - 1 - h2)) +
                                      (type_2 ? k2 : (n_points_1d - 1 - k2));

                                    if ((0.0 < weight[index_w1][v] &&
                                         weight[index_w1][v] < 1.0) &&
                                        (0.0 < weight[index_w2][v] &&
                                         weight[index_w2][v] < 1.0))
                                      locally_relevant_constrains_hn
                                        .emplace_back(index_h,
                                                      index_k,
                                                      weight[index_w1][v] *
                                                        weight[index_w2][v]);
                                  }
                        };

                        const bool is_face_0 =
                          is_set(mask, dealii::internal::constr_face_x) &&
                          is_set(mask, dealii::internal::constr_type_x);
                        const bool is_face_1 =
                          is_set(mask, dealii::internal::constr_face_x) &&
                          not_set(mask, dealii::internal::constr_type_x);
                        const bool is_face_2 =
                          is_set(mask, dealii::internal::constr_face_y) &&
                          is_set(mask, dealii::internal::constr_type_y);
                        const bool is_face_3 =
                          is_set(mask, dealii::internal::constr_face_y) &&
                          not_set(mask, dealii::internal::constr_type_y);
                        const bool is_face_4 =
                          is_set(mask, dealii::internal::constr_face_z) &&
                          is_set(mask, dealii::internal::constr_type_z);
                        const bool is_face_5 =
                          is_set(mask, dealii::internal::constr_face_z) &&
                          not_set(mask, dealii::internal::constr_type_z);

                        const bool is_edge_2 =
                          is_set(mask, dealii::internal::constr_edge_yz) &&
                          is_set(mask, dealii::internal::constr_type_y) &&
                          is_set(mask, dealii::internal::constr_type_z);
                        const bool is_edge_3 =
                          is_set(mask, dealii::internal::constr_edge_yz) &&
                          not_set(mask, dealii::internal::constr_type_y) &&
                          is_set(mask, dealii::internal::constr_type_z);
                        const bool is_edge_6 =
                          is_set(mask, dealii::internal::constr_edge_yz) &&
                          is_set(mask, dealii::internal::constr_type_y) &&
                          not_set(mask, dealii::internal::constr_type_z);
                        const bool is_edge_7 =
                          is_set(mask, dealii::internal::constr_edge_yz) &&
                          not_set(mask, dealii::internal::constr_type_y) &&
                          not_set(mask, dealii::internal::constr_type_z);

                        const bool is_edge_0 =
                          is_set(mask, dealii::internal::constr_edge_zx) &&
                          is_set(mask, dealii::internal::constr_type_x) &&
                          is_set(mask, dealii::internal::constr_type_z);
                        const bool is_edge_1 =
                          is_set(mask, dealii::internal::constr_edge_zx) &&
                          not_set(mask, dealii::internal::constr_type_x) &&
                          is_set(mask, dealii::internal::constr_type_z);
                        const bool is_edge_4 =
                          is_set(mask, dealii::internal::constr_edge_zx) &&
                          is_set(mask, dealii::internal::constr_type_x) &&
                          not_set(mask, dealii::internal::constr_type_z);
                        const bool is_edge_5 =
                          is_set(mask, dealii::internal::constr_edge_zx) &&
                          not_set(mask, dealii::internal::constr_type_x) &&
                          not_set(mask, dealii::internal::constr_type_z);

                        const bool is_edge_8 =
                          is_set(mask, dealii::internal::constr_edge_xy) &&
                          is_set(mask, dealii::internal::constr_type_x) &&
                          is_set(mask, dealii::internal::constr_type_y);
                        const bool is_edge_9 =
                          is_set(mask, dealii::internal::constr_edge_xy) &&
                          not_set(mask, dealii::internal::constr_type_x) &&
                          is_set(mask, dealii::internal::constr_type_y);
                        const bool is_edge_10 =
                          is_set(mask, dealii::internal::constr_edge_xy) &&
                          is_set(mask, dealii::internal::constr_type_x) &&
                          not_set(mask, dealii::internal::constr_type_y);
                        const bool is_edge_11 =
                          is_set(mask, dealii::internal::constr_edge_xy) &&
                          not_set(mask, dealii::internal::constr_type_x) &&
                          not_set(mask, dealii::internal::constr_type_y);

                        if (is_face_0)
                          process_face(0,
                                       mask & dealii::internal::constr_type_y,
                                       mask & dealii::internal::constr_type_z);
                        if (is_face_1)
                          process_face(1,
                                       mask & dealii::internal::constr_type_y,
                                       mask & dealii::internal::constr_type_z);

                        if (is_face_2)
                          process_face(2,
                                       mask & dealii::internal::constr_type_x,
                                       mask & dealii::internal::constr_type_z);
                        if (is_face_3)
                          process_face(3,
                                       mask & dealii::internal::constr_type_x,
                                       mask & dealii::internal::constr_type_z);

                        if (is_face_4)
                          process_face(4,
                                       mask & dealii::internal::constr_type_x,
                                       mask & dealii::internal::constr_type_y);
                        if (is_face_5)
                          process_face(5,
                                       mask & dealii::internal::constr_type_x,
                                       mask & dealii::internal::constr_type_y);

                        {
                          const bool not_flipped =
                            mask & dealii::internal::constr_type_x;

                          // ... edges
                          if (is_face_2 || is_face_4 || is_edge_2)
                            process_edge(2, not_flipped);
                          if (is_face_3 || is_face_4 || is_edge_3)
                            process_edge(3, not_flipped);
                          if (is_face_2 || is_face_5 || is_edge_6)
                            process_edge(6, not_flipped);
                          if (is_face_3 || is_face_5 || is_edge_7)
                            process_edge(7, not_flipped);
                        }

                        // direction 1:
                        {
                          const bool not_flipped =
                            mask & dealii::internal::constr_type_y;

                          if (is_face_0 || is_face_4 || is_edge_0)
                            process_edge(0, not_flipped);
                          if (is_face_1 || is_face_4 || is_edge_1)
                            process_edge(1, not_flipped);
                          if (is_face_0 || is_face_5 || is_edge_4)
                            process_edge(4, not_flipped);
                          if (is_face_1 || is_face_5 || is_edge_5)
                            process_edge(5, not_flipped);
                        }

                        // direction 2:
                        {
                          const bool not_flipped =
                            mask & dealii::internal::constr_type_z;

                          if (is_face_0 || is_face_2 || is_edge_8)
                            process_edge(8, not_flipped);
                          if (is_face_1 || is_face_2 || is_edge_9)
                            process_edge(9, not_flipped);
                          if (is_face_0 || is_face_3 || is_edge_10)
                            process_edge(10, not_flipped);
                          if (is_face_1 || is_face_3 || is_edge_11)
                            process_edge(11, not_flipped);
                        }
                      }

                    std::sort(locally_relevant_constrains_hn.begin(),
                              locally_relevant_constrains_hn.end(),
                              [](const auto &a, const auto &b) {
                                if (std::get<0>(a) < std::get<0>(b))
                                  return true;
                                return (std::get<0>(a) == std::get<0>(b)) &&
                                       (std::get<1>(a) < std::get<1>(b));
                              });

                    // 2) extend for multiple components
                    std::vector<std::tuple<unsigned int, unsigned int, Number>>
                      locally_relevant_constrains_hn_temp;

                    for (unsigned int c = 0; c < n_components; ++c)
                      for (auto i : locally_relevant_constrains_hn)
                        locally_relevant_constrains_hn_temp.emplace_back(
                          std::get<0>(i) + c * dofs_per_component,
                          std::get<1>(i) + c * dofs_per_component,
                          std::get<2>(i));

                    locally_relevant_constrains_hn =
                      locally_relevant_constrains_hn_temp;

#  if false
                    std::cout << "BBB" << std::endl;
                    for (auto i : locally_relevant_constrains_hn_temp)
                      std::cout << std::get<0>(i) << " " << std::get<1>(i)
                                << " " << std::get<2>(i) << " " << std::endl;
                    std::cout << std::endl;
                    
                    std::cout << "B" << std::endl;
                    for (auto i : locally_relevant_constrains_hn)
                      std::cout << std::get<0>(i) << " " << std::get<1>(i)
                                << " " << std::get<2>(i) << " " << std::endl;
                    std::cout << std::endl;

                    std::cout << "C" << std::endl;
                    for (auto i : locally_relevant_constrains)
                      std::cout << std::get<0>(i) << " " << std::get<1>(i)
                                << " " << std::get<2>(i) << " " << std::endl;
                    std::cout << std::endl << std::endl << std::endl;
#  endif


                    // 3) perform vmult with other constraints
                    std::vector<std::tuple<unsigned int, unsigned int, Number>>
                      locally_relevant_constrains_temp;

                    for (unsigned int i = 0;
                         i < dofs_per_component * n_components;
                         ++i)
                      {
                        const auto lower_bound_fu = [](const auto &a,
                                                       const auto &b) {
                          return std::get<0>(a) < b;
                        };

                        const auto upper_bound_fu = [](const auto &a,
                                                       const auto &b) {
                          return a < std::get<0>(b);
                        };

                        const auto i_begin = std::lower_bound(
                          locally_relevant_constrains_hn.begin(),
                          locally_relevant_constrains_hn.end(),
                          i,
                          lower_bound_fu);
                        const auto i_end = std::upper_bound(
                          locally_relevant_constrains_hn.begin(),
                          locally_relevant_constrains_hn.end(),
                          i,
                          upper_bound_fu);

                        if (i_begin == i_end)
                          {
                            // dof is not constrained by hanging-node constraint
                            // (identity matrix): simply copy constraints
                            const auto j_begin = std::lower_bound(
                              locally_relevant_constrains.begin(),
                              locally_relevant_constrains.end(),
                              i,
                              lower_bound_fu);
                            const auto j_end = std::upper_bound(
                              locally_relevant_constrains.begin(),
                              locally_relevant_constrains.end(),
                              i,
                              upper_bound_fu);

                            for (auto v = j_begin; v != j_end; ++v)
                              locally_relevant_constrains_temp.emplace_back(*v);
                          }
                        else
                          {
                            // dof is constrained: build transitive closure
                            for (auto v0 = i_begin; v0 != i_end; ++v0)
                              {
                                const auto j_begin = std::lower_bound(
                                  locally_relevant_constrains.begin(),
                                  locally_relevant_constrains.end(),
                                  std::get<1>(*v0),
                                  lower_bound_fu);
                                const auto j_end = std::upper_bound(
                                  locally_relevant_constrains.begin(),
                                  locally_relevant_constrains.end(),
                                  std::get<1>(*v0),
                                  upper_bound_fu);

                                for (auto v1 = j_begin; v1 != j_end; ++v1)
                                  locally_relevant_constrains_temp.emplace_back(
                                    std::get<0>(*v0),
                                    std::get<1>(*v1),
                                    std::get<2>(*v0) * std::get<2>(*v1));
                              }
                          }
                      }

#  if false
                    std::cout << "D" << std::endl;
                    for (auto i : locally_relevant_constrains_temp)
                      std::cout << std::get<0>(i) << " " << std::get<1>(i)
                                << " " << std::get<2>(i) << " " << std::endl;
                    std::cout << std::endl << std::endl << std::endl;
#  endif

                    locally_relevant_constrains =
                      locally_relevant_constrains_temp;
                  }
              }

            // STEP 2d: transpose COO
            std::sort(locally_relevant_constrains.begin(),
                      locally_relevant_constrains.end(),
                      [](const auto &a, const auto &b) {
                        if (std::get<1>(a) < std::get<1>(b))
                          return true;
                        return (std::get<1>(a) == std::get<1>(b)) &&
                               (std::get<0>(a) < std::get<0>(b));
                      });

#  if false
            std::cout << "OOOO" << std::endl;
            for (auto i : locally_relevant_constrains)
              std::cout << std::get<0>(i) << " " << std::get<1>(i) << " "
                        << std::get<2>(i) << " " << std::endl;
            std::cout << std::endl << std::endl << std::endl;
#  endif

            // STEP 2e: translate COO to CRS
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
        // STEP 3: compute element matrix A_e, apply
        //   locally-relevant constraints C_e^T * A_e * C_e, and get the
        //   the diagonal entry
        //     (C_e^T * A_e * C_e)(i,i)
        //   or
        //     C_e^T(i,:) * A_e * C_e(:,i).
        //
        //   Since, we compute the element matrix column-by-column and as a
        //   result never actually have the full element matrix, we actually
        //   perform following steps:
        //    1) loop over all columns of the element matrix
        //     a) compute column i
        //     b) compute for each j (rows of C_e^T):
        //          (C_e^T(j,:) * A_e(:,i)) * C_e(i,j)
        //       or
        //          (C_e^T(j,:) * A_e(:,i)) * C_e^T(j,i)
        //       This gives a contribution the j-th entry of the
        //       locally-relevant diagonal and comprises the multiplication
        //       by the locally-relevant constraint matrix from the left and
        //       the right. There is no contribution to the j-th vector
        //       entry if the j-th row of C_e^T is empty or C_e^T(j,i) is
        //       zero.

        // set size locally-relevant diagonal
        for (unsigned int v = 0; v < n_lanes_filled; ++v)
          diagonals_local_constrained[v].assign(
            c_pools[v].row_lid_to_gid.size(), Number(0.0));
      }

      void
      prepare_basis_vector(const unsigned int i)
      {
        this->i = i;

        // compute i-th column of element stiffness matrix:
        // this could be simply performed as done at the moment with
        // matrix-free operator evaluation applied to a ith-basis vector
        for (unsigned int j = 0; j < phi.dofs_per_cell; ++j)
          phi.begin_dof_values()[j] = static_cast<Number>(i == j);
      }

      void
      submit()
      {
        const auto ith_column = phi.begin_dof_values();

        // apply local constraint matrix from left and from right:
        // loop over all rows of transposed constrained matrix
        for (unsigned int v = 0;
             v < phi.get_matrix_free().n_active_entries_per_cell_batch(
                   phi.get_current_cell_index());
             ++v)
          {
            const auto &c_pool = c_pools[v];

            for (unsigned int j = 0; j < c_pool.row.size() - 1; ++j)
              {
                // check if the result will be zero, so that we can skip
                // the following computations -> binary search
                const auto scale_iterator =
                  std::lower_bound(c_pool.col.begin() + c_pool.row[j],
                                   c_pool.col.begin() + c_pool.row[j + 1],
                                   i);

                // explanation: j-th row of C_e^T is empty (see above)
                if (scale_iterator == c_pool.col.begin() + c_pool.row[j + 1])
                  continue;

                // explanation: C_e^T(j,i) is zero (see above)
                if (*scale_iterator != i)
                  continue;

                // apply constraint matrix from the left
                Number temp = 0.0;
                for (unsigned int k = c_pool.row[j]; k < c_pool.row[j + 1]; ++k)
                  temp += c_pool.val[k] * ith_column[c_pool.col[k]][v];

                // apply constraint matrix from the right
                diagonals_local_constrained[v][j] +=
                  temp *
                  c_pool.val[std::distance(c_pool.col.begin(), scale_iterator)];
              }
          }
      }

      void
      distribute_local_to_global(
        LinearAlgebra::distributed::Vector<Number> &diagonal_global)
      {
        // STEP 4: assembly results: add into global vector
        for (unsigned int v = 0;
             v < phi.get_matrix_free().n_active_entries_per_cell_batch(
                   phi.get_current_cell_index());
             ++v)
          for (unsigned int j = 0; j < c_pools[v].row.size() - 1; ++j)
            ::dealii::internal::vector_access_add(
              diagonal_global,
              c_pools[v].row_lid_to_gid[j],
              diagonals_local_constrained[v][j]);
      }

    private:
      FEEvaluation<dim,
                   fe_degree,
                   n_q_points_1d,
                   n_components,
                   Number,
                   VectorizedArrayType> &phi;

      unsigned int i;

      std::array<internal::LocalCSR<Number>, n_lanes> c_pools;

      // local storage: buffer so that we access the global vector once
      // note: may be larger then dofs_per_cell in the presence of
      // constraints!
      std::array<std::vector<Number>, n_lanes> diagonals_local_constrained;
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
    LinearAlgebra::distributed::Vector<Number> &        diagonal_global,
    const std::function<void(FEEvaluation<dim,
                                          fe_degree,
                                          n_q_points_1d,
                                          n_components,
                                          Number,
                                          VectorizedArrayType> &)> &local_vmult,
    const unsigned int                                              dof_no,
    const unsigned int                                              quad_no,
    const unsigned int first_selected_component)
  {
    using VectorType = LinearAlgebra::distributed::Vector<Number>;

    // initialize vector
    matrix_free.initialize_dof_vector(diagonal_global, dof_no);

    int dummy = 0;

    matrix_free.template cell_loop<VectorType, int>(
      [&](const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
          LinearAlgebra::distributed::Vector<Number> &        diagonal_global,
          const int &,
          const std::pair<unsigned int, unsigned int> &range) mutable {
        FEEvaluation<dim,
                     fe_degree,
                     n_q_points_1d,
                     n_components,
                     Number,
                     VectorizedArrayType>
          phi(matrix_free, range, dof_no, quad_no, first_selected_component);

        internal::ComputeDiagonalHelper<dim,
                                        fe_degree,
                                        n_q_points_1d,
                                        n_components,
                                        Number,
                                        VectorizedArrayType>
          helper(phi);

        for (unsigned int cell = range.first; cell < range.second; ++cell)
          {
            helper.reinit(cell);

            for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
              {
                helper.prepare_basis_vector(i);
                local_vmult(phi);
                helper.submit();
              }

            helper.distribute_local_to_global(diagonal_global);
          }
      },
      diagonal_global,
      dummy,
      false);
  }

  template <typename CLASS,
            int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components,
            typename Number,
            typename VectorizedArrayType>
  void
  compute_diagonal(
    const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
    LinearAlgebra::distributed::Vector<Number> &        diagonal_global,
    void (CLASS::*cell_operation)(FEEvaluation<dim,
                                               fe_degree,
                                               n_q_points_1d,
                                               n_components,
                                               Number,
                                               VectorizedArrayType> &) const,
    const CLASS *      owning_class,
    const unsigned int dof_no,
    const unsigned int quad_no,
    const unsigned int first_selected_component)
  {
    compute_diagonal<dim,
                     fe_degree,
                     n_q_points_1d,
                     n_components,
                     Number,
                     VectorizedArrayType>(
      matrix_free,
      diagonal_global,
      [&](auto &feeval) { (owning_class->*cell_operation)(feeval); },
      dof_no,
      quad_no,
      first_selected_component);
  }

  namespace internal
  {
    /**
     * If value type of matrix and constrains equals, return a reference
     * to the given AffineConstraint instance.
     */
    template <typename MatrixType,
              typename Number,
              typename std::enable_if<std::is_same<
                typename std::remove_const<typename std::remove_reference<
                  typename MatrixType::value_type>::type>::type,
                typename std::remove_const<typename std::remove_reference<
                  Number>::type>::type>::value>::type * = nullptr>
    const AffineConstraints<typename MatrixType::value_type> &
    create_new_affine_constraints_if_needed(
      const MatrixType &,
      const AffineConstraints<Number> &constraints,
      std::unique_ptr<AffineConstraints<typename MatrixType::value_type>> &)
    {
      return constraints;
    }

    /**
     * If value type of matrix and constrains do not equal, a new
     * AffineConstraint instance with the value type of the matrix is
     * created and a reference to it is returned.
     */
    template <typename MatrixType,
              typename Number,
              typename std::enable_if<!std::is_same<
                typename std::remove_const<typename std::remove_reference<
                  typename MatrixType::value_type>::type>::type,
                typename std::remove_const<typename std::remove_reference<
                  Number>::type>::type>::value>::type * = nullptr>
    const AffineConstraints<typename MatrixType::value_type> &
    create_new_affine_constraints_if_needed(
      const MatrixType &,
      const AffineConstraints<Number> &constraints,
      std::unique_ptr<AffineConstraints<typename MatrixType::value_type>>
        &new_constraints)
    {
      new_constraints =
        std::make_unique<AffineConstraints<typename MatrixType::value_type>>();
      new_constraints->copy_from(constraints);

      return *new_constraints;
    }
  } // namespace internal

  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components,
            typename Number,
            typename VectorizedArrayType,
            typename MatrixType>
  void
  compute_matrix(
    const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
    const AffineConstraints<Number> &                   constraints_in,
    MatrixType &                                        matrix,
    const std::function<void(FEEvaluation<dim,
                                          fe_degree,
                                          n_q_points_1d,
                                          n_components,
                                          Number,
                                          VectorizedArrayType> &)> &local_vmult,
    const unsigned int                                              dof_no,
    const unsigned int                                              quad_no,
    const unsigned int first_selected_component)
  {
    std::unique_ptr<AffineConstraints<typename MatrixType::value_type>>
                                                              constraints_for_matrix;
    const AffineConstraints<typename MatrixType::value_type> &constraints =
      internal::create_new_affine_constraints_if_needed(matrix,
                                                        constraints_in,
                                                        constraints_for_matrix);

    matrix_free.template cell_loop<MatrixType, MatrixType>(
      [&](const auto &, auto &dst, const auto &, const auto range) {
        FEEvaluation<dim,
                     fe_degree,
                     n_q_points_1d,
                     n_components,
                     Number,
                     VectorizedArrayType>
          integrator(
            matrix_free, range, dof_no, quad_no, first_selected_component);

        unsigned int const dofs_per_cell = integrator.dofs_per_cell;

        std::vector<types::global_dof_index> dof_indices(dofs_per_cell);
        std::vector<types::global_dof_index> dof_indices_mf(dofs_per_cell);

        std::array<FullMatrix<typename MatrixType::value_type>,
                   VectorizedArrayType::size()>
          matrices;

        std::fill_n(matrices.begin(),
                    VectorizedArrayType::size(),
                    FullMatrix<typename MatrixType::value_type>(dofs_per_cell,
                                                                dofs_per_cell));

        const auto lexicographic_numbering =
          matrix_free
            .get_shape_info(dof_no,
                            quad_no,
                            first_selected_component,
                            integrator.get_active_fe_index(),
                            integrator.get_active_quadrature_index())
            .lexicographic_numbering;

        for (auto cell = range.first; cell < range.second; ++cell)
          {
            integrator.reinit(cell);

            unsigned int const n_filled_lanes =
              matrix_free.n_active_entries_per_cell_batch(cell);

            for (unsigned int v = 0; v < n_filled_lanes; ++v)
              matrices[v] = 0.0;

            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              {
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  integrator.begin_dof_values()[i] =
                    static_cast<Number>(i == j);

                local_vmult(integrator);

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  for (unsigned int v = 0; v < n_filled_lanes; ++v)
                    matrices[v](i, j) = integrator.begin_dof_values()[i][v];
              }

            for (unsigned int v = 0; v < n_filled_lanes; ++v)
              {
                const auto cell_v =
                  matrix_free.get_cell_iterator(cell, v, dof_no);

                if (matrix_free.get_mg_level() != numbers::invalid_unsigned_int)
                  cell_v->get_mg_dof_indices(dof_indices);
                else
                  cell_v->get_dof_indices(dof_indices);

                for (unsigned int j = 0; j < dof_indices.size(); ++j)
                  dof_indices_mf[j] = dof_indices[lexicographic_numbering[j]];

                constraints.distribute_local_to_global(matrices[v],
                                                       dof_indices_mf,
                                                       dst);
              }
          }
      },
      matrix,
      matrix);

    matrix.compress(VectorOperation::add);
  }

  template <typename CLASS,
            int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components,
            typename Number,
            typename VectorizedArrayType,
            typename MatrixType>
  void
  compute_matrix(
    const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
    const AffineConstraints<Number> &                   constraints,
    MatrixType &                                        matrix,
    void (CLASS::*cell_operation)(FEEvaluation<dim,
                                               fe_degree,
                                               n_q_points_1d,
                                               n_components,
                                               Number,
                                               VectorizedArrayType> &) const,
    const CLASS *      owning_class,
    const unsigned int dof_no,
    const unsigned int quad_no,
    const unsigned int first_selected_component)
  {
    compute_matrix<dim,
                   fe_degree,
                   n_q_points_1d,
                   n_components,
                   Number,
                   VectorizedArrayType,
                   MatrixType>(matrix_free,
                               constraints,
                               matrix,
                               [&](auto &feeval) {
                                 (owning_class->*cell_operation)(feeval);
                               },
                               dof_no,
                               quad_no,
                               first_selected_component);
  }

#endif // DOXYGEN

} // namespace MatrixFreeTools

DEAL_II_NAMESPACE_CLOSE


#endif
